"""Run the registered tests (preregistration §5–7) and write machine-readable results."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from landblind.climatology import in_season
from landblind.config import Config
from landblind.mirror import manifest_digest
from landblind.report import render
from landblind.stats import (
    Sample,
    SlopeSums,
    block_counts,
    dk_se_mean_slope,
    fe_groups,
    holm,
    slope_sums,
    summarize,
    verdict,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------
# Inputs aligned to initialisation dates
# --------------------------------------------------------------------------------------------

@dataclass
class PairData:
    name: str
    dates: pd.DatetimeIndex                 # 00 UTC initialisation dates
    e: dict[str, np.ndarray]                # errors (init, lead, cell)
    s: np.ndarray                           # soil-moisture anomaly on D-1 (init, cell)
    s_l1: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    season: np.ndarray                      # in-season mask (init, cell)
    usable: np.ndarray                      # afternoon target within 3 h (cell)
    night_usable: np.ndarray
    weight: np.ndarray
    regime: np.ndarray
    decile: np.ndarray
    continent: np.ndarray
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def month(self) -> np.ndarray:
        return np.asarray(self.dates.month)


def load_pair(cfg: Config, cov: xr.Dataset, name: str) -> PairData:
    panel = xr.open_dataset(cfg.processed / f"panel_{name}.nc")
    dates = pd.DatetimeIndex(panel.init.values)
    prev = dates - pd.Timedelta(days=1)
    e = {}
    for when in ("afternoon", "night"):
        truth = panel[f"truth_{when}"].values
        alt = panel[f"alt_truth_{when}"].values
        ai, ifs = panel[f"ai_{when}"].values, panel[f"ifs_{when}"].values
        e[f"ai_{when}"] = truth - ai
        e[f"ifs_{when}"] = truth - ifs
        e[f"ai_{when}_alt"] = alt - ai
        e[f"ifs_{when}_alt"] = alt - ifs
    start = cov.season_start_month.values.astype(int)
    extra = {"truth_level": panel["truth_afternoon"].values,
             "ai_level": panel["ai_afternoon"].values,
             "ifs_level": panel["ifs_afternoon"].values}
    if "ai_z500_day5" in panel:
        extra["ez500_ai"] = panel.truth_z500_day5.values - panel.ai_z500_day5.values
    return PairData(
        name=name,
        dates=dates,
        e=e,
        s=cov.s.sel(day=prev.values).values,
        s_l1=cov.s_l1.sel(day=prev.values).values,
        a1=cov.tmax_anom.sel(day=prev.values).values,
        a2=cov.t2m_00utc_anom.sel(day6=dates.values).values,
        season=in_season(np.asarray(dates.month), start, cfg["warm_season_months"]),
        usable=panel.afternoon_usable.values.astype(bool),
        night_usable=panel.night_usable.values.astype(bool),
        weight=cov.weight.values,
        regime=cov.regime.values.astype(str),
        decile=cov.tci_decile.values.astype(int),
        continent=cov.continent.values.astype(str),
        extra=extra,
    )


def all_finite(*arrays: np.ndarray) -> np.ndarray:
    """(init, cell) mask: every array finite (over all lead days for 3-D arrays)."""
    ok = None
    for a in arrays:
        f = np.isfinite(a).all(axis=1) if a.ndim == 3 else np.isfinite(a)
        ok = f if ok is None else ok & f
    return ok


# --------------------------------------------------------------------------------------------
# Fitting helpers
# --------------------------------------------------------------------------------------------

@dataclass
class Fit:
    sample: Sample
    sums: dict[str, SlopeSums]

    def describe(self) -> dict:
        return {"n_obs": self.sample.n, "n_cells": int(np.unique(self.sample.c).size),
                "n_dates": int(np.unique(self.sample.t).size)}


def fit(pd_: PairData, mask: np.ndarray, ys: dict[str, np.ndarray], *, x: np.ndarray | None = None,
        controls: bool | list[np.ndarray] = True, fe: str = "cell+date") -> Fit:
    """Slopes of every outcome in ``ys`` on the soil-moisture covariate over ``mask``."""
    x = pd_.s if x is None else x
    sample = Sample.from_mask(mask, pd_.weight)
    groups = fe_groups(sample, fe, pd_.month)
    if controls is True:
        ctrl_arrays = [pd_.a1, pd_.a2]
    elif controls is False:
        ctrl_arrays = []
    else:
        ctrl_arrays = controls
    ctrl = np.column_stack([sample.gather(a) for a in ctrl_arrays]) if ctrl_arrays else None
    names, blocks = list(ys), []
    for k in names:
        v = sample.gather(ys[k])
        blocks.append(v if v.ndim == 2 else v[:, None])
    widths = [b.shape[1] for b in blocks]
    big = slope_sums(sample, np.column_stack(blocks), sample.gather(x), ctrl, groups)
    sums, start = {}, 0
    for k, wdt in zip(names, widths, strict=True):
        sums[k] = SlopeSums(N=big.N[:, start:start + wdt], D=big.D)
        start += wdt
    return Fit(sample, sums)


def diff(a: SlopeSums, b: SlopeSums) -> SlopeSums:
    return SlopeSums(N=a.N - b.N, D=a.D)


def lead_mean(sums: SlopeSums, cols: np.ndarray, counts: np.ndarray) -> tuple[float, np.ndarray]:
    return float(sums.beta[cols].mean()), sums.boot(counts)[:, cols].mean(axis=1)


def summary_mean(sums: SlopeSums, cols: np.ndarray, counts: np.ndarray, cfg: Config,
                 alternative: str = "less") -> dict:
    value, boot = lead_mean(sums, cols, counts)
    inf = cfg["inference"]
    return summarize(value, boot, sesoi=inf["sesoi"], alternative=alternative,
                     se_dk=dk_se_mean_slope(sums, cols, inf["driscoll_kraay_lags"]))


def profile(sums: SlopeSums, counts: np.ndarray) -> dict:
    boot = sums.boot(counts)
    return {"estimate": sums.beta.tolist(),
            "ci95_low": np.quantile(boot, 0.025, axis=0).tolist(),
            "ci95_high": np.quantile(boot, 0.975, axis=0).tolist()}


# --------------------------------------------------------------------------------------------
# Per-pair analysis
# --------------------------------------------------------------------------------------------

def analyze_pair(cfg: Config, cov: xr.Dataset, name: str) -> dict:
    p = load_pair(cfg, cov, name)
    inf = cfg["inference"]
    lead_days = np.arange(cfg["lead_days"][0], cfg["lead_days"][1] + 1)

    def cols(rng: list[int]) -> np.ndarray:
        return np.flatnonzero((lead_days >= rng[0]) & (lead_days <= rng[1]))

    prim, early, late = cols(cfg["primary_lead_days"]), cols(cfg["lead_growth_early"]), \
        cols(cfg["lead_growth_late"])
    counts = block_counts(p.dates.size, inf["block_days"], inf["n_boot"], inf["seed"])
    counts14 = block_counts(p.dates.size, inf["block_days_alt"], inf["n_boot"], inf["seed"])

    e_ai, e_ifs = p.e["ai_afternoon"], p.e["ifs_afternoon"]
    covariates_ok = all_finite(p.s, p.a1, p.a2)
    core = p.season & p.usable[None, :] & covariates_ok & all_finite(e_ai, e_ifs)
    water = core & (p.regime == "water")[None, :]
    energy = core & (p.regime == "energy")[None, :]
    out: dict = {}

    # ---- primary -----------------------------------------------------------------------------
    f = fit(p, water, {"ai": e_ai, "ifs": e_ifs})
    d = diff(f.sums["ai"], f.sums["ifs"])
    out["sample_primary"] = f.describe()
    out["primary"] = {
        "beta_ai": summary_mean(f.sums["ai"], prim, counts, cfg),
        "beta_ifs": summary_mean(f.sums["ifs"], prim, counts, cfg),
        "delta": summary_mean(d, prim, counts, cfg),
    }
    out["lead_profile"] = {"lead_day": lead_days.tolist(), "ai": profile(f.sums["ai"], counts),
                           "ifs": profile(f.sums["ifs"], counts), "delta": profile(d, counts)}

    # ---- S1 lead growth ------------------------------------------------------------------------
    v_late, b_late = lead_mean(d, late, counts)
    v_early, b_early = lead_mean(d, early, counts)
    out["S1_lead_growth"] = summarize(v_late - v_early, b_late - b_early)

    # ---- S2 diurnal fingerprint (6-hourly systems only) ------------------------------------------
    if p.night_usable.any() and cfg["pairs"][name]["step_hours"] == 6:
        both = water & p.night_usable[None, :] & all_finite(p.e["ai_night"], p.e["ifs_night"])
        fd = fit(p, both, {"ai": e_ai, "ifs": e_ifs, "ai_n": p.e["ai_night"],
                           "ifs_n": p.e["ifs_night"]})
        va, ba = lead_mean(diff(fd.sums["ai"], fd.sums["ifs"]), prim, counts)
        vn, bn = lead_mean(diff(fd.sums["ai_n"], fd.sums["ifs_n"]), prim, counts)
        out["S2_diurnal"] = {"delta_afternoon": summarize(va, ba), "delta_night": summarize(vn, bn),
                             "afternoon_minus_night": summarize(va - vn, ba - bn),
                             "sample": fd.describe()}

    # ---- S3 regime contrast ----------------------------------------------------------------------
    fe_ = fit(p, energy, {"ai": e_ai, "ifs": e_ifs})
    de = diff(fe_.sums["ai"], fe_.sums["ifs"])
    vw, bw = lead_mean(d, prim, counts)
    ve, be = lead_mean(de, prim, counts)
    aw, baw = lead_mean(f.sums["ai"], prim, counts)
    ae, bae = lead_mean(fe_.sums["ai"], prim, counts)
    out["S3_regime"] = {"delta_energy": summarize(ve, be), "beta_ai_energy": summarize(ae, bae),
                        "delta_water_minus_energy": summarize(vw - ve, bw - be),
                        "beta_ai_water_minus_energy": summarize(aw - ae, baw - bae),
                        "sample_energy": fe_.describe()}

    # ---- S4 dose-response across coupling deciles ------------------------------------------------
    rows, boots = [], []
    for k in range(1, 11):
        mk = core & (p.decile == k)[None, :]
        fk = fit(p, mk, {"ai": e_ai, "ifs": e_ifs})
        va_, ba_ = lead_mean(fk.sums["ai"], prim, counts)
        vi_, bi_ = lead_mean(fk.sums["ifs"], prim, counts)
        rows.append({"decile": k, "beta_ai": va_, "beta_ifs": vi_,
                     "ai_ci95": np.quantile(ba_, [0.025, 0.975]).tolist(),
                     "ifs_ci95": np.quantile(bi_, [0.025, 0.975]).tolist(),
                     "area_weight": float(p.weight[p.decile == k].sum()), **fk.describe()})
        boots.append(ba_)
    k_arr = np.arange(1, 11, dtype=float)
    wts = np.array([r["area_weight"] for r in rows])
    km = np.average(k_arr, weights=wts)

    def trend(values: np.ndarray) -> np.ndarray:
        return ((values - np.average(values, weights=wts, axis=-1)[..., None]) * (k_arr - km) * wts
                ).sum(axis=-1) / ((k_arr - km) ** 2 * wts).sum()

    est = trend(np.array([r["beta_ai"] for r in rows]))
    out["S4_dose_response"] = {"trend_per_decile": summarize(float(est), trend(np.stack(boots, 1))),
                               "deciles": rows}

    # ---- S5 wrong-year placebo -------------------------------------------------------------------
    y0, y1 = cfg["placebo_years"]
    prev = p.dates - pd.Timedelta(days=1)
    true_val = out["primary"]["beta_ai"]["estimate"]
    placebo = {}
    for year in range(y0, y1 + 1):
        day_of_month = np.where((prev.month == 2) & (prev.day == 29), 28, prev.day)
        days = pd.to_datetime(pd.DataFrame({"year": year, "month": prev.month,
                                            "day": day_of_month}))
        sp = cov.s.sel(day=days.values).values
        fp = fit(p, water & np.isfinite(sp), {"ai": e_ai}, x=sp)
        placebo[year] = float(fp.sums["ai"].beta[prim].mean())
    vals = np.array(list(placebo.values()))
    out["S5_placebo"] = {"true": true_val, "placebo": placebo,
                         "rank_p": float((1 + (vals <= true_val).sum()) / (vals.size + 1)),
                         "placebo_mean": float(vals.mean()), "placebo_sd": float(vals.std(ddof=1))}

    # ---- S6 circulation pathway (FGN only) -------------------------------------------------------
    if "ez500_ai" in p.extra:
        ez = p.extra["ez500_ai"]
        m6 = water & np.isfinite(ez)
        five = int(np.flatnonzero(lead_days == 5)[0])
        y5 = {"ai5": e_ai[:, five, :]}
        base = fit(p, m6, y5)
        with_z = fit(p, m6, y5, controls=[p.a1, p.a2, ez])
        zslope = fit(p, m6, {"z": ez})
        b0, bz = base.sums["ai5"], with_z.sums["ai5"]
        ratio_boot = bz.boot(counts)[:, 0] / b0.boot(counts)[:, 0]
        out["S6_circulation"] = {
            "beta_day5": summarize(float(b0.beta[0]), b0.boot(counts)[:, 0]),
            "beta_day5_with_z500_error": summarize(float(bz.beta[0]), bz.boot(counts)[:, 0]),
            "retained_fraction": summarize(float(bz.beta[0] / b0.beta[0]), ratio_boot,
                                           alternative="two-sided"),
            "z500_error_slope_m_per_sd": summarize(float(zslope.sums["z"].beta[0]),
                                                   zslope.sums["z"].boot(counts)[:, 0],
                                                   alternative="two-sided"),
            "sample": base.describe()}

    # ---- robustness ------------------------------------------------------------------------------
    rob: dict = {}
    alt_ok = all_finite(p.e["ai_afternoon_alt"], p.e["ifs_afternoon_alt"])
    fa = fit(p, water & alt_ok, {"ai": p.e["ai_afternoon_alt"], "ifs": p.e["ifs_afternoon_alt"]})
    rob["R1_truth_swap"] = {"beta_ai": summary_mean(fa.sums["ai"], prim, counts, cfg),
                            "beta_ifs": summary_mean(fa.sums["ifs"], prim, counts, cfg),
                            "sample": fa.describe()}
    for label, kwargs in {"cellmonth+date": {"fe": "cellmonth+date"}, "date_only": {"fe": "date"},
                          "no_controls": {"controls": False}}.items():
        fr = fit(p, water, {"ai": e_ai, "ifs": e_ifs}, **kwargs)
        rob[f"R2_{label}"] = {"beta_ai": summary_mean(fr.sums["ai"], prim, counts, cfg),
                              "delta": summary_mean(diff(fr.sums["ai"], fr.sums["ifs"]), prim,
                                                    counts, cfg)}
    rob["R3_block14"] = {"beta_ai": summary_mean(f.sums["ai"], prim, counts14, cfg),
                         "delta": summary_mean(d, prim, counts14, cfg)}
    fl = fit(p, water & np.isfinite(p.s_l1), {"ai": e_ai, "ifs": e_ifs}, x=p.s_l1)
    rob["R4_surface_layer"] = {"beta_ai": summary_mean(fl.sums["ai"], prim, counts, cfg),
                               "delta": summary_mean(diff(fl.sums["ai"], fl.sums["ifs"]), prim,
                                                     counts, cfg)}
    loo = {}
    for cont in sorted(set(p.continent[(p.regime == "water") & p.usable])):
        mc = water & (p.continent != cont)[None, :]
        if not mc.any():
            loo[cont] = {"skipped": "no water-limited cells outside this continent"}
            continue
        fc = fit(p, mc, {"ai": e_ai, "ifs": e_ifs})
        loo[cont] = {"beta_ai": summary_mean(fc.sums["ai"], prim, counts, cfg)["estimate"],
                     "delta": summary_mean(diff(fc.sums["ai"], fc.sums["ifs"]), prim, counts,
                                           cfg)["estimate"],
                     "n_cells_dropped": int(((p.continent == cont) & (p.regime == "water")
                                             & p.usable).sum())}
    rob["R5_leave_one_continent_out"] = loo
    inj = {}
    for kappa in cfg["injection_kappas"]:
        e_syn = e_ifs - kappa * p.s[:, None, :]
        fi = fit(p, water, {"syn": e_syn, "ifs": e_ifs})
        rec = float(diff(fi.sums["syn"], fi.sums["ifs"]).beta[prim].mean())
        inj[str(kappa)] = {"injected": -kappa, "recovered": rec, "abs_error": abs(rec + kappa)}
    rob["R7_injection"] = inj
    out["robustness"] = rob

    # ---- exploratory, not pre-registered: slopes of the truth and of each forecast ---------------
    # e = truth - forecast, so beta_e = beta_truth - beta_forecast exactly on the same sample.
    levels = {"truth": p.extra["truth_level"], "ai": p.extra["ai_level"],
              "ifs": p.extra["ifs_level"]}
    expl = {}
    for label, fe in (("cell+date", "cell+date"), ("cellmonth+date", "cellmonth+date")):
        fx = fit(p, water, levels, fe=fe)
        expl[label] = {k: {"lead_profile": fx.sums[k].beta.tolist(),
                           "mean_3_10": summarize(*lead_mean(fx.sums[k], prim, counts),
                                                  alternative="two-sided")}
                       for k in levels}
    out["exploratory_level_slopes"] = expl
    return out


# --------------------------------------------------------------------------------------------
# Member-level secondary analyses (S7-S9)
# --------------------------------------------------------------------------------------------

def analyze_members(cfg: Config, cov: xr.Dataset) -> dict:
    ds = xr.open_dataset(cfg.processed / "members_fgn2022.nc")
    inits = pd.DatetimeIndex(ds.init.values)
    s = cov.s.sel(day=(inits - pd.Timedelta(days=1)).values).values
    start = cov.season_start_month.values.astype(int)
    season = in_season(np.asarray(inits.month), start, cfg["warm_season_months"])
    water = (cov.regime.values.astype(str) == "water")[None, :]
    fields = {k: ds[k].values.astype(np.float64) for k in
              ("truth", "obs_hot", "fgn_mean", "fgn_var", "fgn_crps", "fgn_prob_hot",
               "ifs_mean", "ifs_var", "ifs_crps", "ifs_prob_hot")}
    ok = season & water & np.isfinite(s) & np.logical_and.reduce(
        [np.isfinite(v) for v in fields.values()])
    sqerr = (fields["truth"] - fields["fgn_mean"]) ** 2
    w = cov.weight.values[None, :]
    year_days = pd.date_range("2022-01-01", "2022-12-31", freq="D")
    day_pos = year_days.get_indexer(inits)
    inf = cfg["inference"]
    counts = block_counts(year_days.size, inf["block_days"], inf["n_boot"], inf["seed"])
    m = float(ds.attrs["fgn_n_members"])

    def date_sums(mask: np.ndarray, value: np.ndarray) -> np.ndarray:
        per_init = np.where(mask, w * value, 0.0).sum(axis=1)
        out = np.zeros(year_days.size)
        out[day_pos] = per_init
        return out

    strata = {"dry": ok & (s < cfg["strata"]["dry_max"]),
              "wet": ok & (s > cfg["strata"]["wet_min"])}
    sums = {}
    for label, mk in strata.items():
        components = {
            "w": np.ones_like(s), "obs": fields["obs_hot"], "p_fgn": fields["fgn_prob_hot"],
            "p_ifs": fields["ifs_prob_hot"], "crps_fgn": fields["fgn_crps"],
            "crps_ifs": fields["ifs_crps"], "var_fgn": fields["fgn_var"], "sqerr_fgn": sqerr}
        sums[label] = {k: date_sums(mk, v) for k, v in components.items()}

    def stat(fn, label):
        full = fn({k: v.sum() for k, v in sums[label].items()})
        boot = fn({k: counts @ v for k, v in sums[label].items()})
        return full, boot

    def reliability(z):
        return (z["obs"] - z["p_fgn"]) / z["w"]

    def ifs_minus_fgn(z):
        return (z["p_ifs"] - z["p_fgn"]) / z["w"]

    def crpss(z):
        return 1.0 - z["crps_fgn"] / z["crps_ifs"]

    def ssr(z):
        return np.sqrt((m + 1) / m) * np.sqrt(z["var_fgn"] / z["sqerr_fgn"])

    out = {}
    r, rb = stat(reliability, "dry")
    out["S7_obs_minus_fgn_prob_dry"] = summarize(r, rb, alternative="greater")
    r, rb = stat(ifs_minus_fgn, "dry")
    out["S7_ifs_minus_fgn_prob_dry"] = summarize(r, rb, alternative="greater")
    for label, fn in (("S8_crpss_dry_minus_wet", crpss), ("S9_ssr_dry_minus_wet", ssr)):
        dv, db = stat(fn, "dry")
        wv, wb = stat(fn, "wet")
        out[label] = summarize(dv - wv, db - wb)
        out[label]["dry"], out[label]["wet"] = float(dv), float(wv)
    for label in ("dry", "wet"):
        z = {k: v.sum() for k, v in sums[label].items()}
        out[f"strata_{label}"] = {
            "n_obs": int(strata[label].sum()), "obs_freq": float(z["obs"] / z["w"]),
            "p_fgn": float(z["p_fgn"] / z["w"]), "p_ifs": float(z["p_ifs"] / z["w"]),
            "crps_fgn": float(z["crps_fgn"] / z["w"]), "crps_ifs": float(z["crps_ifs"] / z["w"]),
            "ssr_fgn": float(ssr(z))}
    return out


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------

def _git(cfg: Config, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              cwd=cfg.root).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_analysis(cfg: Config, skip_members: bool = False) -> dict:
    cov = xr.open_dataset(cfg.processed / "covariates.nc")
    results: dict = {
        "preregistration_commit": (_git(cfg, "log", "--format=%H", "--diff-filter=A", "--",
                                        "docs/preregistration.md").splitlines() or ["unknown"])[-1],
        "code_commit": _git(cfg, "rev-parse", "HEAD"),
        "domain": {"n_cells": int(cov.sizes["cell"]),
                   "n_water": int((cov.regime.values.astype(str) == "water").sum()),
                   "n_energy": int((cov.regime.values.astype(str) == "energy").sum()),
                   "tci_tercile_bounds": list(map(float, cov.attrs["tci_tercile_bounds"])),
                   "water_cells_by_continent": pd.Series(
                       cov.continent.values.astype(str)[cov.regime.values.astype(str) == "water"]
                   ).value_counts().to_dict()},
        "data_provenance": {name: manifest_digest(cfg.store(name)) for name in cfg["stores"]},
        "pairs": {},
    }
    for name in cfg["pairs"]:
        log.info("analysing %s", name)
        results["pairs"][name] = analyze_pair(cfg, cov, name)

    fgn, gc = results["pairs"]["fgn2022"]["primary"], results["pairs"]["gencast2020"]["primary"]
    tests = {"H1": fgn["beta_ai"], "H2": fgn["delta"], "H3": gc["beta_ai"], "H4": gc["delta"]}
    adjusted = holm([t["p_one_sided"] for t in tests.values()])
    alpha = cfg["inference"]["alpha"]
    results["confirmatory"] = {
        k: {**t, "p_holm": adj, "verdict": verdict(adj, t["equivalent_within_sesoi"], alpha)}
        for (k, t), adj in zip(tests.items(), adjusted, strict=True)}

    if not skip_members and (cfg.processed / "members_fgn2022.nc").exists():
        results["members_fgn2022"] = analyze_members(cfg, cov)

    out_dir = cfg.results
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    for name, res in results["pairs"].items():
        lp = res["lead_profile"]
        table = {"lead_day": lp["lead_day"]}
        for role in ("ai", "ifs", "delta"):
            for key in ("estimate", "ci95_low", "ci95_high"):
                table[f"{role}_{key}"] = lp[role][key]
        pd.DataFrame(table).to_csv(out_dir / "tables" / f"lead_profile_{name}.csv", index=False,
                                   float_format="%.5f")
        pd.DataFrame(res["S4_dose_response"]["deciles"]).to_csv(
            out_dir / "tables" / f"coupling_deciles_{name}.csv", index=False, float_format="%.5f")
    (out_dir / "summary.md").write_text(render(results))
    log.info("wrote %s and %s", out_dir / "results.json", out_dir / "summary.md")
    return results
