"""End-to-end check of the registered analysis on synthetic data with a planted effect.

The synthetic AI model is too cold over dry soils by ``BETA`` K per standard deviation, growing
with lead time only through the planted profile; the synthetic IFS has no soil-moisture error.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from landmem.analysis import run_analysis
from landmem.config import load_config
from landmem.figures import draw_figures

BETA = -0.30
N_CELLS = 90


def _covariates(rng: np.random.Generator) -> xr.Dataset:
    days = pd.date_range("1990-01-01", "2022-12-31", freq="D")
    lon = np.linspace(0, 358.5, N_CELLS)
    lat = rng.uniform(-45, 65, N_CELLS)
    s = np.zeros((days.size, N_CELLS), dtype=np.float32)
    shocks = rng.normal(size=(days.size, N_CELLS)).astype(np.float32)
    for t in range(1, days.size):  # persistent anomalies, like soil moisture
        s[t] = 0.97 * s[t - 1] + np.sqrt(1 - 0.97**2) * shocks[t]
    regime = np.array(["water", "energy", "middle"] * (N_CELLS // 3))
    return xr.Dataset(
        {
            "s": (("day", "cell"), s),
            "s_l1": (("day", "cell"), s + 0.3 * shocks),
            "tmax_anom": (("day", "cell"), rng.normal(size=s.shape).astype(np.float32)),
            "t2m_00utc_anom": (("day6", "cell"), rng.normal(size=s.shape).astype(np.float32)),
            "q90_afternoon": (("doy", "cell"), np.full((365, N_CELLS), 1.2, np.float32)),
            "tci": ("cell", rng.normal(size=N_CELLS).astype(np.float32)),
            "season_start_month": ("cell", np.where(lat > 0, 6, 12).astype(np.int8)),
            "regime": ("cell", regime),
            "tci_decile": ("cell", (np.arange(N_CELLS) % 10 + 1).astype(np.int8)),
            "continent": ("cell", np.array(["europe", "asia"] * (N_CELLS // 2))),
            "ilon": ("cell", np.arange(N_CELLS, dtype=np.int16)),
            "ilat": ("cell", np.zeros(N_CELLS, dtype=np.int16)),
            "lon": ("cell", lon),
            "lat": ("cell", lat),
            "weight": ("cell", np.cos(np.deg2rad(lat))),
        },
        coords={"day": days.values, "day6": days.values, "doy": np.arange(1, 366)},
        attrs={"tci_tercile_bounds": [-0.4, 0.4]},
    )


def _panel(rng, cov: xr.Dataset, year: int, beta: float, with_z: bool) -> xr.Dataset:
    inits = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    s = cov.s.sel(day=(inits - pd.Timedelta(days=1)).values).values
    leads = np.arange(1, 16)
    profile = beta * np.clip(leads / 5.0, 0, 1.6)  # grows with lead, like land memory
    shape = (inits.size, leads.size, N_CELLS)
    alpha = rng.normal(size=N_CELLS)

    def forecast(slope_profile, noise):
        err = alpha + slope_profile[None, :, None] * s[:, None, :] + noise * rng.normal(size=shape)
        return err.astype(np.float32)

    truth = rng.normal(290, 5, size=shape).astype(np.float32)
    e_ai, e_ifs = forecast(profile, 1.0), forecast(np.zeros_like(profile), 1.0)
    data = {
        "truth_afternoon": truth, "alt_truth_afternoon": truth + 0.1,
        "ai_afternoon": truth - e_ai, "ifs_afternoon": truth - e_ifs,
        "truth_night": truth, "alt_truth_night": truth + 0.1,
        "ai_night": truth - forecast(0.3 * profile, 1.0), "ifs_night": truth - e_ifs,
    }
    ds = xr.Dataset({k: (("init", "lead_day", "cell"), v) for k, v in data.items()},
                    coords={"init": inits.values, "lead_day": leads})
    ds["afternoon_usable"] = ("cell", np.ones(N_CELLS, bool))
    ds["night_usable"] = ("cell", np.ones(N_CELLS, bool))
    if with_z:
        ds["ai_z500_day5"] = (("init", "cell"), rng.normal(5800, 50, (inits.size, N_CELLS)))
        ds["ifs_z500_day5"] = (("init", "cell"), rng.normal(5800, 50, (inits.size, N_CELLS)))
        ds["truth_z500_day5"] = (("init", "cell"), rng.normal(5800, 50, (inits.size, N_CELLS)))
    return ds


def _members(rng, cov: xr.Dataset) -> xr.Dataset:
    inits = pd.date_range("2022-01-01", "2022-12-31", freq="2D")
    shape = (inits.size, N_CELLS)
    truth = rng.normal(0, 1, shape)
    data = {"truth": truth, "obs_hot": (truth > 1.2).astype(float)}
    for tag in ("fgn", "ifs"):
        data.update({f"{tag}_mean": truth + rng.normal(0, 1, shape),
                     f"{tag}_var": np.ones(shape),
                     f"{tag}_crps": np.abs(rng.normal(0.6, 0.1, shape)),
                     f"{tag}_prob_hot": rng.uniform(0, 0.3, shape)})
    return xr.Dataset({k: (("init", "cell"), v) for k, v in data.items()},
                      coords={"init": inits.values}, attrs={"fgn_n_members": 64})


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    cfg = load_config(root=root)
    cfg.raw["inference"]["n_boot"] = 300
    rng = np.random.default_rng(42)
    cov = _covariates(rng)
    cov.to_netcdf(cfg.processed / "covariates.nc")
    _panel(rng, cov, 2022, BETA, True).to_netcdf(cfg.processed / "panel_fgn2022.nc")
    _panel(rng, cov, 2020, BETA, False).to_netcdf(cfg.processed / "panel_gencast2020.nc")
    _members(rng, cov).to_netcdf(cfg.processed / "members_fgn2022.nc")
    res = run_analysis(cfg)
    draw_figures(cfg)
    res["_figures_dir"] = str(cfg.results / "figures")
    return res


def test_primary_recovers_planted_effect(results):
    expected = BETA * np.clip(np.arange(3, 11) / 5.0, 0, 1.6).mean()
    for pair in ("fgn2022", "gencast2020"):
        prim = results["pairs"][pair]["primary"]
        assert prim["beta_ai"]["estimate"] == pytest.approx(expected, abs=0.05)
        assert prim["beta_ifs"]["estimate"] == pytest.approx(0.0, abs=0.05)
        assert prim["delta"]["estimate"] == pytest.approx(expected, abs=0.05)


def test_confirmatory_decisions(results):
    conf = results["confirmatory"]
    assert set(conf) == {"H1", "H2", "H3", "H4"}
    assert all(c["verdict"] == "supported" for c in conf.values())
    assert all(c["p_holm"] >= c["p_one_sided"] for c in conf.values())


def test_secondary_and_robustness_shapes(results):
    fgn = results["pairs"]["fgn2022"]
    assert fgn["S1_lead_growth"]["estimate"] < 0
    assert fgn["S2_diurnal"]["afternoon_minus_night"]["estimate"] < 0
    assert len(fgn["S4_dose_response"]["deciles"]) == 10
    assert len(fgn["S5_placebo"]["placebo"]) == 30
    assert fgn["S5_placebo"]["rank_p"] == pytest.approx(1 / 31)
    assert "S6_circulation" in fgn
    for kappa, rec in fgn["robustness"]["R7_injection"].items():
        assert rec["abs_error"] < 1e-8, kappa
    assert "S8_crpss_dry_minus_wet" in results["members_fgn2022"]


def test_figures_written(results):
    from pathlib import Path

    names = {f.stem for f in Path(results["_figures_dir"]).glob("*.png")}
    assert {"fig1_lead_profile", "fig2_coupling_deciles", "fig3_placebo"} <= names
