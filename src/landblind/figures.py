"""Figures drawn from ``results/results.json`` and the covariates file."""

from __future__ import annotations

import json
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from landblind.config import Config

log = logging.getLogger(__name__)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
AI = "#eb6834"      # categorical slot 2
IFS = "#2a78d6"     # categorical slot 1
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABELS = {"fgn2022": ("FGN (WeatherNext 2)", "2022", "FGN"),
          "gencast2020": ("GenCast", "2020", "GenCast")}


def _style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "text.color": INK, "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "font.family": "sans-serif",
    })


def _save(fig: plt.Figure, cfg: Config, name: str) -> None:
    out = cfg.results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out / f"{name}.png")


def lead_profiles(cfg: Config, res: dict) -> None:
    lo, hi = cfg["primary_lead_days"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, (name, pr) in zip(axes, res["pairs"].items(), strict=True):
        lp = pr["lead_profile"]
        x = np.array(lp["lead_day"])
        ax.axvspan(lo - 0.5, hi + 0.5, color="#f1f0ec", zorder=0, lw=0)
        ax.axhline(0, color=INK_2, lw=0.8)
        model, year, short = LABELS[name]
        for key, color, label in (("ifs", IFS, "IFS ENS"), ("ai", AI, short)):
            est = np.array(lp[key]["estimate"])
            ax.fill_between(x, lp[key]["ci95_low"], lp[key]["ci95_high"], color=color, alpha=0.16,
                            lw=0)
            ax.plot(x, est, color=color, lw=2, marker="o", ms=3.5, label=label)
            ax.annotate(label, (x[-1], est[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        ax.set_title(f"{model} vs IFS ENS, {year}", loc="left")
        ax.set_xlabel("Lead day")
        ax.set_xlim(0.5, x[-1] + 3.2)
        ax.set_xticks([1, 3, 5, 7, 10, 15])
        ax.text(lo - 0.3, 0.98, "primary window", transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=INK_2, va="top")
    axes[0].set_ylabel("Error slope on initial soil-moisture\nanomaly (K per SD)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.text(0.01, -0.04, "Negative: the forecast is too cold when soils start dry. Afternoon 2 m "
             "temperature, water-limited cells, warm season; bands are 95% block-bootstrap "
             "intervals.", fontsize=7.5, color=INK_2)
    _save(fig, cfg, "fig1_lead_profile")


def coupling_deciles(cfg: Config, res: dict) -> None:
    rows = res["pairs"]["fgn2022"]["S4_dose_response"]["deciles"]
    k = np.array([r["decile"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.axhline(0, color=INK_2, lw=0.8)
    for key, color, label, dx in (("ifs", IFS, "IFS ENS", -0.12), ("ai", AI, "FGN", 0.12)):
        est = np.array([r[f"beta_{key}"] for r in rows])
        ci = np.array([r[f"{key}_ci95"] for r in rows])
        ax.errorbar(k + dx, est, yerr=[est - ci[:, 0], ci[:, 1] - est], fmt="o", ms=5,
                    color=color, ecolor=color, elinewidth=1.4, capsize=0, label=label)
    ax.set_xticks(k)
    ax.set_xlabel("Land–atmosphere coupling decile (1 = weakest, 10 = strongest)")
    ax.set_ylabel("Mean error slope, lead days 3–10\n(K per SD)")
    ax.set_title("Where soils control evaporation, 2022", loc="left")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, cfg, "fig2_coupling_deciles")


def placebo(cfg: Config, res: dict) -> None:
    s5 = res["pairs"]["fgn2022"]["S5_placebo"]
    vals = np.array(list(s5["placebo"].values()))
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    rng = np.random.default_rng(0)
    ax.scatter(vals, rng.uniform(-0.25, 0.25, vals.size), s=22, color=INK_2, alpha=0.7,
               label="Soil moisture from a wrong year (1990–2019)")
    ax.scatter([s5["true"]], [0], s=70, color=AI, zorder=3, label="Soil moisture from 2022")
    below = int((vals > s5["true"]).sum())
    ax.annotate(f"2022: {s5['true']:+.3f}\nbelow {below} of {vals.size} wrong years",
                (s5["true"], 0), xytext=(0, 14), textcoords="offset points", ha="left",
                fontsize=8)
    span = max(np.ptp(np.append(vals, s5["true"])), 0.05)
    lo_x = min(vals.min(), s5["true"]) - 0.12 * span
    hi_x = max(vals.max(), s5["true"]) + 0.12 * span
    ax.set_xlim(lo_x, hi_x)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.8)
    ax.set_xlabel("FGN mean error slope, lead days 3–10 (K per SD)")
    ax.legend(loc="upper left", fontsize=7.5, ncol=1, bbox_to_anchor=(0, 1.35))
    ax.grid(axis="y", visible=False)
    _save(fig, cfg, "fig3_placebo")


def regime_map(cfg: Config) -> None:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        log.warning("cartopy not installed; skipping map")
        return
    cov = xr.open_dataset(cfg.processed / "covariates.nc")
    lon, lat = cov.lon.values, cov.lat.values
    dec = cov.tci_decile.values.astype(int)
    water = cov.regime.values.astype(str) == "water"
    fig = plt.figure(figsize=(8.0, 3.9))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor=INK_2)
    ramp = mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    cmap = mpl.colors.ListedColormap(ramp(np.linspace(0, 1, 10)))
    sc = ax.scatter(lon, lat, c=dec, cmap=cmap, vmin=0.5, vmax=10.5, s=3.2, marker="s", lw=0,
                    transform=ccrs.PlateCarree())
    ax.scatter(lon[water], lat[water], s=3.2, marker="s", facecolor="none", edgecolor=AI,
               linewidths=0.35, transform=ccrs.PlateCarree())
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04,
                      ticks=range(1, 11))
    cb.set_label("Coupling decile (terrestrial coupling index, ERA5 1990–2019 warm seasons); "
                 "orange outline: water-limited tercile", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title("Where soil moisture controls evaporation", loc="left")
    _save(fig, cfg, "fig4_coupling_map")


def hot_day_probabilities(cfg: Config, res: dict) -> None:
    m = res.get("members_fgn2022")
    if m is None:
        return
    groups = [("dry", "Dry soils (s < −0.43)"), ("wet", "Wet soils (s > 0.43)")]
    bars = [("obs_freq", "Observed", INK_2), ("p_fgn", "FGN", AI), ("p_ifs", "IFS ENS", IFS)]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    width = 0.24
    for gi, (key, _) in enumerate(groups):
        z = m[f"strata_{key}"]
        for bi, (field, label, color) in enumerate(bars):
            x = gi + (bi - 1) * (width + 0.03)
            ax.bar(x, 100 * z[field], width=width, color=color, label=label if gi == 0 else None)
            ax.text(x, 100 * z[field] + 0.4, f"{100 * z[field]:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(groups)), [g[1] for g in groups])
    ax.set_ylabel("Hot afternoons, lead day 5 (%)")
    ax.set_title("Hot-day probability by initial soil moisture, 2022", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=8)
    fig.text(0.01, -0.1, "Hot: above the local hour-matched 90th percentile (ERA5 1990–2019).\n"
             "Water-limited cells, warm season, every second 00 UTC initialisation; truth HRES-t0.",
             fontsize=7.5, color=INK_2)
    _save(fig, cfg, "fig5_hot_day_probability")


def draw_figures(cfg: Config) -> None:
    _style()
    res = json.loads((cfg.results / "results.json").read_text())
    lead_profiles(cfg, res)
    coupling_deciles(cfg, res)
    placebo(cfg, res)
    hot_day_probabilities(cfg, res)
    regime_map(cfg)
