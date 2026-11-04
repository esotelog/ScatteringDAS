"""
DAS_plots: plotting functions used by the top-level notebooks (DASdata, DASrefInv, DASscatInv,
DASrefModel_*, DASscatModel_*) and the helpers they call, copied unchanged. The previous full
module is kept in ../py_files_old/DAS_plots_old.py (it also has the functions removed here).
"""
import pickle
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import DAS_model_inversion as DASmi
from matplotlib.colors import LogNorm
from matplotlib.ticker import (LogFormatterSciNotation, LogFormatter, LogLocator,
                               FormatStrFormatter, NullFormatter, MaxNLocator,
                               FixedLocator, FixedFormatter, NullLocator)
from matplotlib.colors import LogNorm, AsinhNorm, ListedColormap, Normalize

def plot_reflections(drange, source, best_line_out, angles_rays, xr=0.0,
                     step=None, same_side_only=True, figsize=(6, 8)):
    """
    Plot fibre, reflector, source, and incident/reflected rays.

    Parameters
    ----------
    drange : array-like
        Receiver y positions along the fibre [m].
    source : (xs, ys)
        Source position [m].
    best_line_out : dict
        Output of estimate_best_line (needs 'line', 'xp', 'yp').
    angles_rays : dict
        Output of incident_reflection (kept for API consistency;
        ray paths use source, xp, yp, drange).
    xr : float
        Fibre x position [m]. Default 0.
    step : int or None
        Plot every `step`-th ray. Default ~40 rays.
    same_side_only : bool
        If True, only draw rays with receiver on the same side of the
        reflector as the source.
    """
    xs, ys = map(float, source)
    xr = float(xr)
    yr = np.asarray(drange, float)
    xp = np.asarray(best_line_out["xp"], float)
    yp = np.asarray(best_line_out["yp"], float)
    m_line, b_line = map(float, best_line_out["line"])

    if not (yr.shape == xp.shape == yp.shape):
        raise ValueError("drange, xp, and yp must have the same shape")

    # fibre–reflector crossing
    y_cross = m_line * xr + b_line
    y_fiber_min = min(float(yr.min()), y_cross)
    y_fiber_max = max(float(yr.max()), y_cross)

    sig_s = m_line * xs - ys + b_line
    sig_r = m_line * xr - yr + b_line
    same_side = (sig_r * sig_s) > 0

    x_line = np.linspace(min(xr, xs) - 10.0, max(xr, xs) + 10.0, 300)
    y_line = m_line * x_line + b_line

    if step is None:
        step = max(1, len(yr) // 40)
    idx = np.arange(0, len(yr), int(step))
    if same_side_only:
        idx = idx[same_side[idx]]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_line, y_line, "k-", lw=2, label="reflector")
    ax.plot([xr, xr], [y_fiber_min, y_fiber_max], "b-", lw=1.5, label="fibre")
    ax.plot(np.full_like(yr, xr), yr, "b.", ms=3, label="receivers")
    ax.plot(xr, y_cross, "bs", ms=8, label="fibre–reflector crossing")
    ax.plot(xs, ys, "r*", ms=14, label="source")

    for i in idx:
        ax.plot([xs, xp[i]], [ys, yp[i]], color="C1", lw=0.8, alpha=0.8)
        ax.plot([xp[i], xr], [yp[i], yr[i]], color="C2", lw=0.8, alpha=0.8)
        ax.plot(xp[i], yp[i], "ko", ms=2)

    ax.plot([], [], color="C1", lw=2, label="incident")
    ax.plot([], [], color="C2", lw=2, label="reflected")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(rf"Reflection geometry  $m$={m_line:.4g}, $b$={b_line:.4g}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig, ax

def plot_cost(gsR, RT_batch, idx=0, figsize=(7, 4)):
    """
    Density plot of spectral misfit Csp[idx] vs ct (x) and cn/ct (y).

    Parameters
    ----------
    gsR : dict
        Output of cnct_search_RTomega (needs Csp, ct, cn_ratio, mode, ...).
    RT_batch : dict
        Output of RT_wave_batch (used for optional angle label).
    idx : int
        Batch / trace index into gsR["Csp"].
    """
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterSciNotation


    idx = int(idx)
    Csp = np.asarray(gsR["Csp"][idx], float)   # (n_cn, n_ct)
    ct = np.asarray(gsR["ct"], float)
    r = np.asarray(gsR["cn_ratio"], float)

    ct_best = float(gsR["ct_best"][idx])
    cn_best = float(gsR["cn_best"][idx])
    Csp_min = float(gsR["Csp_min"][idx])
    geom = int(gsR["geom_idx"][idx])
    mode = gsR.get("mode", "?")

    angle = None
    if RT_batch is not None and "angle" in RT_batch:
        angle = float(np.round(RT_batch["angle"][idx], 1))

    pos = Csp[Csp > 0]
    if pos.size == 0:
        raise ValueError(f"Csp[{idx}] has no positive values to plot")
    Csp_plot = np.clip(Csp, np.nanmin(pos), None)

    fig, ax = plt.subplots(figsize=figsize)
    pcm = ax.pcolormesh(
        ct, r, Csp_plot,
        shading="nearest",
        cmap="turbo",
        norm=LogNorm(vmin=Csp_plot.min(), vmax=Csp_plot.max()),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$c_t$")
    ax.set_ylabel(r"$c_n / c_t$")

    title = rf"Spectral misfit $C_{{\mathrm{{sp}}}}$, mode={mode}, idx={idx}"
    if angle is not None:
        title += rf", angle={angle}$^\circ$"
    ax.set_title(title)

    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$C_{\mathrm{sp}}$")
    cb.formatter = LogFormatterSciNotation(base=10)
    cb.update_ticks()

    fig.tight_layout()
    plt.show()
    return fig, ax
def plot_cost_sum(gsR, figsize=(7, 4), robust=True, vmin_percentile=1.0, ratio_axis=False):
    """
    Density plot of summed misfit.

    Layouts:
      1) cnct_ratio_search_time (Cw_sum, cn, cn_ratio):
           x = cn,  y = cn/ct
           Cw_sum[i, j] <-> cn_ratio[i], cn[j]
      2) DASref ratio (Csp_sum, ct, cn_ratio):
           x = ct,  y = cn/ct
      3) DASref (cn, ct) (Csp_sum, cn, ct):
           x = ct,  y = cn
           -- or, with ratio_axis=True: x = ct, y = cn/ct (see below)

    Marks argmin (*). If true_cnct is set, marks that node (+).

    `ratio_axis` (only meaningful for layout 3, a plain (cn,ct) grid such
    as cnct_search_Rtime_Gd_cc_obs's output): re-expresses that SAME
    Csp_sum data with cn/ct on the vertical axis instead of cn, without
    re-running any search. Since ratio = cn/ct depends on both grid axes,
    this is a curvilinear (cn, ct) -> (ct, cn/ct) remap -- pcolormesh is
    given 2D (ct, ratio) coordinate arrays matching Csp_sum's own shape,
    each column (fixed ct) simply relabeled by its own cn/ct values. This
    is the same approach as tests/plot_cost_sum_ratio_axis.py /
    tests/plots_test/cost_sum_ratio_axis.png -- useful because a genuine
    ratio-parameterized search (cnct_search_Rtime_ratio_Gd_cc) with
    true_cnct inserted can look flat due to the exact-match outlier (see
    the `robust` docstring below), whereas re-deriving the ratio view from
    a plain grid's own well-contrasted data sidesteps that entirely.
    Ignored (no effect) for layouts 1/2, which already have cn/ct as an
    axis.

    When `true_cnct` is inserted exactly onto the search grid (as done for
    noise-free synthetic tests), that single node's cost collapses to
    ~machine precision -- many orders of magnitude below every other grid
    point. Anchoring the log color scale's vmin to that outlier squeezes
    all of the actually-informative contrast into a sliver at the top of
    the palette, making the plot look flat/washed out even though the
    underlying cost surface has a clear, well-defined minimum (see
    tests/plots_test/cost_sum_ratio_axis.png for the same data plotted
    without that artifact). `robust=True` (default) anchors vmin to the
    `vmin_percentile`-th percentile of the (positive) cost values instead
    of the raw minimum, so the palette spans the bulk of the surface; any
    point below that percentile (e.g. the exact true_cnct node) is simply
    clipped to the bottom-most color. Set `robust=False` to restore the
    previous raw-min/max behavior.
    """

    if "Cw_sum" in gsR:
        Csp = np.asarray(gsR["Cw_sum"], float)
        C_label = r"$C_{\mathrm{w}}$"
        C_min = float(gsR["Cw_sum_min"])
        C_true_key = "Cw_sum_true"
    elif "Csp_sum" in gsR:
        Csp = np.asarray(gsR["Csp_sum"], float)
        C_label = r"$C_{\mathrm{sp}}$"
        C_min = float(gsR["Csp_sum_min"])
        C_true_key = "Csp_sum_true"
    else:
        raise KeyError("gsR needs 'Cw_sum' or 'Csp_sum'")

    ct_sum_best = float(gsR["ct_sum_best"])
    cn_sum_best = float(gsR["cn_sum_best"])
    mode = gsR.get("mode", "?")
    domain = gsR.get("domain", "?")
    ratio_best = cn_sum_best / ct_sum_best

    pos = Csp[Csp > 0]
    if pos.size == 0:
        raise ValueError("summed cost has no positive values to plot")
    Csp_plot = np.clip(Csp, np.nanmin(pos), None)

    # (1) scattering ratio search: horizontal = cn
    use_cn_x = (
        "Cw_sum" in gsR
        or gsR.get("axes") == ("cn_ratio", "cn")
        or ("cn" in gsR and "cn_ratio" in gsR and "ct" not in gsR)
    )

    if use_cn_x:
        if "cn" not in gsR or "cn_ratio" not in gsR:
            raise KeyError("cnct_ratio_search_time result needs 'cn' and 'cn_ratio'")
        x = np.asarray(gsR["cn"], float)
        y = np.asarray(gsR["cn_ratio"], float)
        C_mesh = Csp_plot                    # (n_ratio, n_cn)
        xlabel = r"$c_n$ [m/Pa]"
        ylabel = r"$c_n / c_t$"
        x_star = cn_sum_best
        y_star = ratio_best
    elif "cn_ratio" in gsR:
        if "ct" not in gsR:
            raise KeyError("ratio layout needs 'ct'")
        x = np.asarray(gsR["ct"], float)
        y = np.asarray(gsR["cn_ratio"], float)
        C_mesh = Csp_plot                    # (n_ratio, n_ct)
        xlabel = r"$c_t$ [m/Pa]"
        ylabel = r"$c_n / c_t$"
        x_star = ct_sum_best
        y_star = ratio_best
    else:
        if "cn" not in gsR or "ct" not in gsR:
            raise KeyError("gsR needs 'cn' and 'ct'")
        cn_ax = np.asarray(gsR["cn"], float)
        ct_ax = np.asarray(gsR["ct"], float)
        C_mesh = Csp_plot                    # (n_cn, n_ct)
        xlabel = r"$c_t$ [m/Pa]"
        x_star = ct_sum_best
        if ratio_axis:
            CN, CT = np.meshgrid(cn_ax, ct_ax, indexing="ij")  # both (n_cn, n_ct)
            x = CT
            y = CN / CT
            ylabel = r"$c_n / c_t$"
            y_star = ratio_best
        else:
            x = ct_ax
            y = cn_ax
            ylabel = r"$c_n$ [m/Pa]"
            y_star = cn_sum_best

    vmax = Csp_plot.max()
    if robust:
        vmin = float(np.nanpercentile(Csp_plot, vmin_percentile))
        vmin = min(vmin, vmax)  # degenerate case: percentile collapses onto vmax
    else:
        vmin = Csp_plot.min()

    fig, ax = plt.subplots(figsize=figsize)
    pcm = ax.pcolormesh(
        x, y, C_mesh,
        shading="nearest",
        cmap="turbo",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    title = (
        rf"Summed misfit {C_label}, mode={mode}, domain={domain}"
        f"\n$c_n$={cn_sum_best:.2e}, $c_t$={ct_sum_best:.2e}, "
        f"$c_n/c_t$={ratio_best:.3g}"
        f"\n{C_label}" + r"$_{\mathrm{min}}$=" + f"{C_min:.3g}"
    )
    if "s_hat_best" in gsR:
        title += rf", $\hat{{s}}$={float(gsR['s_hat_best']):.3g}"
    if robust and vmin > Csp_plot.min() * 1.001:
        title += f"\n(colorbar clipped below {vmin:.3g}, {vmin_percentile:g}th pct)"

    ax.plot(x_star, y_star, "r*", ms=12, label=rf"{C_label} min", zorder=5)

    true_pair = gsR.get("true_cnct")
    if true_pair is not None:
        cn_t, ct_t = map(float, true_pair)
        if use_cn_x or "cn_ratio" in gsR:
            x_t, y_t = cn_t, cn_t / ct_t
            if not use_cn_x:
                x_t = ct_t
        elif ratio_axis:
            x_t, y_t = ct_t, cn_t / ct_t
        else:
            x_t, y_t = ct_t, cn_t
        ax.plot(x_t, y_t, "w+", ms=14, mew=2, label=r"true $(c_n,c_t)$", zorder=6)
        C_true = gsR.get(C_true_key)
        if C_true is not None:
            title += (
                f"\ntrue $c_n$={cn_t:.2e}, $c_t$={ct_t:.2e}"
                f", {C_label}" + r"$_{\mathrm{true}}$=" + f"{float(C_true):.3g}"
            )

    ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=9)

    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(C_label)
    cb.formatter = LogFormatterSciNotation(base=10)
    cb.update_ticks()

    fig.tight_layout()
    plt.show()
    return fig, ax

def plot_cost_angle(gs, figsize=(7, 4)):
    """
    Density plot of Csp from cnct_anglesearch_omega.

    Matches cnct_anglesearch_omega / cnct_search_RTomegaV2:
        Csp[i, j] <-> cn[i], ct[j]
        horizontal = log10(cn), vertical = log10(ct)  (linear axes)
    """

    Csp = np.asarray(gs["Csp"], float)   # (n_cn, n_ct)
    cn = np.asarray(gs["cn"], float)
    ct = np.asarray(gs["ct"], float)

    log_cn = np.log10(cn)   # x-axis
    log_ct = np.log10(ct)   # y-axis

    mode = gs.get("mode", "?")
    angle = float(gs["angle"]) if "angle" in gs else None

    pos = Csp[Csp > 0]
    if pos.size == 0:
        raise ValueError("Csp has no positive values to plot")
    Csp_plot = np.clip(Csp, np.nanmin(pos), None)

    # pcolormesh(X, Y, C) needs C.shape = (len(Y), len(X)) = (n_ct, n_cn)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    pcm = ax.pcolormesh(
        log_cn, log_ct, Csp_plot.T,
        shading="nearest",
        cmap="turbo",
        norm=LogNorm(vmin=Csp_plot.min(), vmax=Csp_plot.max()),
    )
    ax.set_aspect("auto")
    ax.set_xlabel(r"$\log_{10} c_n$ [m/Pa]")
    ax.set_ylabel(r"$\log_{10} c_t$ [m/Pa]")

    title = rf"Spectral misfit $C_{{\mathrm{{sp}}}}$, mode={mode}"
    if angle is not None:
        title += rf", angle={angle:.1f}$^\circ$"
    ax.set_title(title)

    cb = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r"$C_{\mathrm{sp}}$")
    cb.formatter = LogFormatterSciNotation(base=10)
    cb.update_ticks()

    plt.show()
    return fig, ax
    
def plot_s_hat(gsR, figsize=(7, 4), neg_color="black", vmax_cap=None, ratio_axis=False):
    """
    Density plot of s_hat (same axis standard as plot_cost_sum).

    - s_hat > 0 : turbo + LogNorm
    - s_hat <= 0 / non-finite : solid neg_color
    - Star at Csp_sum minimum

    Axes (log-scaled physical units):
      - if 'cn_ratio' in gsR:  x = ct, y = cn/ct (native ratio grid)
      - elif ratio_axis:       x = ct, y = cn/ct -- curvilinear remap of a
                               plain (cn, ct) grid, same technique as
                               plot_cost_sum's own ratio_axis (see that
                               docstring): CN, CT = meshgrid(cn, ct,
                               indexing="ij"); x = CT; y = CN / CT. Regions
                               of the plot with no underlying grid cell
                               (inherent to a curvilinear mesh) render
                               blank.
      - else:                  x = ct, y = cn

    s_hat[i, j] <-> row axis[i], ct[j]
    """

    if "s_hat" not in gsR:
        raise KeyError("gsR missing 's_hat'")
    if "ct" not in gsR:
        raise KeyError("gsR needs 'ct'")

    S = np.asarray(gsR["s_hat"], float)   # (n_row, n_ct)
    ct = np.asarray(gsR["ct"], float)
    cn_sum_best = float(gsR["cn_sum_best"])
    ct_sum_best = float(gsR["ct_sum_best"])
    s_hat_best = float(gsR.get("s_hat_best", np.nan))
    mode = gsR.get("mode", "?")
    domain = gsR.get("domain", "time")
    a = gsR.get("a", None)
    ratio_best = cn_sum_best / ct_sum_best

    use_ratio = "cn_ratio" in gsR
    if use_ratio:
        y = np.asarray(gsR["cn_ratio"], float)
        x = ct
        S_plot = S                            # (n_ratio, n_ct)
        xlabel = r"$c_t$ [m/Pa]"
        ylabel = r"$c_n / c_t$"
        x_star = ct_sum_best
        y_star = ratio_best
    else:
        if "cn" not in gsR:
            raise KeyError("gsR needs 'cn' or 'cn_ratio'")
        cn = np.asarray(gsR["cn"], float)
        S_plot = S                            # (n_cn, n_ct)
        if ratio_axis:
            CN, CT = np.meshgrid(cn, ct, indexing="ij")   # both (n_cn, n_ct)
            x = CT
            y = CN / CT
            xlabel = r"$c_t$ [m/Pa]"
            ylabel = r"$c_n / c_t$"
            x_star = ct_sum_best
            y_star = ratio_best
        else:
            x = ct
            y = cn
            xlabel = r"$c_t$ [m/Pa]"
            ylabel = r"$c_n$ [m/Pa]"
            x_star = ct_sum_best
            y_star = cn_sum_best

    bad = ~np.isfinite(S_plot) | (S_plot <= 0)
    S_mesh = np.ma.masked_where(bad, S_plot)
    n_neg = int(np.count_nonzero(~np.isfinite(S) | (S <= 0)))

    pos = S[np.isfinite(S) & (S > 0)]
    if pos.size == 0:
        raise ValueError("no positive s_hat values to plot")

    vmin = float(np.min(pos))
    vmax = float(np.max(pos))
    if vmax_cap is not None:
        vmax = min(vmax, float(vmax_cap))
    if vmin >= vmax:
        vmax = vmin * 1.01 if vmin > 0 else 1.0

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color=neg_color)

    fig, ax = plt.subplots(figsize=figsize)
    pcm = ax.pcolormesh(
        x, y, S_mesh,
        shading="nearest",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    title = (
        rf"Scale $\hat{{s}}$, mode={mode}, domain={domain}"
        f"\n$c_n$={cn_sum_best:.2e}, $c_t$={ct_sum_best:.2e}, "
        f"$c_n/c_t$={ratio_best:.3g}"
        f"\n$\\hat{{s}}_{{\\mathrm{{best}}}}$={s_hat_best:.3g}"
        f", n(s_hat<=0)={n_neg}"
    )
    if a is not None:
        title += f", a={float(a):.3g}"
    ax.set_title(title, fontsize=11)

    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$\hat{s}>0$ (log)")
    cb.formatter = LogFormatterSciNotation(base=10)
    cb.update_ticks()

    ax.plot(
        x_star, y_star,
        "w*" if neg_color == "black" else "k*",
        ms=12,
        markeredgecolor="k",
        markeredgewidth=0.6,
        label=r"$C_{\mathrm{sp}}$ min",
    )
    ax.plot([], [], ls="none", marker="s", color=neg_color, label="s_hat <= 0")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    plt.show()
    return fig, ax


def _save_plot(fig, pkl_path, name, dpi=150):
    """
    Purpose: save a density plot immediately to <pkl_path's parent
        dir>/other_plots/<name>.png (creating that subfolder if needed).
        Used by the one-shot plotting functions (plot_ratiosearch,
        plot_cnct_search) -- there is no reason to wait for a manual
        trigger since the figure's final content is already known when
        it's built. Contrast pick_ratiosearch, the interactive
        click-to-pick plot, which stays open for a picking session and
        instead saves on demand (Save button / Esc) via _bind_save_key, so the saved
        image can reflect whatever picks are on it at the time.
    Input: fig; pkl_path (str/Path, the run's .pkl -- other_plots/ is
        created next to it, matching plot_waveform_picks' waveform_plots/
        convention); name (str, filename stem, no extension -- callers
        encode their own options into it, e.g. "{which}_{output}"); dpi
        (resolution of the PNG, default 150).
    Output: None (side effect: writes the PNG right away).
    """
    out_dir = Path(pkl_path).parent / "other_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{name}.png"
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"saved {save_path}", flush=True)


def _bind_save_key(fig, pkl_path, name, key="escape"):
    """
    Purpose: give an interactive figure an on-demand "save" trigger that
        writes it to <pkl_path's parent dir>/other_plots/<name>.png
        (creating that subfolder on first save) -- lets pick_ratiosearch's
        interactive plot be saved once the picks on it look right,
        without re-running the cell with fig.savefig(...) by hand.
        Two triggers, both calling the same save:
          1) a "Save figure" ipywidgets Button displayed under the plot --
             the reliable one in VS Code notebooks, where editor shortcuts
             (e.g. 's' -> save-file prompt, Esc -> leave cell edit mode)
             can swallow keypresses before they reach the canvas;
          2) a keypress (`key`, default Esc) on the focused canvas (click
             inside the plot first with %matplotlib widget).
        The button is skipped silently outside a Jupyter front-end (e.g.
        a non-interactive Agg backend in tests/).
    Input: fig (the Figure to bind on); pkl_path (str/Path, the run's
        .pkl -- other_plots/ is created next to it); name (str, filename
        stem, no extension); key (str, matplotlib key name for the
        keypress trigger, default "escape").
    Output: None (connects the handlers / displays the button as a side
        effect; does not save immediately).
    """
    out_dir = Path(pkl_path).parent / "other_plots"
    save_path = out_dir / f"{name}.png"

    def _save(*_):
        # shared save action for both the button and the keypress
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved {save_path}", flush=True)

    def on_key(event):
        if event.key == key:
            _save()

    fig.canvas.mpl_connect("key_press_event", on_key)

    # button under the figure; only meaningful inside a Jupyter front-end
    try:
        import ipywidgets as widgets
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            btn = widgets.Button(description="Save figure", icon="save",
                                 tooltip=f"save to {save_path}")
            btn.on_click(_save)
            display(btn)
    except ImportError:
        pass


def _run_grid(pkl_path, pack):
    """
    Purpose: find the grid type of a search run -- "ratio" ((cn/ct, ct)
        grid), "ratio_cn" ((cn/ct, cn) grid) or "cnct" ((cn, ct) grid) --
        from the run's *_data.pkl
        (cnct_search() writes it next to <run>.pkl):
          search_fn == "cnct_searchR_obs"            -> search_param[1]
                        ("cn_ctratio" -> "ratio", "non_ratio" -> "cnct");
          search_fn == "cnct_ratiosearch_Rtime_obs"  -> "ratio";
          search_fn == "cnct_search_Rtime_Gd_cc_obs" -> "cnct".
        Falls back to the pack's own keys ('cn_ratio' -> ratio) when there
        is no *_data.pkl (e.g. a pickle saved by hand).
    Input: pkl_path (str/Path, <run>.pkl); pack (dict, out_full).
    Output: "ratio" or "cnct".
    """
    pkl_path = Path(pkl_path)
    data_path = pkl_path.with_name(f"{pkl_path.stem}_data.pkl")
    grid = None
    if data_path.exists():
        with open(data_path, "rb") as f:
            d = pickle.load(f)
        fn = d.get("search_fn")
        if fn in ("cnct_searchR_obs", "cnct_search_RfromT"):
            grid = {"cn_ctratio": "ratio", "ratio_ct": "ratio", "ratio_cn": "ratio_cn",
                    "non_ratio": "cnct"}[d["search_param"][1]]
        elif fn == "cnct_ratiosearch_Rtime_obs":
            grid = "ratio"
        elif fn == "cnct_search_Rtime_Gd_cc_obs":
            grid = "cnct"
    if pack.get("grid") == "ratio_cn":                  # (cn/ct, cn) grid, axis 'cn'
        pack_grid = "ratio_cn"
    else:
        pack_grid = "ratio" if "cn_ratio" in pack else "cnct"
    if grid is not None and grid != pack_grid:
        raise ValueError(f"{data_path.name} says grid={grid} but the pack is a {pack_grid} grid")
    return pack_grid if grid is None else grid


def _load_search(pkl_path):
    """
    Purpose: load any search-result pickle -- a single dict (full-mode
        cnct_searchR_obs, cnct_search_Rtime_Gd_cc_obs) or an
        (out_full, out_constraint) tuple (constrained cnct_searchR_obs,
        cnct_ratiosearch_Rtime_obs) -- and its grid type.
    Input: pkl_path (str/Path).
    Output: (out_full, out_constraint or None, grid) with grid "ratio"/"cnct".
    """
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    out_full, out_constraint = (res if isinstance(res, tuple) else (res, None))
    return out_full, out_constraint, _run_grid(pkl_path, out_full)


def _log_edges(v):
    """
    Purpose: cell edges of a log-spaced 1-D axis, midpoints taken in
        log10 space (geometric means) -- pcolormesh's shading="nearest"
        takes LINEAR midpoints, which on a coarse log grid can put the
        first edge <= 0 and cut the first row/column off a log axis.
    Input: v (1-D positive axis, >= 2 values).
    Output: (v.size + 1,) edges.
    """
    lv = np.log10(np.asarray(v, float))
    mid = 0.5 * (lv[1:] + lv[:-1])
    return 10 ** np.concatenate([[lv[0] - (mid[0] - lv[0])], mid, [lv[-1] + (lv[-1] - mid[-1])]])


def _mesh(ax, x, y, Z, **kw):
    """
    Purpose: pcolormesh on log axes -- 1-D axes get log-space cell edges
        (_log_edges, shading="flat"); 2-D curvilinear meshes (ratio_axis
        remap) keep shading="nearest".
    Input: ax; x, y (1-D axes or 2-D meshes); Z (n_y, n_x); **kw to pcolormesh.
    Output: the QuadMesh.
    """
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        return ax.pcolormesh(_log_edges(x), _log_edges(y), Z, shading="flat", **kw)
    return ax.pcolormesh(x, y, Z, shading="nearest", **kw)


def _ratio_ticks(axis):
    """
    Purpose: plain-number ticks on a log-scaled cn/ct axis (0.01, 0.02, 0.05,
        0.1, 0.2, 0.5, 1, 2) instead of 10^k labels.
    Input: axis (ax.xaxis or ax.yaxis, already log-scaled).
    Output: None (sets locators and formatters).
    """
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FormatStrFormatter("%g"))
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10)))
    axis.set_minor_formatter(NullFormatter())

def _best_txt(cn, ct):
    """
    Purpose: best-node text shared by all search-map titles:
        "(c_n, c_t, c_n/c_t) = (1.93e-10, 5.52e-11, 3.5)".
    Input: cn, ct (floats, actual compliances at the best node [m/Pa]).
    Output: str (matplotlib mathtext).
    """
    return rf"$(c_n, c_t, c_n/c_t)$ = ({cn:.2e}, {ct:.2e}, {cn / ct:.3g})"


def _grid_axes(gsR, grid, ratio_axis=False):
    """
    Purpose: plot axes for one search pack. x is ct, except
        grid="ratio_cn": x = cn (native axis 'cn'), y = cn/ct; y is
        grid="ratio": cn/ct (native axis 'cn_ratio');
        grid="cnct" : cn (native axis 'cn'), or cn/ct when ratio_axis=True
                      (curvilinear remap: x = CT, y = CN/CT, 2-D meshes;
                      cells with no grid node stay blank).
    Input: gsR (pack dict); grid ("ratio"/"cnct"); ratio_axis (bool, only
        for grid="cnct").
    Output: dict x, y (1-D axes, or 2-D meshes for the remap), xlabel,
        ylabel, x_best, y_best (Csp_sum-minimum position, None if no node
        complies) and best_txt (title text for that node).
    """
    if grid == "ratio_cn":                             # x = cn axis, y = cn/ct; ct = cn/ratio
        x = np.asarray(gsR["cn"], float)
        y = np.asarray(gsR["cn_ratio"], float)
        cn_b, y_b, ct_b = gsR.get("cn_sum_best"), gsR.get("ratio_sum_best"), gsR.get("ct_sum_best")
        txt = None if cn_b is None else _best_txt(cn_b, ct_b)
        return {"x": x, "y": y, "xlabel": r"$c_n$ [m/Pa]", "ylabel": r"$c_n / c_t$",
                "x_best": cn_b, "y_best": y_b, "best_txt": txt}

    ct = np.asarray(gsR["ct"], float)
    ct_b = gsR.get("ct_sum_best")
    if grid == "ratio":
        y = np.asarray(gsR["cn_ratio"], float)
        y_b = gsR.get("ratio_sum_best")
        ylabel = r"$c_n / c_t$"
        txt = None if y_b is None else _best_txt(y_b * ct_b, ct_b)
        return {"x": ct, "y": y, "xlabel": r"$c_t$ [m/Pa]", "ylabel": ylabel,
                "x_best": ct_b, "y_best": y_b, "best_txt": txt}

    cn = np.asarray(gsR["cn"], float)
    cn_b = gsR.get("cn_sum_best")
    txt = None if cn_b is None else _best_txt(cn_b, ct_b)
    if ratio_axis:
        CN, CT = np.meshgrid(cn, ct, indexing="ij")          # (n_cn, n_ct)
        y_b = None if cn_b is None else cn_b / ct_b
        return {"x": CT, "y": CN / CT, "xlabel": r"$c_t$ [m/Pa]", "ylabel": r"$c_n / c_t$",
                "x_best": ct_b, "y_best": y_b, "best_txt": txt}
    return {"x": ct, "y": cn, "xlabel": r"$c_t$ [m/Pa]", "ylabel": r"$c_n$ [m/Pa]",
            "x_best": ct_b, "y_best": cn_b, "best_txt": txt}


def _plot_search_cost(gsR, which, figsize, grid="ratio", kind="cost", rel_max=None,
                      ratio_axis=False):
    """
    Purpose: cost density plot of one search pack, for either grid (y =
    cn/ct for the ratio grid, cn -- or cn/ct with ratio_axis -- for the
    (cn, ct) grid; x = ct), in one of three scalings (same minimum
    location in all three; only the color scale differs):
      kind="cost"      : summed misfit Csp_sum, log color scale.
      kind="cost_norm" : normalized misfit gsR['cost_norm'] = Csp_sum /
                         data_energy (NRMS^2, 0 = perfect fit, 1 = model
                         explains none of the data), linear color scale;
                         comparable across groups/runs. Needs a run made
                         after cost_norm was added to the search.
      kind="cost_rel"  : relative misfit Csp_sum / min(Csp_sum) (>= 1),
                         linear color scale from 1 to rel_max (values
                         above saturate) -- shows how flat or sharp the
                         minimum is; works for any run.
    Input: gsR (one pack -- out_full or out_constraint); which
        ("full"/"constraint", used only for the title); figsize; grid
        ("ratio"/"cnct"); kind; rel_max (float or None, only for
        kind="cost_rel": colorbar upper bound, e.g. 1.2 = within 20% of
        the minimum; None -> the grid maximum); ratio_axis (grid="cnct" only).
    Output: (fig, ax).
    """
    g = _grid_axes(gsR, grid, ratio_axis)
    Csp = np.asarray(gsR["Csp_sum"], float)   # (n_y, n_ct)
    mode = gsR.get("mode", "?")
    domain = gsR.get("domain", "?")
    dsub = "f" if which == "full" else "c"           # cost symbol: C_f (full) or C_c (constrained)

    # NaN entries compare False to > 0, so they're excluded from `pos`
    pos = Csp[Csp > 0]
    if pos.size == 0:
        n_compliant = gsR.get("n_compliant")
        extra = f" ({n_compliant} grid nodes comply)" if n_compliant is not None else ""
        raise ValueError(f"Csp_sum has no positive values to plot{extra}")
    Csp_plot = np.clip(Csp, np.nanmin(pos), None)

    # map to plot + color scale per kind (nanmin/nanmax: the constrained
    # pack has NaN outside the compliant region)
    extend = "neither"
    if kind == "cost":                               # log10 values, linear colors (= LogNorm colors)
        Z = np.log10(Csp_plot)
        norm = Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))      # actual min / max of log10 C
        cost_ticks = np.linspace(norm.vmin, norm.vmax, 6)          # ends + 4 interior ticks
        cb_label = rf"$\log_{{10}} C_{dsub}$"
    elif kind == "cost_norm":
        if "cost_norm" not in gsR:
            raise KeyError("pack has no 'cost_norm' (run made before it was added) "
                           "-- rerun the search, or use output='cost_rel'")
        Z = np.asarray(gsR["cost_norm"], float)
        norm = LogNorm(vmin=np.nanmin(Z), vmax=np.nanmax(Z))           # log colors, plain tick labels
        cb_label = rf"$C_{{{dsub},\mathrm{{norm}}}}$"              # = Csp_sum / data energy
    elif kind == "cost_rel":
        Z = Csp / np.nanmin(pos)                     # >= 1, NaN kept
        vmax = float(rel_max) if rel_max is not None else float(np.nanmax(Z))
        norm = LogNorm(vmin=1.0, vmax=vmax)                            # log colors, plain tick labels
        extend = "max" if np.nanmax(Z) > vmax else "neither"
        cb_label = rf"$C_{dsub} / C_{{{dsub},\mathrm{{min}}}}$"
    else:
        raise ValueError('kind must be "cost", "cost_norm" or "cost_rel"')

    fig, ax = plt.subplots(figsize=figsize)
    pcm = _mesh(ax, g["x"], g["y"], Z, cmap="turbo", norm=norm)
    # log-scaled axes (10^x tick labels)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(g["xlabel"])
    ax.set_ylabel(g["ylabel"])
    if "c_n / c_t" in g["ylabel"]:                    # cn/ct axis: plain numbers 0.01 ... 2
        _ratio_ticks(ax.yaxis)

    # title: "Cost function C_t (which)" / c_t, c_n/c_t / C_t,min, s_t,best
    # (C_f, s_f for frequency-domain runs)
    which_txt = "Full" if which == "full" else "Constrained"
    head = {"cost": rf"{which_txt} cost function ($C_{dsub}$)",
            "cost_norm": rf"{which_txt} normalized cost function ($C_{{{dsub},\mathrm{{norm}}}}$)",
            "cost_rel": rf"{which_txt} relative cost function ($C_{dsub}/C_{{{dsub},\mathrm{{min}}}}$)"}[kind]
    title = head

    s_hat_best = gsR.get("s_hat_best")
    s_in_cost = "no s_hat" not in str(gsR.get("cost", ""))   # RfromT: s_hat is a diagnostic only
    if g["best_txt"] is not None:
        title += f"\nbest values: {g['best_txt']}"
        if s_hat_best is not None and s_in_cost:
            title += rf", $\hat{{s}}$={s_hat_best:.3g}"
        # red star marks the cost minimum (values already in the title)
        ax.plot(g["x_best"], g["y_best"], "r*", ms=12)
    else:
        title += "\n(no grid node satisfies the constraint)"

    ax.set_title(title, fontsize=11)

    cb = fig.colorbar(pcm, ax=ax, pad=0.02, extend=extend)
    cb.set_label(cb_label)
    # log color bar labelled at 1, 2, 3, 5 x 10^k: compact e-notation for the raw cost
    # (2e6, 3e6, 1e7), plain numbers for cost_norm / cost_rel (0.6, 1, 2, 3, 5, 10)
    if kind == "cost":
        # log10(C) bar from the actual min to max, 6 evenly spaced ticks; fewest decimals
        # (1, up to 2) that keep the labels distinct, same number on every label
        nd = next((d for d in (1, 2) if len({f"{t:.{d}f}" for t in cost_ticks}) == cost_ticks.size), 2)
        cb.set_ticks(cost_ticks)
        cb.set_ticklabels([f"{t:.{nd}f}" for t in cost_ticks])
    else:
        # log color bar labelled at 1, 2, 3, 5 x 10^k with plain numbers (0.6, 1, 2, 3, 5, 10)
        cb.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        cb.ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10)))
        cb.ax.yaxis.set_minor_formatter(NullFormatter())

    fig.tight_layout()
    plt.show()
    return fig, ax


def _plot_search_s_hat(gsR, which, figsize, grid="ratio", neg_color="black", vmax_cap=None,
                       ratio_axis=False, linear=False):
    """
    Purpose: s_hat density plot of one search pack, for either grid (axes
    as in _plot_search_cost).
    Input: gsR (one pack); which ("full"/"constraint", used for the title
        and to know whether gsR['mask'] distinguishes constraint-excluded
        cells from genuinely bad ones); figsize; grid ("ratio"/"cnct");
        neg_color (fill for genuinely s_hat<=0/non-finite cells);
        vmax_cap (colorbar upper-bound cap); ratio_axis (grid="cnct" only);
        linear (bool, linear colorbar -- for a constrained pack, whose
        s_hat is limited to the s range; default log).
    Output: (fig, ax).
    """
    g = _grid_axes(gsR, grid, ratio_axis)
    S = np.asarray(gsR["s_hat"], float)   # (n_y, n_ct)
    mode = gsR.get("mode", "?")
    domain = gsR.get("domain", "time")
    a = gsR.get("a", None)
    mask = gsR.get("mask")   # bool grid, True=compliant; None for the unconstrained pack

    # NaN in a constrained pack's s_hat means "excluded by the s_hat
    # constraint", NOT "computed and found <=0": excluded -> blank,
    # genuinely <=0/non-finite WITHIN the compliant region -> neg_color.
    not_finite_or_nonpos = ~np.isfinite(S) | (S <= 0)
    if mask is not None:
        mask = np.asarray(mask, bool)
        excluded = ~mask
        bad = mask & not_finite_or_nonpos
    else:
        excluded = np.zeros(S.shape, bool)
        bad = not_finite_or_nonpos

    n_bad = int(np.count_nonzero(bad))

    pos = S[np.isfinite(S) & (S > 0)]
    if pos.size == 0:
        n_compliant = gsR.get("n_compliant")
        extra = f" ({n_compliant} grid nodes comply)" if n_compliant is not None else ""
        raise ValueError(f"no positive s_hat values to plot{extra}")

    vmin = float(np.min(pos))
    vmax = float(np.max(pos))
    if vmax_cap is not None:
        vmax = min(vmax, float(vmax_cap))
    if vmin >= vmax:
        vmax = vmin * 1.01 if vmin > 0 else 1.0

    fig, ax = plt.subplots(figsize=figsize)

    # layer 1 (underneath): solid neg_color block ONLY at genuinely bad
    # cells within the compliant region
    if n_bad:
        bad_mesh = np.ma.masked_where(~bad, np.zeros(S.shape))
        _mesh(
            ax, g["x"], g["y"], bad_mesh,
            cmap=ListedColormap([neg_color]),
            vmin=0, vmax=1,
        )

    # layer 2 (on top): s_hat, NaN (transparent) at excluded and bad cells
    S_main = np.where(excluded | bad, np.nan, S)
    pcm = _mesh(
        ax, g["x"], g["y"], S_main,
        cmap="turbo",
        norm=Normalize(vmin=vmin, vmax=vmax) if linear else LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(g["xlabel"])
    ax.set_ylabel(g["ylabel"])
    if "c_n / c_t" in g["ylabel"]:                    # cn/ct axis: plain numbers 0.01 ... 2
        _ratio_ticks(ax.yaxis)

    s_hat_best = gsR.get("s_hat_best")
    dsub = "f" if domain == "frequency" else "t"   # s_t (time) or s_f (frequency)

    # title: "Scale s_t (which)" / c_t, c_n/c_t, s_t,best
    title = rf"Scale $\hat{{s}}_{dsub}$ ({which})"
    if g["best_txt"] is not None:
        title += f"\n{g['best_txt']}"
        if s_hat_best is not None:
            title += rf", $\hat{{s}}_{{{dsub},\mathrm{{best}}}}$={s_hat_best:.3g}"
        ax.plot(
            g["x_best"], g["y_best"],
            "w*" if neg_color == "black" else "k*",
            ms=12, markeredgecolor="k", markeredgewidth=0.6,
            label=rf"$C_{dsub}$ min",
        )
    else:
        title += "\n(no grid node satisfies the constraint)"

    ax.set_title(title, fontsize=11)

    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    if linear:
        cb.set_label(r"$\hat{s}$")
    else:
        cb.set_label(r"$\hat{s}>0$ (log)")
        cb.formatter = LogFormatterSciNotation(base=10)
        cb.update_ticks()

    if n_bad:
        ax.plot([], [], ls="none", marker="s", color=neg_color, label="s_hat<=0")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    plt.show()
    return fig, ax


def _plot_ratiosearch_cost_sum(gsR, which, figsize, kind="cost", rel_max=None):
    """
    Purpose: cost half of plot_ratiosearch -- (cn/ct ratio, ct) grid;
        now a thin wrapper of _plot_search_cost(grid="ratio") (same plot).
    Input/Output: see _plot_search_cost.
    """
    if "cn_ratio" not in gsR or "ct" not in gsR:
        raise KeyError("gsR needs 'cn_ratio' and 'ct' (cnct_ratiosearch_Rtime_obs output)")
    return _plot_search_cost(gsR, which, figsize, grid="ratio", kind=kind, rel_max=rel_max)


def _plot_ratiosearch_s_hat(gsR, which, figsize, neg_color="black", vmax_cap=None):
    """
    Purpose: s_hat half of plot_ratiosearch -- (cn/ct ratio, ct) grid;
        now a thin wrapper of _plot_search_s_hat(grid="ratio") (same plot).
    Input/Output: see _plot_search_s_hat.
    """
    if "s_hat" not in gsR or "cn_ratio" not in gsR or "ct" not in gsR:
        raise KeyError("gsR needs 'cn_ratio', 'ct', and 's_hat' (cnct_ratiosearch_Rtime_obs output)")
    return _plot_search_s_hat(gsR, which, figsize, grid="ratio", neg_color=neg_color,
                              vmax_cap=vmax_cap)


def _ratiosearch_load_and_plot(pkl_path, which, output, figsize, neg_color="black", vmax_cap=None,
                               rel_max=None):
    """
    Purpose: shared load+dispatch step behind plot_ratiosearch and
        pick_ratiosearch -- loads the (out_full, out_constraint) pickle,
        selects the requested pack, and builds the requested density plot.
        Deliberately does NOT decide what happens to the resulting figure
        (auto-save vs. interactive picking is the caller's job) -- that's
        why pick_ratiosearch calls this directly instead of
        plot_ratiosearch, to avoid plot_ratiosearch's own unconditional
        auto-save firing before any picking has happened.
    Input: same pkl_path/which/output/figsize/neg_color/vmax_cap/rel_max
        as plot_ratiosearch.
    Output: (gsR, fig, ax) -- gsR is the selected pack dict (pick_ratiosearch
        needs its grid axes/s_hat array), fig/ax the plot.
    """
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    out_full, out_constraint = res if isinstance(res, tuple) else (res, None)

    if which == "full":
        gsR = out_full
    elif which == "constraint":
        if out_constraint is None:
            raise ValueError(f"{Path(pkl_path).name} is a full-mode run (no constrained pack) "
                             "-- use which='full'")
        gsR = out_constraint
    else:
        raise ValueError('which must be "full" or "constraint"')

    if output in ("cost", "cost_norm", "cost_rel"):
        fig, ax = _plot_ratiosearch_cost_sum(gsR, which, figsize, kind=output, rel_max=rel_max)
    elif output == "s_hat":
        fig, ax = _plot_ratiosearch_s_hat(gsR, which, figsize, neg_color=neg_color, vmax_cap=vmax_cap)
    else:
        raise ValueError('output must be "cost", "cost_norm", "cost_rel" or "s_hat"')
    if "Q_hat" in gsR and np.isfinite(np.asarray(gsR["Q_hat"], float)).any():   # Q run: title with Q_best
        s_r = gsR.get("s_hat_range")
        _q_title(ax, gsR, "ratio", output, "global" if which == "full" else
                 "constraint" + ("" if s_r is None else f" s {s_r[0]:g}-{s_r[1]:g}"))

    return gsR, fig, ax


def _reconcile_loaded_picks(picks, best_pick, keep_fn, which, save_path):
    """
    Purpose: make picks resumed with load_existing=True consistent with the
        pack being displayed. The picks file (<run>_picks.json) is shared by
        which="full" and which="constraint" sessions and survives a rerun of
        the search, so a resumed file can start from another pack's best fit.
    Input: picks (list of pick dicts, modified in place); best_pick (dict,
        the displayed pack's best fit, or None); keep_fn (callable(pick) ->
        bool, False for a loaded click that is not allowed in this session,
        e.g. outside the constraint mask); which ("full"/"constraint");
        save_path (for the messages).
    Output: None (picks updated in place: pick 1 = displayed best fit,
        disallowed clicks dropped, picks renumbered 1..n).
    """
    if best_pick is not None:
        first = picks[0] if picks else None
        same = (first is not None and first.get("source") == "best_fit"
                and np.isclose(first.get("ct", np.nan), best_pick["ct"], rtol=1e-6)
                and np.isclose(first.get("cn_ratio", np.nan), best_pick["cn_ratio"], rtol=1e-6))
        if not same:
            if first is not None and first.get("source") == "best_fit":
                print(f"{save_path.name}: stored best fit (cn/ct={first.get('cn_ratio', np.nan):.3g}, "
                      f"ct={first.get('ct', np.nan):.3g}, s_hat={first.get('s_hat', np.nan):.3g}) "
                      f"is not the best fit of the displayed '{which}' pack -- replaced", flush=True)
                picks.pop(0)
            else:
                print(f"{save_path.name}: no best-fit entry first -- added the '{which}' best fit",
                      flush=True)
            picks.insert(0, best_pick)
    n0 = len(picks)
    picks[:] = [p for k, p in enumerate(picks) if k == 0 or keep_fn(p)]
    if len(picks) < n0:
        print(f"{save_path.name}: dropped {n0 - len(picks)} loaded pick(s) not allowed for "
              f"which='{which}' (outside the s_hat constraint)", flush=True)
    for k, p in enumerate(picks, 1):
        p["pick"] = k


def pick_ratiosearch(pkl_path, which="constraint", output="cost",
                      save_name=None, load_existing=False, figsize=(7, 4), group=None,
                      **plot_kwargs):
    """
    Interactive click-to-pick on a plot_ratiosearch density plot.

    Displays the same plot plot_ratiosearch would. If load_existing=True
    and a picks file already exists (from an earlier pick_ratiosearch
    call on this same pkl_path/save_name), that file's picks are loaded
    and shown as the starting point, so you can resume/add to a previous
    session instead of starting over. Otherwise (load_existing=False, the
    default, or no existing file yet), pick 1 is added automatically from
    the DISPLAYED pack's own best fit (ratio_sum_best, ct_sum_best,
    s_hat_best) -- right-click never removes it. Each LEFT-click after
    that finds the nearest (cn_ratio, ct) grid node to the click location
    (nearest in log10 space, matching the plot's log-scaled axes). With
    which="constraint", a node that does NOT satisfy the constraint pack's
    s_hat range (its 'mask') is rejected with a printed message and NOT
    recorded; with which="full" every node can be picked (s_hat read from
    the full pack, so it may lie outside the constraint range -- a
    message flags it). An accepted click appends a (pick, source, which,
    cn_ratio, ct, s_hat[, Q_hat]) row to the picks table (Q_hat only for
    runs with a Q estimate -- cnct_searchR_obs with Q_opts or a spectrum
    run with Q free; values read from the DISPLAYED pack, so for a
    constrained Q run they are the Q / s_hat chosen under the constraint), marked on the plot with a small 'x', and the
    whole table is written to a JSON file after every accepted
    click/undo, so nothing is lost mid-session. RIGHT-click removes the
    most recent user click (not the auto-added best fit, nor anything
    loaded via load_existing). Requires an interactive matplotlib backend
    (%matplotlib widget in the notebook) -- a plain inline backend does
    not deliver click events.

    Purpose
    -------
    Let the user pick points off a ratiosearch density plot by clicking
    (restricted to the s_hat-compliant region for which="constraint",
    anywhere on the grid for which="full"), with the
    best-fit node recorded automatically as the first entry, and the
    option to resume a previous picking session instead of starting over.

    Input
    -----
    pkl_path : str or Path
        Path to a ratio-grid search result pickle (as saved by
        cnct_search() / cnct_search_scat() / run_search_driver.py): a
        (cn/ct, ct) grid ("ratio_ct", formerly "cn_ctratio", or a legacy
        cnct_ratiosearch_Rtime_obs run) or a (cn/ct, cn) grid ("ratio_cn",
        plotted with cn on the x-axis; clicks snap to (cn/ct, cn) nodes).
        The picks file is written next to it, in the same directory.
    which : {"full", "constraint"}
        Which pack to plot/pick from -- see plot_ratiosearch.
        "constraint": picks must lie in the constrained pack's compliant-
        node mask. "full": any grid node can be picked, including nodes
        outside the s_hat constraint.
    output : {"cost", "cost_norm", "cost_rel", "s_hat"}
        Which map to DISPLAY while picking -- see plot_ratiosearch. The
        recorded value at each pick is always s_hat, regardless of which
        map is shown.
    save_name : str or None
        Picks file base name; defaults to "<pkl stem>_picks.json".
    load_existing : bool
        If True and that picks file already exists, load and resume from
        it instead of starting fresh with just the auto-added best fit.
        Has no effect if the file doesn't exist yet.
    figsize : tuple
        Passed to plot_ratiosearch.
    group : None or int (0, 1, 2, ...), only names the saved PNG: None
        (default) <which>_<output>_picks.png; g group<g>_<which>_<output>_picks.png
        (pkl_path = a group<g>_<run>.pkl from DAS_modeling.extracting_search;
        its picks JSON is already group<g>_<run>_picks.json). Group files
        have no s_hat: recorded s_hat is NaN.
    **plot_kwargs :
        Extra keyword arguments passed through to plot_ratiosearch (e.g.
        neg_color, vmax_cap for output="s_hat"; rel_max for "cost_rel").

    Output
    ------
    (fig, ax, picks) : the plot's Figure/Axes and the live picks list
        (list of dicts with keys pick, source ("best_fit" or "click"),
        which, cn_ratio, ct, cn (actual compliances at the node), s_hat
        [, Q_hat]; picks[0] is the auto-added best fit unless
        load_existing resumed a file that didn't start with one) -- also
        continuously written to <pkl_path's directory>/save_name as a
        JSON array, so `json.load(open(save_path))` hands back that same
        list directly (numeric fields stay numeric, no parsing needed).
        Clicking the "Save figure" button shown under the plot (or
        pressing Esc with the canvas focused -- click inside it first)
        saves a snapshot of the figure -- picks and all -- to
        <pkl_path's parent dir>/other_plots/[group<g>_]<which>_<output>_picks.png.
    """
    pkl_path = Path(pkl_path)
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    out_full, out_constraint = res if isinstance(res, tuple) else (res, None)

    # compliant-node mask: enforced on clicks only for which="constraint";
    # for which="full" it is only used to flag out-of-range picks
    # (full-mode run: no constrained pack -> every node allowed)
    if out_constraint is None:
        constraint_mask = np.ones(np.shape(out_full["Csp_sum"]), bool)
        s_hat_range = None
    else:
        constraint_mask = np.asarray(out_constraint["mask"], bool)
        s_hat_range = out_constraint.get("s_hat_range")
    enforce_mask = (which == "constraint")

    # _ratiosearch_load_and_plot (not plot_ratiosearch) -- this is the
    # interactive session, so the figure is saved on demand (button/Esc)
    # below instead of plot_ratiosearch's own unconditional auto-save
    is_rcn = out_full.get("grid") == "ratio_cn"         # (cn/ct, cn) grid: x axis is cn
    if is_rcn:
        gsR, _, _, fig, ax = _cnct_load_and_plot(pkl_path, which, output, figsize, **plot_kwargs)
    else:
        gsR, fig, ax = _ratiosearch_load_and_plot(
            pkl_path, which, output, figsize, **plot_kwargs
        )
    cn_ratio = np.asarray(gsR["cn_ratio"], float)
    x_ax = np.asarray(gsR["cn"] if is_rcn else gsR["ct"], float)   # horizontal axis values
    # s_hat map (NaN when the pack has none, e.g. group files of extracting_search)
    s_hat = (np.asarray(gsR["s_hat"], float) if "s_hat" in gsR
             else np.full(np.shape(gsR["Csp_sum"]), np.nan))
    # Q of the displayed pack (runs with a Q estimate only)
    Q_hat = np.asarray(gsR["Q_hat"], float) if "Q_hat" in gsR else None
    if Q_hat is not None and not np.isfinite(Q_hat).any():
        Q_hat = None                                     # e.g. spectrum run with beta = 0
    log_ratio_axis = np.log10(cn_ratio)
    log_x_axis = np.log10(x_ax)

    def _node(i, j):
        """actual (cn, ct) of node (cn_ratio[i], x_ax[j])"""
        if is_rcn:
            return float(x_ax[j]), float(x_ax[j] / cn_ratio[i])
        return float(cn_ratio[i] * x_ax[j]), float(x_ax[j])

    def _xval(p):
        """horizontal-axis value of a pick (cn for ratio_cn, ct otherwise); old
        picks files have no 'cn' key -> cn = cn_ratio * ct"""
        if is_rcn:
            return p.get("cn", p["cn_ratio"] * p["ct"])
        return p["ct"]

    prefix = "" if group is None else f"group{int(group)}_"   # output PNG name only
    _bind_save_key(fig, pkl_path, f"{prefix}{which}_{output}_picks")

    if save_name is None:
        save_name = f"{pkl_path.stem}_picks.json"
    save_path = pkl_path.parent / save_name

    picks = []
    marker_line, = ax.plot([], [], "wx", ms=6, mew=1.5, mec="k", zorder=10, label="picks")

    def _save():
        with open(save_path, "w") as f:
            json.dump(picks, f, indent=2)

    def _redraw():
        marker_line.set_data([_xval(p) for p in picks], [p["cn_ratio"] for p in picks])
        fig.canvas.draw_idle()

    # the displayed pack's own best fit (pick 1)
    best_pick = None
    if gsR.get("ratio_sum_best") is not None and gsR.get("ct_sum_best") is not None:
        best_pick = {
            "pick": 1,
            "source": "best_fit",
            "which": which,
            "cn_ratio": float(gsR["ratio_sum_best"]),
            "ct": float(gsR["ct_sum_best"]),            # actual ct (also for ratio_cn)
            "cn": float(gsR["ratio_sum_best"] * gsR["ct_sum_best"]),
            "s_hat": float(gsR.get("s_hat_best", np.nan)),
        }
        if Q_hat is not None:                            # Q at the displayed pack's best node
            ib, jb = np.unravel_index(np.nanargmin(np.asarray(gsR["Csp_sum"], float)), Q_hat.shape)
            best_pick["Q_hat"] = float(Q_hat[ib, jb])

    def _allowed(p):
        """Loaded click allowed here: inside the mask for which='constraint'."""
        if not enforce_mask:
            return True
        j = int(np.argmin(np.abs(log_x_axis - np.log10(_xval(p)))))
        i = int(np.argmin(np.abs(log_ratio_axis - np.log10(p["cn_ratio"]))))
        return bool(constraint_mask[i, j])

    if load_existing and save_path.exists():
        # resume a previous session -- extend in place, since on_click's
        # closure below captures `picks` by reference, not by reassignment;
        # then make pick 1 the DISPLAYED pack's best fit (the file is shared
        # by full/constraint sessions and survives reruns) and drop clicks
        # outside the constraint for which="constraint"
        with open(save_path) as f:
            picks.extend(json.load(f))
        print(f"loaded {len(picks)} existing pick(s) from {save_path}", flush=True)
        _reconcile_loaded_picks(picks, best_pick, _allowed, which, save_path)
        _redraw()
        _save()
    elif best_pick is not None:
        # pick 1: the displayed pack's own best fit, recorded automatically
        # before any click -- kept regardless of what clicking does later
        picks.append(best_pick)
        _redraw()
        _save()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            j = int(np.argmin(np.abs(log_x_axis - np.log10(event.xdata))))
            i = int(np.argmin(np.abs(log_ratio_axis - np.log10(event.ydata))))
            cn_ij, ct_ij = _node(i, j)                  # actual compliances at the node
            if not constraint_mask[i, j]:
                if enforce_mask:
                    print(
                        f"pick at cn_ratio={cn_ratio[i]:.3g}, cn={cn_ij:.3g}, ct={ct_ij:.3g} is "
                        f"OUTSIDE the s_hat constraint range {s_hat_range} -- not recorded",
                        flush=True,
                    )
                    return
                # which="full": accepted, just flagged
                print(
                    f"pick at cn_ratio={cn_ratio[i]:.3g}, cn={cn_ij:.3g}, ct={ct_ij:.3g} "
                    f"(s_hat={s_hat[i, j]:.3g}) is outside the s_hat constraint "
                    f"range {s_hat_range} -- recorded (which='full')",
                    flush=True,
                )
            pick = {
                "pick": len(picks) + 1,
                "source": "click",
                "which": which,
                "cn_ratio": float(cn_ratio[i]),
                "ct": ct_ij,
                "cn": cn_ij,
                "s_hat": float(s_hat[i, j]),
            }
            if Q_hat is not None:
                pick["Q_hat"] = float(Q_hat[i, j])
            picks.append(pick)
            print(f"pick {pick['pick']}: cn/ct={pick['cn_ratio']:.3g}, cn={pick['cn']:.3g}, ct={pick['ct']:.3g}, "
                  f"s_hat={pick['s_hat']:.3g}"
                  + ("" if Q_hat is None else f", Q_hat={pick['Q_hat']:.3g}"), flush=True)
        elif event.button == 3:
            # never remove the auto-added best-fit entry (picks[0])
            if len(picks) > 1:
                picks.pop()
        else:
            return
        _redraw()
        _save()

    fig.canvas.mpl_connect("button_press_event", on_click)
    return fig, ax, picks


def _cnct_load_and_plot(pkl_path, which, output, figsize, ratio_axis=False,
                        neg_color="black", vmax_cap=None, rel_max=None):
    """
    Purpose: shared load + dispatch behind plot_cnct_search and
        pick_cnct_search: load any search pickle (_load_search), select
        the pack (`which`), and plot it on the run's own grid. Does NOT
        save (auto-save vs interactive save is the caller's job).
    Input: same as plot_cnct_search.
    Output: (gsR, out_constraint, grid, fig, ax).
    """
    out_full, out_constraint, grid = _load_search(pkl_path)
    if which == "full":
        gsR = out_full
    elif which == "constraint":
        if out_constraint is None:
            raise ValueError(f"{Path(pkl_path).name} is a full-mode run (no constrained "
                             "pack) -- use which='full'")
        gsR = out_constraint
    else:
        raise ValueError('which must be "full" or "constraint"')
    if ratio_axis and grid in ("ratio", "ratio_cn"):
        ratio_axis = False                         # already a cn/ct axis

    if output in ("cost", "cost_norm", "cost_rel"):
        fig, ax = _plot_search_cost(gsR, which, figsize, grid=grid, kind=output,
                                    rel_max=rel_max, ratio_axis=ratio_axis)
    elif output == "s_hat":
        fig, ax = _plot_search_s_hat(gsR, which, figsize, grid=grid, neg_color=neg_color,
                                     vmax_cap=vmax_cap, ratio_axis=ratio_axis)
    else:
        raise ValueError('output must be "cost", "cost_norm", "cost_rel" or "s_hat"')
    return gsR, out_constraint, grid, fig, ax


def plot_cnct_search(pkl_path, which="full", output="cost", figsize=(7, 4),
                     neg_color="black", vmax_cap=None, rel_max=None, ratio_axis=False,
                     group=None, dpi=400):
    """
    Density plot of ANY compliance grid-search result pickle: runs of
    cnct_searchR_obs (all four search_mode x grid combinations),
    cnct_ratiosearch_Rtime_obs and cnct_search_Rtime_Gd_cc_obs.

    Same parameters as plot_ratiosearch. The grid (vertical axis) is read
    from the run's *_data.pkl (search_fn / search_param[1], see _run_grid):
    ratio grid -> y = cn/ct; (cn, ct) grid -> y = cn; x = ct always.
    A single-dict pickle (full mode) only has which="full"; a two-pack
    pickle (constrained mode) has both.

    Purpose
    -------
    One plotting call for every search run, without the caller having to
    know how it was run.

    Input
    -----
    pkl_path : str or Path, <run>.pkl written by cnct_search().
    which : {"full", "constraint"}, pack to plot.
    output : {"cost", "cost_norm", "cost_rel", "s_hat"} -- see plot_ratiosearch;
        "Q_hat" for runs with a Q estimate (cnct_searchR_obs with Q_opts,
        cnct_search_Rspectrum with Q free): map of the chosen Q per node.
    figsize : tuple.
    neg_color, vmax_cap : output="s_hat" only (see plot_ratiosearch).
    rel_max : output="cost_rel" only (e.g. 1.2).
    ratio_axis : bool, (cn, ct) grid only -- plot cn/ct instead of cn on
        the vertical axis (curvilinear remap of the same grid; cells with
        no grid node stay blank). Ignored for the ratio grid.
    group : None or int (0, 1, 2, ...), labels the plot only (pkl_path is
        not changed): the title starts with "Group <g>:" and the saved PNG
        is group<g>_<which>_<output>.png (None, default: <which>_<output>.png),
        so group plots (pkl_path = a group<g>_<run>.pkl written by
        DAS_modeling.extracting_search) do not overwrite the run's plots.
    dpi : resolution of the saved PNG (default 400).

    which="constraint" shows the FULL map of `output` with the constraint
    drawn on it (_overlay_constraint): solid white contours at the
    s_hat_range limits, star at the global best, white x at the constrained
    best, and a title line with the constrained best when it differs from
    the global best. (pick_cnct_search still shows the constrained pack.)
    Exception -- Q-grid runs (cnct_searchR_obs with Q_opts) constrained in
    the search: the constrained pack has its OWN cost map (each node uses
    the lowest-cost Q with s_hat in range), so that map is plotted
    (_plot_constrained_own): kept region outlined, excluded nodes blank,
    white x = constrained best, small star = global best.

    Output
    ------
    (fig, ax), also saved right away to
    <pkl_path's dir>/other_plots/[group<g>_]<which>_<output>[_ratio].png.
    """
    prefix = "" if group is None else f"group{int(group)}_"   # output PNG name only
    if which == "full" and output == "Q_hat":           # Q map of the full pack
        out_full, _, grid = _load_search(pkl_path)
        fig, ax = _plot_search_Q(out_full, "full", figsize, grid=grid, vmax_cap=vmax_cap)
    elif which != "constraint":
        _, _, grid, fig, ax = _cnct_load_and_plot(
            pkl_path, which, output, figsize, ratio_axis=ratio_axis,
            neg_color=neg_color, vmax_cap=vmax_cap, rel_max=rel_max,
        )
    else:
        # constraint: FULL map + s_hat-limit contours / constrained best (_overlay_constraint)
        out_full, out_c, grid = _load_search(pkl_path)
        if out_c is None:
            raise ValueError(f"{Path(pkl_path).name} is a full-mode run (no constrained "
                             "pack) -- use which='full'")
        ratio_axis = ratio_axis and grid == "cnct"         # already a cn/ct axis otherwise
        s_r = out_c.get("s_hat_range")
        if _is_qgrid_constraint(out_full, out_c):          # Q grid: constrained pack's OWN maps
            label = "constraint" + ("" if s_r is None else f" s {s_r[0]:g}-{s_r[1]:g}") + \
                    ", Q per node under the constraint"
            fig, ax = _plot_constrained_own(out_full, out_c, grid, output, label, figsize,
                                            rel_max=rel_max, neg_color=neg_color,
                                            vmax_cap=vmax_cap, ratio_axis=ratio_axis)
            suffix = "_ratio" if (ratio_axis and grid == "cnct") else ""
            _save_plot(fig, pkl_path, f"{prefix}{which}_{output}{suffix}", dpi=dpi)
            return fig, ax
        label = "full; constraint" + ("" if s_r is None else f" s {s_r[0]:g}-{s_r[1]:g}")
        if output in ("cost", "cost_norm", "cost_rel"):
            fig, ax = _plot_search_cost(out_full, label, figsize, grid=grid, kind=output,
                                        rel_max=rel_max, ratio_axis=ratio_axis)
        elif output == "s_hat":
            fig, ax = _plot_search_s_hat(out_full, label, figsize, grid=grid, neg_color=neg_color,
                                         vmax_cap=vmax_cap, ratio_axis=ratio_axis)
        elif output == "Q_hat":
            fig, ax = _plot_search_Q(out_full, label, figsize, grid=grid, vmax_cap=vmax_cap)
        else:
            raise ValueError('output must be "cost", "cost_norm", "cost_rel", "s_hat" or "Q_hat"')
        _overlay_constraint(ax, out_full, out_c, grid, ratio_axis=ratio_axis)
        _overlay_title(ax, out_full, out_c, grid, output, ratio_axis=ratio_axis)
        fig.tight_layout()
        fig.canvas.draw_idle()                           # widget backend: redraw after overlay
    if which == "full":                                   # Q runs: "global" title with Q_best
        out_full = _load_search(pkl_path)[0]
        if "Q_hat" in out_full and np.isfinite(np.asarray(out_full["Q_hat"], float)).any():
            _q_title(ax, out_full, grid, output, "global")
    if group is not None:                                 # group number in the title too
        ax.title.set_text(f"Group {int(group)}: {ax.get_title()}")
    suffix = "_ratio" if (ratio_axis and grid == "cnct") else ""
    _save_plot(fig, pkl_path, f"{prefix}{which}_{output}{suffix}", dpi=dpi)
    return fig, ax


def plot_waveform_picks_scat(run_dir, scat_data, xlim=None, figsize=(9, 9), group=None, dpi=400):
    """
    Purpose
    -------
    Counterpart of plot_waveform_picks for the reflection/transmission
    search (DAS_modeling.cnct_search_RfromT): for every pick of a
    pick_ratiosearch session, rebuild the modeled reflection of every fitted
    pair exactly as the search did and plot it over the observed reflected
    trace, 10 pairs per 5x2-panel PNG.
    Model of pick k at pair i: h = RfromT(scat_data, i, ..., cn_k, ct_k, eps)
    (reflection predicted from the observed transmission), aligned to the
    observed reflection with DAS_inversion.align_by_window_cc (the run's cc
    and max_lag, inside scat_data['time_window']). No amplitude scalar: the
    search cost is 0.5 w (d - h)^2 without s_hat.

    Input
    -----
    run_dir : str or Path
        Run folder saved by cnct_search_scat() (or submit_detached) and
        picked with pick_ratiosearch: needs <run>.pkl (search result),
        <run>_data.pkl (medium, search_param) and *_picks.json.
    scat_data : dict
        Prepared EXACTLY as for the run (DASscatInv.ipynb: fft_traces_scat,
        geometric_factor -> Gr, Gt, omega, and the same low-pass of the
        transmitted traces), otherwise the rebuilt models differ from the
        ones the search fitted.
    xlim : (float, float) or None
        Time window of every panel; None -> scat_data['time_window'] padded
        by 30% on each side.
    figsize : tuple
        Size of each 5x2 figure.
    group : None or int (0, 1, 2, ...)
        None (default): the run itself (<run>.pkl, <run>_picks.json).
        g: a group of the run (DAS_modeling.extracting_search): reads
        group<g>_<run>.pkl (the group's pairs) and group<g>_<run>_picks.json
        (pick_ratiosearch on the group file), writes to group<g>_waveform_plots/.
    dpi : resolution of the saved PNGs (default 400).

    Output
    ------
    n_saved : int
        Number of PNGs written to run_dir/[group<g>_]waveform_plots/<run>_w<start>.png
        (the folder is emptied first, so it only holds this call's plots)
        (start = 0, 10, 20, ...; the last one may hold fewer pairs). One
        shared legend: run_dir/[group<g>_]waveform_plots/<run>_legend.png, one line
        per pick "cn=..., ct=..., cn/ct=..." (" (best_estimate)" for the
        auto-added best-fit pick).
    """
    # RfromT and align_by_window_cc come from DAS_model_inversion (DASmi, imported at the top)
    import shutil                                         # empties the output folder

    run_dir = Path(run_dir)
    name = run_dir.name
    prefix = "" if group is None else f"group{int(group)}_"   # group files: group<g>_<run>*
    with open(run_dir / f"{name}_data.pkl", "rb") as f:   # cnct_search_scat's run parameters
        run_input = pickle.load(f)
    picks_file = run_dir / f"{prefix}{name}_picks.json"
    if not picks_file.exists():
        raise FileNotFoundError(f"{picks_file} not found -- run pick_ratiosearch on "
                                f"{prefix}{name}.pkl first")
    with open(picks_file) as f:
        picks = json.load(f)
    if not picks:
        raise ValueError(f"{picks_file} has no picks")
    with open(run_dir / f"{prefix}{name}.pkl", "rb") as f:   # run or group result (its pairs)
        res = pickle.load(f)
    out = res[0] if isinstance(res, tuple) else res

    # run parameters: (search_mode, grid, eps, space, nsearch, cc[, constraint[, trace_opts[, max_lag]]])
    vp, vs, rho, wave = run_input["medium"]
    sp = run_input["search_param"]
    eps, cc_mode = float(sp[2]), sp[5]
    max_lag = sp[8] if len(sp) >= 9 else None
    idx_array = np.asarray(out["idx"], int).ravel()      # pairs actually fitted (after selection)

    rr = scat_data["reflection_traces"]
    taxis = np.asarray(rr.taxis, float)
    tw = scat_data["time_window"]
    if xlim is None:
        pad = 0.3 * (tw[1] - tw[0])
        xlim = (tw[0] - pad, tw[1] + pad)

    # actual compliances of each pick (old picks files: cn = cn_ratio * ct)
    pick_cnct = [(float(p.get("cn", p["cn_ratio"] * p["ct"])), float(p["ct"])) for p in picks]
    pick_styles = ["--" if p["source"] == "best_fit" else "-" for p in picks]
    pick_labels = [f"cn={cn:.3g}, ct={ct:.3g}, cn/ct={cn / ct:.3g}"
                   + (" (best_estimate)" if p["source"] == "best_fit" else "")
                   for (cn, ct), p in zip(pick_cnct, picks)]
    pick_colors, pick_widths = _pick_colors_widths(picks)

    def _models(i):
        """modeled reflection of every pick at pair i, aligned as in the search"""
        d = np.asarray(rr.data[i], float)
        rows = []
        for cn, ct in pick_cnct:
            h = DASmi.RfromT(scat_data, i, vp, vs, rho, cn, ct, eps, wave)["Er_time"]
            rows.append(DASmi.align_by_window_cc(np.asarray(h, float), d, taxis=taxis,
                                                  time_window=tw, cc=cc_mode, max_lag=max_lag)[0])
        return np.array(rows)

    out_dir = run_dir / f"{prefix}waveform_plots"
    if out_dir.exists():                                  # start empty: no PNGs left from older picks
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # shared legend (same picks in every panel)
    fig_l, ax_l = plt.subplots(figsize=(5, 0.4 * (len(picks) + 1)))
    ax_l.plot([], [], "k", lw=1.3, label="data (reflection)")
    for style, label, color, width in zip(pick_styles, pick_labels, pick_colors, pick_widths):
        ax_l.plot([], [], style, lw=width, color=color, label=label)
    ax_l.axis("off")
    leg = ax_l.legend(loc="center", fontsize=12, frameon=False)
    fig_l.canvas.draw()                                   # crop the PNG to the legend itself
    bbox = leg.get_window_extent(fig_l.canvas.get_renderer()).transformed(fig_l.dpi_scale_trans.inverted())
    fig_l.savefig(out_dir / f"{name}_legend.png", dpi=dpi, bbox_inches=bbox.padded(0.02))
    plt.close(fig_l)

    theta = np.asarray(scat_data["theta_i_deg"], float)
    n_saved = 0
    for start in range(0, idx_array.size, 10):
        items = list(range(start, min(start + 10, idx_array.size)))
        fig, axes = plt.subplots(5, 2, figsize=figsize, sharex=True)
        for slot in range(10):                            # left column 0-4, right column 5-9
            ax = axes[slot % 5, slot // 5]
            if slot >= len(items):
                ax.axis("off")                            # last file may hold fewer pairs
                continue
            item = items[slot]
            i = int(idx_array[item])
            H = _models(i)
            ax.plot(taxis, rr.data[i], "k", lw=1.3)
            for m, (style, color, width) in enumerate(zip(pick_styles, pick_colors, pick_widths)):
                ax.plot(taxis, H[m], style, lw=width, color=color)
            ax.set_xlim(*xlim)
            ax.set_title(f"item={item}, pair={i}, angle={theta[i]:.1f} deg, z_r={rr.daxis[i]:.1f} m",
                         fontsize=8)
            if slot < 5:
                ax.set_ylabel("amplitude")
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        fig.suptitle(f"{prefix}{name}: items {items[0]}-{items[-1]}, data vs picked RfromT (no s_hat)",
                     y=1.01, fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_w{start}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        n_saved += 1
    return n_saved


def _pick_colors_widths(picks, lw=0.9, lw_best=1.8):
    """
    Assign one distinct color + line width per pick for plot_waveform_picks.

    Purpose
    -------
    The best-fit pick (source == "best_fit") is drawn in red and thicker;
    every other pick gets its own unique non-red color, so no two curves
    share a color (and none collides with the black data trace or the red
    best fit).

    Input
    -----
    picks : list of dict
        pick_ratiosearch's *_picks.json entries (needs the "source" key).
    lw, lw_best : float
        Line width for regular picks / for the best-fit pick.

    Output
    ------
    colors : list
        One matplotlib color per pick (same order as `picks`).
    widths : list of float
        One line width per pick (same order as `picks`).
    """
    is_best = [p["source"] == "best_fit" for p in picks]
    n_other = len(picks) - sum(is_best)

    # non-red palette: tab10 without red (C3); if more picks than that,
    # tab20 without its two reds (6, 7); beyond that, sample viridis
    # (no red hues) so colors stay unique for any number of picks
    tab10 = [c for k, c in enumerate(plt.cm.tab10.colors) if k != 3]
    tab20 = [c for k, c in enumerate(plt.cm.tab20.colors) if k not in (6, 7)]
    if n_other <= len(tab10):
        palette = tab10
    elif n_other <= len(tab20):
        palette = tab20
    else:
        palette = list(plt.cm.viridis(np.linspace(0, 1, n_other)))

    others = iter(palette)
    colors = ["red" if b else next(others) for b in is_best]   # red reserved for best fit
    widths = [lw_best if b else lw for b in is_best]            # best fit drawn thicker
    return colors, widths


def plot_waveform_picks(run_dir, ref_data, inc_data, xlim=None, figsize=(9, 9)):
    """
    Rebuild and plot the fitted synthetic waveform (data vs R_synth_Gd_cc)
    at every pick from a pick_ratiosearch session, for every trace in the
    run, saved as 5x2-panel grid PNGs (10 traces per file).

    Generalizes DASrefInv.ipynb's save_R_synth_grid/export_R_synth_grids
    cells (a hardcoded cnct_list of (cn, ct) candidates) to instead use a
    run directory's own saved picks (cn_ratio, ct, s_hat from
    pick_ratiosearch's *_picks.json).

    Everything except ref_data/inc_data is inferred from `run_dir`:
      - *_data.pkl (cnct_search()'s own naming) or *_input.pkl (glob,
        fallback for a folder manually renamed to that) -- idx_array,
        medium, search_param (for eps)
      - *_picks.json (glob) -- the (cn_ratio, ct, s_hat) picks to overlay
      - *.pkl EXCLUDING *_data.pkl/*_input.pkl -- the (out_full,
        out_constraint) search result; out_full is used for the
        lag_samples/s_hat grid lookup (guaranteed non-NaN everywhere, and
        lag_samples is identical in both packs -- see
        cnct_ratiosearch_Rtime_obs's docstring)

    For each pick, the fitted synthetic at each trace is rebuilt with
    s_hat_at_ratio_ct_list (global s_hat/cn/ct at the nearest grid node),
    lag_at_list (that trace's own cc-realignment lag), and R_synth_Gd_cc
    (the aligned waveform itself) -- see those functions' docstrings in
    DASref_modeling.py.

    Picks with a Q estimate ('Q_hat' in the picks JSON: runs of
    cnct_searchR_obs with Q_opts, or spectrum runs with Q free): each
    pick's synthetic is DASrefmod.R_batch_InvQ with the pick's own
    cn = cn_ratio*ct, ct, s_hat and Q (Q = inf -> beta = 0; rd from the
    run), re-aligned to the trace with DASinv.align_by_window_cc (the
    run's cc and max_lag), exactly as the search built it (a = 1 model,
    aligned, times s_hat). The grid lag lookup is not used there, since a
    constrained pick's Q (and lag) can differ from the full pack's.

    Purpose
    -------
    Visually check, trace by trace, how well each picked (cn_ratio, ct)
    node's synthetic reflection matches the observed data.

    Input
    -----
    run_dir : str or Path
        A run folder as saved by cnct_search()/run_search_driver.py AND
        picked via pick_ratiosearch (e.g. ./inv_results/run1_cons) --
        needs `<run_dir>.pkl` (or any *.pkl there other than
        *_data.pkl/*_input.pkl), `*_data.pkl` (or `*_input.pkl`), and
        `*_picks.json` all present.
    ref_data : dict
        As built by the notebook's own data-loading cell.
    inc_data : (inc_wave, Gd)
        inc_wave : dict, RAW direct-arrival trace (dir_data['stf_trace']).
        Gd : array-like, direct geometric factor (e.g. gd['Gd']).
    xlim : (float, float) or None
        Time-axis window for every panel; defaults to
        ref_data['time_window'] padded by 30% on each side.
    figsize : tuple
        Passed to plt.subplots for each 5x2 grid figure.

    Output
    ------
    n_saved : int
        Number of grid PNG files written to
        run_dir/waveform_plots/<run_dir name>_w<start>.png, one per 10
        traces in idx_array (start = 0, 10, 20, ...). Grid panels carry no
        legend themselves -- one shared legend (since the pick set is the
        same across every panel/file) is written once to
        run_dir/waveform_plots/<run_dir name>_legend.png, labeled
        "cnct_r=..., ct=..., shat=...[, Q=...]" per pick, with " (best_estimate)"
        appended for the auto-added best-fit pick.
    """
    run_dir = Path(run_dir)
    inc_wave, Gd = inc_data

    # cnct_search()'s own naming is *_data.pkl; *_input.pkl is also
    # accepted since some run folders were manually renamed to that
    input_candidates = sorted(run_dir.glob("*_data.pkl")) or sorted(run_dir.glob("*_input.pkl"))
    if len(input_candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one *_data.pkl (or *_input.pkl) in {run_dir}, found {input_candidates}"
        )
    with open(input_candidates[0], "rb") as f:
        run_input = pickle.load(f)

    picks_candidates = sorted(run_dir.glob("*_picks.json"))
    if len(picks_candidates) != 1:
        raise FileNotFoundError(f"expected exactly one *_picks.json in {run_dir}, found {picks_candidates}")
    with open(picks_candidates[0]) as f:
        picks = json.load(f)
    if not picks:
        raise ValueError(f"{picks_candidates[0]} has no picks")

    pkl_candidates = [
        p for p in sorted(run_dir.glob("*.pkl"))
        if not (p.stem.endswith("_input") or p.stem.endswith("_data"))
    ]
    if len(pkl_candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one main *.pkl (excluding *_input.pkl/*_data.pkl) in {run_dir}, "
            f"found {pkl_candidates}"
        )
    with open(pkl_candidates[0], "rb") as f:
        res = pickle.load(f)
    out_full = res[0] if isinstance(res, tuple) else res   # full-mode runs: single dict
    gsR = out_full   # non-NaN everywhere; lag_samples identical in both packs

    # traces actually fitted: the result's own 'idx' (a trace_opts selection
    # may have dropped some of the requested run_input["idx_array"])
    idx_array = np.asarray(gsR.get("idx", run_input["idx_array"]), int).ravel()
    vp, vs, rho, wave = run_input["medium"]
    # eps position in search_param: cnct_searchR_obs starts with
    # (search_mode, grid, eps, ...); the older search functions with eps
    sp = run_input["search_param"]
    search_fn = run_input.get("search_fn")
    eps = sp[2] if search_fn in ("cnct_searchR_obs", "cnct_search_Rspectrum") else sp[0]

    # picks with Q: R_batch_InvQ + re-alignment (cc / max_lag of the run)
    use_Q = any("Q_hat" in p for p in picks)
    if use_Q:
        rd = out_full.get("rd")
        if rd is None:
            rd = (out_full.get("freq_opts") or {}).get("rd")
        if rd is None:
            raise ValueError("picks have Q_hat but the run has no direct path 'rd'")
        cc_mode = sp[5] if search_fn == "cnct_searchR_obs" else "signed"
        max_lag = sp[8] if (search_fn == "cnct_searchR_obs" and len(sp) >= 9) else None
        taxis_r = np.asarray(ref_data["reflection_traces"].taxis, float)

    def _synth_Q(idx):
        """Synthetic of every pick at trace idx (R_batch_InvQ, aligned, x s_hat)."""
        d_tr = np.asarray(ref_data["reflection_traces"].data[idx], float)
        rows = []
        for p in picks:
            Qp = float(p.get("Q_hat", np.inf))
            beta = 0.0 if not np.isfinite(Qp) else 1.0 / Qp
            h = DASmi.R_batch_InvQ(ref_data, inc_wave, [idx], 1.0, vp, vs, rho,
                                       p["cn_ratio"] * p["ct"], p["ct"], Gd=Gd, eps=eps,
                                       beta=beta, rd=rd, wave=wave)["Er_time"][0]
            h_cc = DASmi.align_by_window_cc(
                h, d_tr, taxis=taxis_r, time_window=ref_data["time_window"],
                cc=cc_mode, max_lag=max_lag)[0]
            rows.append(float(p["s_hat"]) * h_cc)
        return {"time": taxis_r, "Er_time": np.array(rows)}

    if xlim is None:
        tbegin, tend = ref_data["time_window"]
        pad = 0.3 * (tend - tbegin)
        xlim = (tbegin - pad, tend + pad)

    ratio_ct_list = [(float(p["cn_ratio"]), float(p["ct"])) for p in picks]
    s_list = None if use_Q else DASmi.s_hat_at_ratio_ct_list(gsR, ratio_ct_list)

    pick_styles = ["--" if p["source"] == "best_fit" else "-" for p in picks]
    pick_labels = [
        f"cnct_r={p['cn_ratio']:.2g}, ct={p['ct']:.2g}, shat={p['s_hat']:.2g}"
        + (f", Q={p['Q_hat']:.3g}" if "Q_hat" in p else "")
        + (" (best_estimate)" if p["source"] == "best_fit" else "")
        for p in picks
    ]
    # best fit -> red + thicker; every other pick -> its own non-red color
    pick_colors, pick_widths = _pick_colors_widths(picks)

    rr =ref_data["reflection_traces"]
    out_dir = run_dir / "waveform_plots"
    out_dir.mkdir(exist_ok=True)
    out_prefix = f"{run_dir.name}_w"

    # one shared legend, as its own figure -- the pick set (colors/styles/
    # labels) is identical across every grid file, so it's drawn once here
    # rather than cluttering a panel in every one of them
    fig_legend, ax_legend = plt.subplots(figsize=(5, 0.4 * (len(picks) + 1)))
    ax_legend.plot([], [], "k", lw=1.3, label="data")
    for style, label, color, width in zip(pick_styles, pick_labels, pick_colors, pick_widths):
        ax_legend.plot([], [], style, lw=width, color=color, label=label)
    ax_legend.axis("off")
    ax_legend.legend(loc="center", fontsize=8, frameon=False)
    fig_legend.tight_layout()
    fig_legend.savefig(out_dir / f"{run_dir.name}_legend.png", dpi=140, bbox_inches="tight")
    plt.close(fig_legend)

    n_traces = idx_array.size
    n_saved = 0
    for start in range(0, n_traces - (n_traces % 10 or 10) + 1, 10):
        item_ids = list(range(start, min(start + 10, n_traces)))
        if len(item_ids) < 10:
            break   # partial trailing group -- matches export_R_synth_grids's own range() cutoff
        left_ids, right_ids = item_ids[:5], item_ids[5:]

        fig, axes = plt.subplots(5, 2, figsize=figsize, sharex=True, sharey=False)
        for row in range(5):
            for col, ids in enumerate((left_ids, right_ids)):
                item = ids[row]
                idx = int(idx_array[item])
                ax = axes[row, col]

                if use_Q:                                # Q picks: R_batch_InvQ + re-alignment
                    synth = _synth_Q(idx)
                else:
                    lag_arr = DASmi.lag_at_list(gsR, s_list["i"], s_list["j"], idx)
                    synth = DASmi.R_synth_Gd_cc(
                        s_list, lag_arr, ref_data, inc_wave, idx,
                        vp, vs, rho, Gd=Gd, eps=eps, a=1.0, wave=wave,
                    )

                ax.plot(rr.taxis, rr.data[idx], "k", lw=1.3)
                # same color/width per pick as the shared legend
                for m, (style, color, width) in enumerate(zip(pick_styles, pick_colors, pick_widths)):
                    ax.plot(synth["time"], synth["Er_time"][m], style, lw=width, color=color)

                ax.set_xlim(*xlim)
                ax.set_title(
                    f"item={item}, idx={idx}, angle={ref_data['theta_i_deg'][idx]:.1f} deg",
                    fontsize=8,
                )
                if col == 0:
                    ax.set_ylabel("amplitude")

        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("time [s]")
        fig.suptitle(
            f"{run_dir.name}: items {left_ids[0]}-{right_ids[-1]}, data vs picked "
            + ("R_batch_InvQ (with Q)" if use_Q else "R_synth_Gd_cc"),
            y=1.01, fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{out_prefix}{start}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        n_saved += 1

    return n_saved


def plot_R_synth(syn, R_batch, figsize=(8, 3.5)):
    """
    Overlay R_synth traces on the R_wave_batch 'data' for the same idx.

    syn     : output of R_synth (Er_time (n_m, nt), cn, ct, s_hat, idx, angle, time)
    R_batch : output of R_wave_batch (Er_time, idx, angle, time)
    """

    idx = int(syn["idx"])
    batch_idx = np.asarray(R_batch["idx"], int).ravel()
    hit = np.flatnonzero(batch_idx == idx)
    if hit.size == 0:
        raise ValueError(f"idx={idx} not in R_batch['idx']")
    k = int(hit[0])

    t_syn = np.asarray(syn["time"], float)
    t_dat = np.asarray(R_batch["time"], float)
    d = np.asarray(R_batch["Er_time"][k], float)
    H = np.asarray(syn["Er_time"], float)   # (n_m, nt)
    cn = np.asarray(syn["cn"], float)
    ct = np.asarray(syn["ct"], float)
    s_hat = np.asarray(syn["s_hat"], float)

    angle = float(syn.get("angle", R_batch["angle"][k]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t_dat, d, color="k", lw=1.6, label="data")

    for m in range(H.shape[0]):
        lab = (
            rf"$c_n$={cn[m]:.2e}, $c_t$={ct[m]:.2e}, "
            rf"$\hat{{s}}$={s_hat[m]:.3g}"
        )
        ax.plot(t_syn, H[m], lw=1.1, label=lab)

    ax.set_xlabel("time [s]")
    ax.set_ylabel("amplitude")
    ax.set_title(rf"angle = {angle:.1f}$^\circ$")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig, ax

def _plot_search_Q(gsR, which, figsize, grid="ratio", inf_color="lightgray", vmax_cap=None,
                   linear=None):
    """
    Purpose: Q_hat density plot of one cnct_search_Rspectrum pack (axes as
        in _plot_search_cost). Q_hat = 1/beta_hat; cells with beta = 0
        (Q = inf, no attenuation fitted) are drawn in inf_color; cells
        excluded by the s_hat constraint (NaN) are blank.
    Input: gsR (one pack, needs 'Q_hat'); which ("full"/"constraint", title);
        figsize; grid ("ratio"/"cnct"); inf_color; vmax_cap (colorbar
        upper-bound cap on finite Q); linear (True/False colorbar scale;
        None = linear when Q_max/Q_min <= 20, e.g. a Q list of 20-150, log
        otherwise).
    Output: (fig, ax).
    """
    if "Q_hat" not in gsR:
        raise KeyError("pack has no 'Q_hat' (not a cnct_search_Rspectrum run)")
    g = _grid_axes(gsR, grid)
    Q = np.asarray(gsR["Q_hat"], float)                  # (n_y, n_ct), inf where beta = 0
    is_inf = np.isinf(Q)
    fin = Q[np.isfinite(Q) & (Q > 0)]

    fig, ax = plt.subplots(figsize=figsize)
    if is_inf.any():                                     # layer 1: Q = inf cells
        _mesh(ax, g["x"], g["y"], np.ma.masked_where(~is_inf, np.zeros(Q.shape)),
              cmap=ListedColormap([inf_color]), vmin=0, vmax=1)
        ax.plot([], [], ls="none", marker="s", color=inf_color, label=r"$Q=\infty$ ($\beta=0$)")
    pcm = None
    if fin.size:                                         # layer 2: finite Q, log scale
        vmin, vmax = float(fin.min()), float(fin.max())
        if vmax_cap is not None:
            vmax = min(vmax, float(vmax_cap))
        if vmin >= vmax:
            vmax = vmin * 1.01
        lin = (vmax / vmin <= 20) if linear is None else bool(linear)   # narrow range: linear
        pcm = _mesh(ax, g["x"], g["y"], np.where(np.isfinite(Q), Q, np.nan),
                    cmap="turbo",
                    norm=Normalize(vmin=vmin, vmax=vmax) if lin else LogNorm(vmin=vmin, vmax=vmax))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$c_t$ [m/Pa]")
    ax.set_ylabel(g["ylabel"])

    # title: "Quality factor Q (which)" / c_t, c_n/c_t, Q_best at the C_f minimum
    title = rf"Quality factor $\hat{{Q}}$ ({which})"
    Csp = np.asarray(gsR["Csp_sum"], float)
    if g["best_txt"] is not None and np.isfinite(Csp).any():
        i, j = np.unravel_index(np.nanargmin(Csp), Csp.shape)
        title += f"\n{g['best_txt']}" + rf", $\hat{{Q}}_{{\mathrm{{best}}}}$={Q[i, j]:.3g}"
        ax.plot(g["x_best"], g["y_best"], "w*", ms=12, markeredgecolor="k",
                markeredgewidth=0.6,
                label=rf"$C_{'f' if gsR.get('domain') == 'frequency' else 't'}$ min")
    else:
        title += "\n(no grid node satisfies the constraint)"
    ax.set_title(title, fontsize=11)
    if pcm is not None:
        cb = fig.colorbar(pcm, ax=ax, pad=0.02)
        cb.set_label(r"$\hat{Q}$" if lin else r"$\hat{Q}$ (log)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    plt.show()
    return fig, ax


def _q_title(ax, pack, grid, output, which_txt):
    """
    Purpose: title of a Q-run plot (Q estimated per node):
        "<map name> (<which_txt>)"
        "c_t=..., c_n/c_t=..., s_best=..., Q_best=..."  (best node of `pack`)
    Input: ax; pack (full or constrained pack with 'Q_hat'); grid
        ("ratio"/"cnct"); output (map plotted); which_txt ("global" or
        "constraint s 0.2-3").
    Output: None (sets the title).
    """
    dsub = "f" if pack.get("domain") == "frequency" else "t"
    head = {"cost": rf"Cost function $C_{dsub}$",
            "cost_norm": rf"Normalized cost function $C_{dsub}$",
            "cost_rel": rf"Relative cost function $C_{dsub}/C_{{{dsub},\mathrm{{min}}}}$",
            "s_hat": rf"Scale $\hat s_{dsub}$",
            "Q_hat": r"Quality factor $\hat Q$"}[output]
    g = _grid_axes(pack, grid)
    if g["best_txt"] is None:
        ax.set_title(f"{head} ({which_txt})\n(no grid node satisfies the constraint)", fontsize=11)
        return
    C = np.asarray(pack["Csp_sum"], float)
    i, j = np.unravel_index(np.nanargmin(C), C.shape)
    ax.set_title(f"{head} ({which_txt})\n{g['best_txt']}, "
                 rf"$\hat s_{{{dsub},\mathrm{{best}}}}$={pack['s_hat'][i, j]:.3g}, "
                 rf"$\hat Q_{{\mathrm{{best}}}}$={pack['Q_hat'][i, j]:.3g}", fontsize=11)


def _is_qgrid_constraint(out_full, out_c):
    """
    Purpose: True when the constrained pack has its OWN cost map: a Q-grid
        search (cnct_searchR_obs with Q_opts, 's_hat_Q' in out_full)
        constrained in the search, where each node uses the lowest-cost Q
        with s_hat inside the range (not the full pack's Q). False for runs
        without Q and for constraints added afterwards on the full maps
        (Add_constrain_cntct_output: 'constraint_type' set).
    Input: out_full, out_c (packs). Output: bool.
    """
    return "s_hat_Q" in out_full and out_c.get("constraint_type") is None


def _plot_constrained_own(out_full, out_c, grid, output, label, figsize, rel_max=None,
                          neg_color="black", vmax_cap=None, ratio_axis=False):
    """
    Purpose: plot a Q-grid constrained pack's OWN maps (cost / cost_norm /
        cost_rel / s_hat / Q_hat at the Q chosen under the s constraint);
        every node where at least one Q gives s_hat in range is plotted;
        nodes where no Q does are blank. White x = constrained best, small star = global
        best (full pack, for reference); title adds the global best.
    Input: out_full, out_c (packs); grid ("ratio"/"cnct"); output; label
        (title label); figsize; rel_max, neg_color, vmax_cap, ratio_axis as
        plot_cnct_search.
    Output: (fig, ax).
    """
    if output in ("cost", "cost_norm", "cost_rel"):
        fig, ax = _plot_search_cost(out_c, label, figsize, grid=grid, kind=output,
                                    rel_max=rel_max, ratio_axis=ratio_axis)
    elif output == "s_hat":
        fig, ax = _plot_search_s_hat(out_c, label, figsize, grid=grid, neg_color=neg_color,
                                     vmax_cap=vmax_cap, ratio_axis=ratio_axis, linear=True)
    elif output == "Q_hat":
        fig, ax = _plot_search_Q(out_c, label, figsize, grid=grid, vmax_cap=vmax_cap, linear=True)
    else:
        raise ValueError('output must be "cost", "cost_norm", "cost_rel", "s_hat" or "Q_hat"')
    for ln in list(ax.lines):                            # drop the base plot's best-star
        if ln.get_marker() == "*":
            ln.remove()
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    g = _grid_axes(out_full, grid, ratio_axis)
    gc = _grid_axes(out_c, grid, ratio_axis)
    X, Y = np.meshgrid(g["x"], g["y"]) if np.ndim(g["x"]) == 1 else (g["x"], g["y"])
    mask = np.asarray(out_c["mask"], bool)
    handles = []
    if output == "Q_hat" and np.isinf(np.asarray(out_c["Q_hat"], float)).any():
        handles.append(plt.Line2D([], [], ls="none", marker="s", color="lightgray",
                                  label=r"$Q=\infty$ ($\beta=0$)"))
    if gc["x_best"] is not None:
        ax.plot(gc["x_best"], gc["y_best"], "wx", ms=7, mew=1.5, zorder=4)
        handles.append(plt.Line2D([], [], color="w", marker="x", ls="none", ms=6, mew=1.5,
                                  label="best constrained"))
    if g["x_best"] is not None:
        ax.plot(g["x_best"], g["y_best"], "*", color="0.85", ms=8, mec="k", mew=0.5, zorder=4)
        handles.append(plt.Line2D([], [], color="0.85", marker="*", ls="none", ms=8, mec="k",
                                  mew=0.5, label="best global"))
    s_r = out_c.get("s_hat_range")
    _q_title(ax, out_c, grid, output,
             "constraint" + ("" if s_r is None else f" s {s_r[0]:g}-{s_r[1]:g}"))
    # legend corner: fewest kept nodes (blank area preferred), never over the star or the x
    lx, ly = np.log10(ax.get_xlim()), np.log10(ax.get_ylim())
    fx, fy = (np.log10(X) - lx[0]) / (lx[1] - lx[0]), (np.log10(Y) - ly[0]) / (ly[1] - ly[0])
    marks = [((np.log10(x0) - lx[0]) / (lx[1] - lx[0]), (np.log10(y0) - ly[0]) / (ly[1] - ly[0]))
             for x0, y0 in ((g["x_best"], g["y_best"]), (gc["x_best"], gc["y_best"])) if x0 is not None]
    region = {"upper right": lambda u, v: (u > 0.5) & (v > 0.55), "upper left": lambda u, v: (u < 0.5) & (v > 0.55),
              "lower right": lambda u, v: (u > 0.5) & (v < 0.45), "lower left": lambda u, v: (u < 0.5) & (v < 0.45)}
    loc = min(region, key=lambda k: np.count_nonzero(mask & region[k](fx, fy))
              + 10 ** 6 * sum(bool(region[k](u, v)) for u, v in marks))
    leg = ax.legend(handles=handles, loc=loc, fontsize=8, facecolor="0.45", framealpha=0.8)
    for t in leg.get_texts():
        t.set_color("w")
    fig.tight_layout()
    fig.canvas.draw_idle()
    return fig, ax


def _overlay_title(ax, out_full, out_c, grid, output, ratio_axis=False):
    """
    Purpose: short 3-line title of a constraint-overlay plot (full map +
        constraint drawn on it):
            "<C_t | C_t,norm | C_t/C_t,min | s_t | Q>"
            "global best: c_t=..., c_n/c_t=..., s=...[, Q=...]"
            "constrained <s range>[, <Q range>] best: c_t=..., c_n/c_t=..., s=...[, Q=...]"
        (C_f / s_f for frequency-domain runs; Q only when the run has a finite Q).
    Input: ax; out_full, out_c (packs); grid ("ratio"/"cnct"); output (map
        plotted); ratio_axis (grid="cnct" remap).
    Output: None (sets the title).
    """
    d = "f" if out_full.get("domain") == "frequency" else "t"
    head = {"cost": rf"$C_{d}$", "cost_norm": rf"$C_{{{d},\mathrm{{norm}}}}$",
            "cost_rel": rf"$C_{d}/C_{{{d},\mathrm{{min}}}}$", "s_hat": rf"$\hat s_{d}$",
            "Q_hat": r"$\hat Q$"}[output]

    def best(pack):
        g = _grid_axes(pack, grid, ratio_axis)
        C = np.asarray(pack["Csp_sum"], float)
        if g["best_txt"] is None or not np.isfinite(C).any():
            return "no node inside the range"
        i, j = np.unravel_index(np.nanargmin(C), C.shape)
        txt = f"{g['best_txt']}, " + rf"$\hat s$={pack['s_hat'][i, j]:.3g}"
        Q = np.asarray(pack.get("Q_hat", np.full(C.shape, np.inf)), float)
        if np.isfinite(Q).any():
            txt += rf", $\hat Q$={Q[i, j]:.3g}"
        return txt

    rng = [f"{k} {r[0]:g}-{r[1]:g}" for k, r in (("s", out_c.get("s_hat_range")),
                                                 ("Q", out_c.get("Q_range"))) if r is not None]
    ax.set_title(f"{head}\nglobal best: {best(out_full)}\n"
                 f"constrained {', '.join(rng)} best: {best(out_c)}", fontsize=10)


def _overlay_constraint(ax, out_full, out_c, grid, ratio_axis=False):
    """
    Purpose: draw a constrained pack's limits and best node on a map of the
        FULL pack: solid white contours at the s_hat_range bounds (full
        s_hat map), dashed white contours at the Q_range bounds (full Q_hat
        map, Q = inf treated as above any bound), tiny white dots on the
        nodes inside both ranges (only when s AND Q ranges are applied),
        small white x at the constrained best node; the global-best star of
        the base plot is shrunk to match. Legend on a gray background (white
        marks stay visible), in the axes corner holding the fewest
        constrained nodes (away from the constrained area).
    Input: ax (axes with the full map); out_full, out_c (packs); grid
        ("ratio"/"cnct"); ratio_axis (grid="cnct" only: map drawn with
        cn/ct on the vertical axis, as _grid_axes). Time- or frequency-
        domain packs (C_t / C_f from out_full['domain']).
        Not used for Q-grid constraints (own cost map: _plot_constrained_own).
    Output: title line for the constrained best ("" when it is the global best).
    """
    g = _grid_axes(out_full, grid, ratio_axis)
    if np.ndim(g["x"]) == 1:
        X, Y = np.meshgrid(g["x"], g["y"])                # node centers, (n_y, n_ct)
    else:
        X, Y = g["x"], g["y"]                             # ratio_axis remap: already 2-D
    dsub = "f" if out_full.get("domain") == "frequency" else "t"
    handles = []
    s_r, Q_r = out_c.get("s_hat_range"), out_c.get("Q_range")
    mask = np.asarray(out_c.get("mask", np.zeros(X.shape, bool)), bool)
    if s_r is not None:                                   # log10 contour: smooth on log axes
        S = np.asarray(out_full["s_hat"], float)
        lS = np.log10(np.where(np.isfinite(S) & (S > 0), S, np.nan))
        ax.contour(X, Y, lS, levels=sorted(np.log10(s_r)), colors="w", linewidths=1.3,
                   linestyles="solid")
        handles.append(plt.Line2D([], [], color="w", ls="-", label=f"s {s_r[0]:g}-{s_r[1]:g}"))
    if Q_r is not None:
        Q = np.asarray(out_full["Q_hat"], float)
        lQ = np.log10(np.where(np.isinf(Q), 1e12, np.where(Q > 0, Q, np.nan)))   # inf -> above any bound
        ax.contour(X, Y, lQ, levels=sorted(np.log10(Q_r)), colors="w", linewidths=1.3,
                   linestyles="dashed")
        handles.append(plt.Line2D([], [], color="w", ls="--", label=f"Q {Q_r[0]:g}-{Q_r[1]:g}"))

    if s_r is not None and Q_r is not None and mask.any():   # s AND Q overlap: tiny dots
        ax.scatter(X[mask], Y[mask], s=3, c="w", marker=".", linewidths=0, zorder=3)
        handles.append(plt.Line2D([], [], color="w", marker=".", ms=4, ls="none", label="s and Q range"))

    for ln in ax.lines:                                   # shrink the base plot's global-best star
        if ln.get_marker() == "*":
            ln.set_markersize(8)

    gc = _grid_axes(out_c, grid, ratio_axis)
    txt = ""
    if gc["best_txt"] is None:
        txt = "constrained: no node inside the range"
    else:
        ax.plot(gc["x_best"], gc["y_best"], "wx", ms=7, mew=1.5, zorder=4)
        handles.append(plt.Line2D([], [], color="w", marker="x", ls="none", ms=6, mew=1.5,
                                  label="best constrained"))
        # relative test only (ct ~1e-10 is far below isclose's default atol = 1e-8)
        same = (np.isclose(gc["x_best"], g["x_best"], rtol=1e-6, atol=0)
                and np.isclose(gc["y_best"], g["y_best"], rtol=1e-6, atol=0))
        if same:
            txt = "constrained best = global best"
        else:                                             # values at the constrained best node
            C = np.asarray(out_c["Csp_sum"], float)
            i, j = np.unravel_index(np.nanargmin(C), C.shape)
            txt = (f"constrained (x): {gc['best_txt']}, $C_{dsub}$={C[i, j]:.3g}, "
                   rf"$\hat s$={out_c['s_hat'][i, j]:.3g}")
            if "Q_hat" in out_c:
                txt += rf", $\hat Q$={out_c['Q_hat'][i, j]:.3g}"
    # legend corner with the fewest constrained nodes (node positions in axes fractions)
    fx = (np.log10(X) - np.log10(ax.get_xlim()[0])) / np.diff(np.log10(ax.get_xlim()))[0]
    fy = (np.log10(Y) - np.log10(ax.get_ylim()[0])) / np.diff(np.log10(ax.get_ylim()))[0]
    corners = {"upper right": (fx > 0.55) & (fy > 0.6), "upper left": (fx < 0.45) & (fy > 0.6),
               "lower right": (fx > 0.55) & (fy < 0.4), "lower left": (fx < 0.45) & (fy < 0.4)}
    # avoid the corners holding the global-best star or the constrained-best x (heavy penalty)
    def _frac(x0, y0):
        return ((np.log10(x0) - np.log10(ax.get_xlim()[0])) / np.diff(np.log10(ax.get_xlim()))[0],
                (np.log10(y0) - np.log10(ax.get_ylim()[0])) / np.diff(np.log10(ax.get_ylim()))[0])
    marks = [_frac(g["x_best"], g["y_best"])] if g["x_best"] is not None else []
    if gc["x_best"] is not None:
        marks.append(_frac(gc["x_best"], gc["y_best"]))
    box = {"upper right": lambda u, v: u > 0.5 and v > 0.55, "upper left": lambda u, v: u < 0.5 and v > 0.55,
           "lower right": lambda u, v: u > 0.5 and v < 0.45, "lower left": lambda u, v: u < 0.5 and v < 0.45}
    loc = min(corners, key=lambda k: np.count_nonzero(mask & corners[k])
              + 10 ** 6 * sum(box[k](u, v) for u, v in marks))
    leg = ax.legend(handles=handles, loc=loc, fontsize=8, facecolor="0.45", framealpha=0.8)
    for t in leg.get_texts():
        t.set_color("w")
    return txt


def plot_wiggle_reflection_tt(das, tt_fit_fracture, trans_tt=None, scale=0.2, trace_step=5,
                              linewidth=0.5, figsize=(10, 6), xlim=None, ylim=None, save_path=None,
                              tt_ms=3.0, tt_alpha=0.8, label_size=14, legend_size=14,
                              trans_dmax=3600.0):
    """
    Purpose
    -------
    Wiggle plot of a DAS panel (as DAS_filter.plot_wiggle in DASrefData.ipynb)
    with the fitted SS traveltimes drawn as DOTS on the plotted traces only
    (channels das.daxis[::trace_step]), so each dot sits on its own trace:
      - reflection SS_r: traveltime of the best-fit fracture line
        (DASproc.estimate_best_line), Golden Tech #F1B91A (Mines palette);
      - transmission SS_t: quadratic fit of the S picks saved by
        DASscatData.ipynb in data/transmission_tt.pkl, neon green #39FF14.
    Colors are fixed; tt_ms / tt_alpha set both. Presentation style
    (tasks/modify_plot.txt): figure 10 x 6, 14 pt ticks / labels / legend.

    Input
    -----
    das : jin_pylib Data2D_XT panel (e.g. DAS_filter), with .data (nx, nt),
        .taxis [s] (panel time) and .daxis [m].
    tt_fit_fracture : dict, reflection traveltime of the fitted fracture:
        'tt_best_das' (n,) panel time [s] = best_line_out['tt'] + ts - tfile_begin
        'depth_best'  (n,) channel depths [m] = drange (= daxis[trace_range])
    trans_tt : dict, str/Path or None
        Transmission traveltime: the dict saved in data/transmission_tt.pkl
        ('daxis' [m], 'SStr_fit' [s] panel time, NaN outside the fit gate)
        or the path of that pkl. None -> reflection only.
    scale, trace_step, linewidth : wiggle options, as plot_wiggle (trace_step
        also selects the channels that get a dot).
    figsize : figure size [in], default (10, 6).
    xlim, ylim : (min, max) depth [m] / time [s] limits, or None (full panel).
        Time increases downwards (ylim given as (t_min, t_max)).
    save_path : str/Path or None, PNG to save the figure to.
    tt_ms, tt_alpha : dot size [pt] and opacity (0 = invisible, 1 = solid) of
        BOTH traveltimes.
    label_size : font size [pt] of the tick labels and axis labels.
    legend_size : font size [pt] of the legend.
    trans_dmax : float or None, SS_t dots only down to this depth [m]
        (default 3600); None -> the whole fit gate.

    Output
    ------
    (fig, ax)
    """
    d_all = np.asarray(das.daxis, float)
    d_plot = d_all[::trace_step]                              # depths of the plotted traces

    def on_plotted(depth, tt):
        """keep the (depth, tt) samples that fall on a plotted trace (finite tt only)"""
        depth, tt = np.asarray(depth, float), np.asarray(tt, float)
        if depth.shape != tt.shape:
            raise ValueError(f"depth ({depth.size}) and traveltime ({tt.size}) lengths differ")
        k = np.clip(np.searchsorted(d_plot, depth), 1, d_plot.size - 1)
        nearest = np.where(np.abs(d_plot[k - 1] - depth) <= np.abs(d_plot[k] - depth), k - 1, k)
        hit = np.isclose(d_plot[nearest], depth, rtol=0, atol=1e-3) & np.isfinite(tt)
        return depth[hit], tt[hit]

    missing = {"tt_best_das", "depth_best"} - set(tt_fit_fracture)
    if missing:
        raise KeyError(f"tt_fit_fracture needs keys {sorted(missing)}")
    d_r, t_r = on_plotted(tt_fit_fracture["depth_best"], tt_fit_fracture["tt_best_das"])

    if trans_tt is not None and not isinstance(trans_tt, dict):   # path of transmission_tt.pkl
        with open(trans_tt, "rb") as f:
            trans_tt = pickle.load(f)

    fig = plt.figure(figsize=figsize)
    das.plot_wiggle(scale=scale, trace_step=trace_step, linewidth=linewidth)   # also inverts the time axis
    ax = plt.gca()

    # fitted traveltimes as dots on the plotted traces, fixed colors
    ax.plot(d_r, t_r, "o", color="#F1B91A", ms=tt_ms, alpha=tt_alpha, mew=0,
            label=r"$SS_r$ fitted travel time")                       # reflection: Golden Tech
    if trans_tt is not None:
        d_t = np.asarray(trans_tt["daxis"], float)
        t_t = np.asarray(trans_tt["SStr_fit"], float).copy()
        if trans_dmax is not None:
            t_t[d_t > trans_dmax] = np.nan                    # SS_t only down to trans_dmax
        d_t, t_t = on_plotted(d_t, t_t)
        ax.plot(d_t, t_t, "o", color="#39FF14", ms=tt_ms, alpha=tt_alpha, mew=0,
                label=r"$SS_t$ fitted travel time")                   # transmission: neon green

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(max(ylim), min(ylim))                    # keep time increasing downwards
    ax.set_xlabel("depth [m]", fontsize=label_size)
    ax.set_ylabel("time [s]", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=label_size)
    ax.legend(loc="lower right", fontsize=legend_size, markerscale=2)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=130)
    return fig, ax


def plot_scattering_rays(scat_data, source, fracture_line, figsize=(7, 8), xlim=(0, 120),
                         ylim=None, label_size=14, save_path=None):
    """
    Purpose
    -------
    Geometry sketch of the reflection/transmission pairs (2-D: x = horizontal
    distance from the well [m], y = depth [m], increasing downwards): fiber
    along x = 0, source S, fracture line y = m x + b, and the rays of the
    FIRST (P1) and LAST (P2) ref-trans pair of scattering_data:
        S -> P             black (incident ray)
        P -> dr (fiber)    orange, Golden Tech #F1B91A (reflected ray)
        P -> dt (fiber)    neon green #39FF14 (transmitted ray)
    Fracture: P1 -> P2 black thick (the imaged part), P2 -> foot of the
    perpendicular from S grey thick, the rest of the line (fiber/b -> P1 and
    beyond) grey dashed. The unit normal n (grey, thin) is drawn at the
    incidence point P2 on the source side, with the incidence angle theta_i
    (arc) between n and the incident ray. The intercept b is marked on the
    fiber. Only S, P1, P2 (shallow side of the fracture, no dots), n and
    theta_i are written on the plot; the depths are the
    y-axis ticks, colored as the rays: dr1, dr2 orange, dt1, dt2 green,
    b black, source depth red (1 decimal).

    Input
    -----
    scat_data : dict or str/Path, data/scattering_data.pkl (DASscatData.ipynb):
        'ref_points' (n_pairs, 2) reflection points (x, y) [m],
        'reflection_traces'.daxis / 'transmitted_traces'.daxis (n_pairs,)
        receiver depths dr / dt [m] of each pair, optional 'normal' (2,).
    source : (xs, ys) source position [m].
    fracture_line : (m, b) fracture y = m x + b [m].
    figsize : figure size [in].
    xlim : (x_min, x_max) [m], default (0, 120).
    ylim : (y_min, y_max) depth range [m] or None (fit all points + margin);
        depth increases downwards.
    label_size : font size [pt] of ticks, axis labels and annotations.
    Axes are drawn to scale (equal aspect), so ray angles and the normal
    are true.
    save_path : str/Path or None, PNG to save the figure to.

    Output
    ------
    (fig, ax)
    """
    if not isinstance(scat_data, dict):                       # path of scattering_data.pkl
        with open(scat_data, "rb") as f:
            scat_data = pickle.load(f)
    xs, ys = map(float, source)
    m, b = map(float, fracture_line)
    P = np.asarray(scat_data["ref_points"], float)           # reflection points (x, y)
    dr = np.asarray(scat_data["reflection_traces"].daxis, float)
    dt = np.asarray(scat_data["transmitted_traces"].daxis, float)
    n = np.asarray(scat_data.get("normal", np.array([-m, 1.0]) / np.hypot(m, 1.0)), float)
    ends = (0, P.shape[0] - 1)                                 # first (P1) and last (P2) pair

    # foot of the perpendicular from S on the fracture (where the normal is drawn)
    xf = (xs + m * (ys - b)) / (1.0 + m * m)
    yf = m * xf + b
    x_hi = xlim[1] if xlim is not None else max(xs, xf) * 1.2
    x1, x2 = P[ends[0], 0], P[ends[1], 0]

    fig, ax = plt.subplots(figsize=figsize)
    # fracture: dashed grey outside, grey thick P2 -> normal, black thick P1 -> P2
    ax.plot([0, x1], [b, m * x1 + b], "--", color="grey", lw=1.5)
    ax.plot([xf, x_hi], [yf, m * x_hi + b], "--", color="grey", lw=1.5)
    ax.plot([x2, xf], [m * x2 + b, yf], "-", color="grey", lw=3)
    ax.plot([x1, x2], [m * x1 + b, m * x2 + b], "-", color="k", lw=3.5)
    # unit normal at the incidence point P2 (grey, thin), on the source side, and the
    # incidence angle theta_i between the normal and the ray back to S (arc + label)
    xn, yn = P[ends[1]]
    if np.dot(n, [xs - xn, ys - yn]) < 0:                     # normal toward the incident side
        n = -n
    L = 30.0                                                  # normal arrow length [m]
    ax.annotate("", xy=(xn + L * n[0], yn + L * n[1]), xytext=(xn, yn),
                arrowprops=dict(arrowstyle="-|>", color="grey", lw=0.75))
    ax.text(xn + 1.12 * L * n[0], yn + 1.12 * L * n[1], r"$\hat{n}$", fontsize=label_size,
            color="grey", ha="center", va="center")
    a_n = np.degrees(np.arctan2(n[1], n[0]))                  # normal direction [deg, data coords]
    a_s = np.degrees(np.arctan2(ys - yn, xs - xn))            # direction P2 -> S
    t1, t2 = (a_n, a_s) if (a_s - a_n) % 360 < 180 else (a_s, a_n)   # short arc between them
    r_arc = 0.4 * L
    ax.add_patch(Arc((xn, yn), 2 * r_arc, 2 * r_arc, theta1=t1, theta2=t2, color="k", lw=0.8))
    a_mid = np.radians(t1 + ((t2 - t1) % 360) / 2)            # label at the middle of the arc
    ax.text(xn + 1.35 * r_arc * np.cos(a_mid), yn + 1.35 * r_arc * np.sin(a_mid), r"$\theta_i$",
            fontsize=label_size, ha="center", va="center")

    # fiber, intercept b, source
    ax.axvline(0, color="blue", lw=2)
    ax.spines["left"].set_color("blue"); ax.spines["left"].set_linewidth(2)   # fiber = left axis when xlim starts at 0
    ax.plot(0, b, "o", color="k", ms=5, zorder=5, clip_on=False)
    ax.plot(xs, ys, "o", color="red", ms=10, zorder=6)
    ax.annotate("S", xy=(xs, ys), xytext=(8, 0), textcoords="offset points",
                fontsize=label_size, color="red", va="center")

    # rays of the first / last pair: S -> P (arrowhead at P), P -> dr, P -> dt (arrowheads at the fiber)
    for j, k in enumerate(ends, start=1):
        px, py = P[k]
        ax.annotate(f"$P_{j}$", xy=(px, py), xytext=(5, 6), textcoords="offset points",
                    fontsize=label_size, ha="left", va="bottom")   # label on the shallow side, no dot
        ax.annotate("", xy=(px, py), xytext=(xs, ys),         # incident ray, arrowhead at P
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.2))
        for d_rec, col in ((dr[k], "#F1B91A"), (dt[k], "#39FF14")):
            ax.annotate("", xy=(0, d_rec), xytext=(px, py),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))

    if ylim is None:                                           # all drawn points + 5% margin
        y_all = np.r_[dr[list(ends)], dt[list(ends)], ys, b, P[list(ends), 1]]
        pad = 0.05 * np.ptp(y_all)
        ylim = (y_all.min() - pad, y_all.max() + pad)
    ax.set_ylim(max(ylim), min(ylim))                         # depth increases downwards
    if xlim is not None:
        ax.set_xlim(xlim)                                     # fiber (x = 0) on the left axis
    ax.set_aspect("equal", adjustable="box")                  # true angles
    ax.set_xlabel("x [m]", fontsize=label_size)
    ax.set_ylabel("depth along fiber [m]", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=label_size)

    # y ticks = receiver depths and intercept, colored as the rays (darker shades for legibility)
    yt = [(dr[k], "#C8930A") for k in ends] + [(dt[k], "#1FA80A") for k in ends] + [(b, "k"), (ys, "red")]
    ax.set_yticks([v for v, _ in yt])
    ax.set_yticklabels([f"{v:.1f}" for v, _ in yt])
    for lab, (_, col) in zip(ax.get_yticklabels(), yt):
        lab.set_color(col)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=130)
    return fig, ax


def plot_pairs_table(scat_data, save_dir="./plots/DASplots", fname="pairs_table.png",
                     figsize=(4, 1.4), font_size=14, dpi=400):
    """
    Purpose
    -------
    Table figure (white background) of the transmission/reflection pairs of
    scattering_data, showing only the FIRST and LAST pair with a row of
    vertical dots in between (more pairs not shown):
        Index | T depth (m) | R depth (m) | theta_i (deg)
    Depths and angle with 1 decimal.

    Input
    -----
    scat_data : dict or str/Path, data/scattering_data.pkl (DASscatData.ipynb):
        'transmitted_traces'.daxis, 'reflection_traces'.daxis (n_pairs,)
        receiver depths [m], 'theta_i_deg' (n_pairs,) incidence angle [deg].
    save_dir : str/Path or None, folder of the saved PNG (created if needed);
        None -> not saved.
    fname : str, PNG file name.
    figsize : figure size [in].
    font_size : font size [pt] of the table.
    dpi : resolution of the saved PNG.

    Output
    ------
    (fig, ax); PNG saved to <save_dir>/<fname>, cropped to the table.
    """
    if not isinstance(scat_data, dict):                       # path of scattering_data.pkl
        with open(scat_data, "rb") as f:
            scat_data = pickle.load(f)
    dt = np.asarray(scat_data["transmitted_traces"].daxis, float)
    dr = np.asarray(scat_data["reflection_traces"].daxis, float)
    th = np.asarray(scat_data["theta_i_deg"], float)
    k_last = th.size - 1

    def row(k):
        return [f"{k}", f"{dt[k]:.1f}", f"{dr[k]:.1f}", f"{th[k]:.1f}"]

    cells = [row(0), [r"$\vdots$"] * 4, row(k_last)]          # first, dots, last
    cols = ["Index", "T depth (m)", "R depth (m)", r"$\theta_i$ (deg)"]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.axis("off")
    tab = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(font_size)
    tab.auto_set_column_width(list(range(len(cols))))          # columns just wide enough for the text
    tab.scale(1.08, 1.5)                                       # small side padding, row height
    for (r, c), cell in tab.get_celld().items():
        cell.set_facecolor("white")
        if r == 0:
            cell.set_text_props(fontweight="bold")             # header
    fig.tight_layout()
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()                                      # table extent in inches -> crop to it
        bbox = tab.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(Path(save_dir) / fname, dpi=dpi, facecolor="white", bbox_inches=bbox.padded(0.02))
    return fig, ax


def inversion_table(run_dir, scat_data, save_dir="./plots/DASplots", fname=None,
                    font_size=12, dpi=400):
    """
    Purpose
    -------
    Table figure (white background, slim cells) of the group results of one
    cnct_search_RfromT run, one row per group file group<g>_<run>.pkl
    (DAS_modeling.extracting_search), at each group's best node (minimum of
    Csp_sum):
        Group | No. traces | Angles (deg) | Min cost norm | cn (m/Pa) | cn/ct | ct (m/Pa)
    A value at the edge of the search grid gets an asterisk, and so does the
    value derived from it (e.g. ratio_cn grid: cn/ct at its edge -> cn/ct* and
    ct*; cn at its edge -> cn* and ct*).

    Input
    -----
    run_dir : str/Path, run folder, e.g. inv_results/RfromT/weighted_ratio_cn_snr10
        (holds group<g>_<run>.pkl).
    scat_data : dict or str/Path, data/scattering_data.pkl; its 'theta_i_deg'
        gives the incidence-angle range of each group (pairs = out['idx']).
    save_dir : str/Path or None, folder of the PNG (created if needed);
        None -> not saved.
    fname : str or None, PNG name; None -> "inversion_table_<run>.png".
    font_size : table font size [pt].
    dpi : resolution of the saved PNG.

    Output
    ------
    (fig, ax, rows) -- rows: list of the table rows (strings); PNG cropped to
    the table saved to <save_dir>/<fname>.
    """
    run_dir = Path(run_dir)
    run = run_dir.name
    if not isinstance(scat_data, dict):                       # path of scattering_data.pkl
        with open(scat_data, "rb") as f:
            scat_data = pickle.load(f)
    theta = np.asarray(scat_data["theta_i_deg"], float)
    files = sorted(run_dir.glob(f"group*_{run}.pkl"),
                   key=lambda p: int(p.name[len("group"):p.name.index("_")]))
    if not files:
        raise FileNotFoundError(f"no group*_{run}.pkl in {run_dir}")

    rows = []
    for p in files:
        with open(p, "rb") as f:
            o = pickle.load(f)
        o = o[0] if isinstance(o, tuple) else o
        grid = o.get("grid")
        i, j = int(o["i_sum_best"]), int(o["j_sum_best"])
        n_i, n_j = np.shape(o["Csp_sum"])
        edge_p, edge_q = i in (0, n_i - 1), j in (0, n_j - 1)   # best node on the grid edge
        # which printed values sit on the edge (axis value + the one derived from it)
        if grid == "ratio_cn":                                   # axes cn/ct, cn; ct = cn / ratio
            star = {"ratio": edge_p, "cn": edge_q, "ct": edge_p or edge_q}
        elif grid == "ratio_ct":                                 # axes cn/ct, ct; cn = ratio ct
            star = {"ratio": edge_p, "ct": edge_q, "cn": edge_p or edge_q}
        else:                                                    # axes cn, ct; ratio derived
            star = {"cn": edge_p, "ct": edge_q, "ratio": edge_p or edge_q}
        cn, ct = float(o["cn_sum_best"]), float(o["ct_sum_best"])
        th = theta[np.asarray(o["idx"], int)]
        cost = float(np.nanmin(o["cost_norm"]))
        s = lambda k: "*" if star[k] else ""
        rows.append([str(o.get("group", len(rows))), str(th.size),
                     f"{th.max():.1f}-{th.min():.1f}", f"{cost:.2f}",
                     f"{cn:.2e}{s('cn')}", f"{cn / ct:.2f}{s('ratio')}", f"{ct:.2e}{s('ct')}"])

    cols = ["Group", "No. traces", "Angles (deg)", "Min cost norm",
            r"$c_n$ (m/Pa)", r"$c_n/c_t$", r"$c_t$ (m/Pa)"]
    fig, ax = plt.subplots(figsize=(6, 0.3 * (len(rows) + 1)), facecolor="white")
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(font_size)
    tab.auto_set_column_width(list(range(len(cols))))          # columns just wide enough
    tab.scale(1.05, 1.4)                                        # slim side padding, row height
    for cell in tab.get_celld().values():
        cell.set_facecolor("white")                             # plain (not bold) header
    fig.tight_layout()
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()                                       # crop the PNG to the table
        bbox = tab.get_window_extent(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(Path(save_dir) / (fname or f"inversion_table_{run}.png"), dpi=dpi,
                    facecolor="white", bbox_inches=bbox.padded(0.02))
    return fig, ax, rows


def plot_reflections_paper(drange, source, best_line_out, xr=0.0, step=2,
                           same_side_only=True, trans_lim=(3100.0, np.inf), xlim=(0.0, 100.0),
                           ray_lw=0.3, source_ms=5, label_size=10, legend_size=8, equal_aspect=False,
                           figsize=(5, 7.2), show_dpi=150,
                           save_path="./paper_plots/reflection_geometry.png", dpi=600):
    """
    Purpose
    -------
    Paper version of plot_reflections: source, fiber, reflector and the incident,
    reflected and transmitted rays (constant velocity, so the transmitted ray is the
    straight continuation of the incident ray through the reflector to the fiber).
        incident    source -> P          dark grey
        reflected   P -> receiver yr     golden orange #F1B91A
        transmitted P -> fiber           neon green #39FF14
    Transmitted rays are drawn only when they arrive at the fiber: the ray travels toward
    it (reflection point between fiber and source) and reaches it inside trans_lim
    (default: deeper than 3100 m, the top of the DAS cut window). Rays with the reflection point near or beyond the source x reach the
    fiber line only km away, or never. The depth window covers all drawn arrivals.
    The fiber is a single line. The reflector stops at the fiber (no crossing): solid
    from the last incident ray (farthest reflection point) to the reflection point
    closest to the fiber, dashed from there to the fiber. It is drawn after the rays,
    on top of them.
    Layout follows plot_scattering_rays (plots/DASplots/scattering_sketch.png): fiber on
    the left axis (x = 0), no title, no grid; depth increases upwards, as in plot_reflections.
    The depth span (~800 m) is much larger than xlim (100 m), so equal aspect (true
    angles) gives a very narrow strip; it is optional (equal_aspect=True).

    Input
    -----
    drange : (N,) receiver depths along the fiber [m] (reflection arrivals).
    source : (xs, ys) source position [m].
    best_line_out : dict from estimate_best_line ('line' = (m, b), 'xp', 'yp').
    xr : fiber x position [m].
    step : plot every step-th ray; default 2 = every second branch trace; 1 = all rays;
        None = about 40 rays.
    same_side_only : only rays whose receiver is on the source side of the reflector.
    trans_lim : (y_min, y_max) fiber depths [m] where transmitted rays may arrive
        (default = deeper than 3100 m, top of the DAS cut window); None = no limit
        (some rays then arrive km away or at negative depths).
    xlim : (x_min, x_max) horizontal range [m]; default (0, 100).
    ray_lw : line width of the rays [pt].
    source_ms : marker size of the source dot [pt].
    label_size : font size [pt] of tick numbers and axis labels.
    legend_size : font size [pt] of the legend.
    equal_aspect : True = same scale in x and depth (true ray angles, narrow figure).
    figsize : figure size [in].
    show_dpi : resolution of the figure shown in the notebook (sharper text on screen).
    save_path : PNG path, or None to skip saving.
    dpi : resolution of the saved PNG.

    Output
    ------
    (fig, ax)
    """
    xs, ys = map(float, source)
    xr = float(xr)
    yr = np.asarray(drange, float)
    xp = np.asarray(best_line_out["xp"], float)
    yp = np.asarray(best_line_out["yp"], float)
    m_line, b_line = map(float, best_line_out["line"])

    # rays to draw (same selection as plot_reflections)
    same_side = (m_line * xr - yr + b_line) * (m_line * xs - ys + b_line) > 0
    if step is None:
        step = max(1, len(yr) // 40)
    idx = np.arange(0, len(yr), int(step))
    if same_side_only:
        idx = idx[same_side[idx]]

    # transmitted ray: continue source -> P straight until it reaches the fiber (x = xr)
    ux, uy = xp - xs, yp - ys
    with np.errstate(divide="ignore", invalid="ignore"):
        yt = yp + (xr - xp) / ux * uy                         # fiber depth of the transmitted arrival
    hits = (xr - xp) * ux > 0                                 # ray moves toward the fiber
    if trans_lim is not None:                                 # ... and arrives on the recorded fiber
        hits &= (yt >= min(trans_lim)) & (yt <= max(trans_lim))
    idx_t = idx[hits[idx]]                                    # drawn rays with a transmitted arrival

    # reflector: last incident ray (farthest point) -> closest point (solid) -> fiber (dashed)
    x_far = xp[idx].max()
    x_near = xp[idx].min()                                    # reflection point closest to the fiber
    line_y = lambda x: m_line * np.asarray(x) + b_line

    # fiber: single line covering all drawn arrivals and the reflector end
    y_fib = np.r_[yr[idx], yt[idx_t], line_y(xr)]

    c_inc, c_ref, c_trans = "dimgrey", "#F1B91A", "#39FF14"  # dark grey, golden orange, neon green (as scattering_sketch.png)
    fig, ax = plt.subplots(figsize=figsize, dpi=show_dpi)
    for i in idx:                                             # rays first ...
        ax.plot([xs, xp[i]], [ys, yp[i]], color=c_inc, lw=ray_lw, zorder=1)
        ax.plot([xp[i], xr], [yp[i], yr[i]], color=c_ref, lw=ray_lw, zorder=1)
    for i in idx_t:
        ax.plot([xp[i], xr], [yp[i], yt[i]], color=c_trans, lw=ray_lw, zorder=1)
    ax.plot([xr, xr], [y_fib.min(), y_fib.max()], "b-", lw=2, zorder=2, clip_on=False)
    ax.plot([x_near, x_far], line_y([x_near, x_far]), "k-", lw=2, zorder=3)   # ... reflector on top
    ax.plot([xr, x_near], line_y([xr, x_near]), "k--", lw=2, zorder=3)
    ax.plot(xs, ys, "ro", ms=source_ms, zorder=4)

    # legend in the requested order (proxy handles)
    for kw in (dict(color="r", marker="o", ls="", ms=source_ms, label="source"),
               dict(color="b", lw=2, label="fiber"),
               dict(color="k", lw=2, label="reflector"),
               dict(color=c_inc, lw=ray_lw, label="incident rays"),     # same widths as in the figure
               dict(color=c_ref, lw=ray_lw, label="reflected rays"),
               dict(color=c_trans, lw=ray_lw, label="transmitted rays")):
        ax.plot([], [], **kw)

    pad = 0.03 * np.ptp(y_fib)                                # depth range: fiber + 3% margin
    ax.set_ylim(y_fib.min() - pad, y_fib.max() + pad)         # depth increases upwards (as plot_reflections)
    ax.set_xlim(xlim)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")              # true angles
    ax.set_xlabel("distance from fiber [m]", fontsize=label_size)
    ax.set_ylabel("length along fiber [m]", fontsize=label_size)
    ax.tick_params(axis="both", labelsize=label_size)
    ax.legend(loc="best", fontsize=legend_size)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig, ax
