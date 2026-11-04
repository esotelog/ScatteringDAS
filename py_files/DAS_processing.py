"""
DAS_processing: functions used by DASdata.ipynb (DAS pre-processing, reflected / direct /
transmitted S extraction and export) plus the low-pass helpers used by DASscatInv.ipynb
and DASref_modeling.py. Functions are copied unchanged from the previous module, kept in
../py_files_old/DAS_processing_old.py (it also has the functions removed here).
"""
import numpy as np
import pandas as pd
from scipy.signal import correlate  
from scipy.signal import hilbert
from scipy.signal.windows import hann, tukey
from scipy.fft import fft, fftfreq, ifft
import json
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt
from datetime import datetime
from jin_pylib import ProcessUtil


def load_picks_for_window(pickfile_path, tbegin, tend, daxis):
    """
    Load P/S picks from pickfile and express them on a windowed Data2D panel.

    Time: Stime/Ptime in CSV are on full-file axis (t=0 at file start).
          Returned times are on window axis: t_local = t_file - tbegin.

    Depth: matched by nearest Z in CSV to each daxis[i].

    Parameters
    ----------
    pickfile_path : str
        e.g. './data/pickfiles/pickfile_EventA.csv'
    tbegin : float
        File-time start of window [s] (same as select_time(tbegin, tend, ...))
    tend : float
        File-time end of window [s]
    daxis : 1D array
        DAS_filter.daxis after select_depth [m]

    Returns
    -------
    direct_picks : dict
        'z'        : daxis (copy)
        't_win'    : [0, tend - tbegin]
        'P_on_trace', 'S_on_trace' : local time [s] per trace (NaN = no pick)
        'z_p', 't_p', 'z_s', 't_s' : sparse picks inside window (for QC plot)
    """
    daxis = np.asarray(daxis, float)
    dbegin = daxis[0]
    dend = daxis[-1]
    t_win_len = tend - tbegin

    pf = pd.read_csv(pickfile_path)
    z_csv = pf['Z'].to_numpy(float)
    t_p_file = pf['Ptime'].to_numpy(float)
    t_s_file = pf['Stime'].to_numpy(float)

    n_tr = daxis.size
    P_on_trace = np.full(n_tr, np.nan)
    S_on_trace = np.full(n_tr, np.nan)

    for i, zi in enumerate(daxis):
        j = np.argmin(np.abs(z_csv - zi))

        if t_p_file[j] > 0:
            tp = t_p_file[j] - tbegin
            if 0 <= tp <= t_win_len and dbegin <= z_csv[j] <= dend:
                P_on_trace[i] = tp

        if t_s_file[j] > 0:
            ts = t_s_file[j] - tbegin
            if 0 <= ts <= t_win_len and dbegin <= z_csv[j] <= dend:
                S_on_trace[i] = ts

    # sparse picks (for overlay on wiggle)
    has_p = (t_p_file > 0) & (t_p_file - tbegin >= 0) & (t_p_file - tbegin <= t_win_len)
    has_p &= (z_csv >= dbegin) & (z_csv <= dend)
    has_s = (t_s_file > 0) & (t_s_file - tbegin >= 0) & (t_s_file - tbegin <= t_win_len)
    has_s &= (z_csv >= dbegin) & (z_csv <= dend)

    direct_picks = {
        'P_on_trace': P_on_trace,
        'S_on_trace': S_on_trace,
    }
    return direct_picks

def flatten_s_arrivals(das_obj, S_pick, t_ref=0.0, interpolate_picks=True):
    """
    Flatten traces using S pick times on das_obj.taxis.

    Parameters
    ----------
    das_obj : Data2D object
        e.g. DAS_filter
    S_pick : 1D array
        S arrival time [s] per trace (from pick_data['S_on_trace'])
    t_ref : float
        Align all S picks to this time [s]
    interpolate_picks : bool
        Fill NaN picks along daxis before shifting

    Returns
    -------
    das_flat : copy of das_obj with flattened .data
    S_pick_used : 1D array (possibly interpolated)
    """
    S_pick = np.asarray(S_pick, float).copy()
    daxis = np.asarray(das_obj.daxis, float)
    dt = das_obj.dt

    if S_pick.size != daxis.size:
        raise ValueError("S_pick length must match len(das_obj.daxis)")

    if interpolate_picks:
        m = ~np.isnan(S_pick)
        if m.sum() >= 2:
            S_pick = np.interp(daxis, daxis[m], S_pick[m])

    data_flat = das_obj.data.copy()
    n_tr = data_flat.shape[0]

    for i in range(n_tr):
        if np.isnan(S_pick[i]):
            continue
        k = int(round((S_pick[i] - t_ref) / dt))
        if k != 0:
            data_flat[i, :] = np.roll(das_obj.data[i, :], -k)

    das_flat = das_obj.copy()
    das_flat.data = data_flat
    return das_flat, S_pick

def traces_in_depth_interval(daxis, dbegin, dend):
    daxis = np.asarray(daxis, float)
    return np.where((daxis >= dbegin) & (daxis <= dend))[0]


def depth_to_trace_index(daxis, depth_m):
    daxis = np.asarray(daxis, float)
    return int(np.argmin(np.abs(daxis - depth_m)))


def cc_window_indices(taxis, t_center, win_s):
    taxis = np.asarray(taxis, float)
    t0 = t_center - win_s / 2
    t1 = t_center + win_s / 2
    i0 = int(np.searchsorted(taxis, t0, side="left"))
    i1 = int(np.searchsorted(taxis, t1, side="right"))
    if i1 <= i0:
        raise ValueError("CC window empty: check t_center, win_s, taxis")
    return i0, i1


def cc_lag_samples(seg, ref, max_lag_samples=None):
    """Lag [samples] for np.roll(trace, -lag). Positive = trace late vs ref."""
    seg = np.asarray(seg, float) - np.mean(seg)
    ref = np.asarray(ref, float) - np.mean(ref)
    n = len(ref)
    c = correlate(seg, ref, mode="full")
    lags = np.arange(-(n - 1), n)
    if max_lag_samples is not None:
        m = np.abs(lags) <= int(max_lag_samples)
        c = np.where(m, c, -np.inf)
    return int(lags[np.argmax(c)])


def cc_peak_after_lag(seg, ref, lag_samples):
    """Shape similarity in window after applying lag (roughly 0–1)."""
    seg = np.asarray(seg, float) - np.mean(seg)
    ref = np.asarray(ref, float) - np.mean(ref)
    seg_al = np.roll(seg, -int(lag_samples))
    c = correlate(seg_al, ref, mode="valid")
    denom = np.linalg.norm(seg_al) * np.linalg.norm(ref) + 1e-12
    return float(c.max() / denom)


def cc_lags_in_depth_gate(
    das_flat,
    dbegin,
    dend,
    ref_depth,
    t_center,
    win_s,
    max_lag_ms=None,
):
    """
    CC each trace in [dbegin, dend] to reference at ref_depth.
    No roll on gather — lags only, for QC plots.

    Returns
    -------
    dict with plot-ready 1D arrays (same length, one per trace in gate):
        depth_m, trace_i, t_cc, cc_peak
    Scalars: ref_depth, ref_i, t_center, win_s
    """
    data = das_flat.data
    daxis = np.asarray(das_flat.daxis, float)
    taxis = np.asarray(das_flat.taxis, float)
    dt = float(das_flat.dt)

    i0, i1 = cc_window_indices(taxis, t_center, win_s)

    # nearest-channel indices for gate ends and reference
    i_begin = depth_to_trace_index(daxis, dbegin)
    i_end = depth_to_trace_index(daxis, dend)
    ref_i = depth_to_trace_index(daxis, ref_depth)

    lo, hi = (i_begin, i_end) if i_begin <= i_end else (i_end, i_begin)
    trace_ids = np.arange(lo, hi + 1, dtype=int)

    if ref_i < lo or ref_i > hi:
       raise ValueError(
         f"ref_depth (nearest i={ref_i}, d={daxis[ref_i]:.3f}) "
         f"must lie inside gate [{lo}, {hi}] "
         f"(d={daxis[lo]:.3f} .. {daxis[hi]:.3f})"
       )

    ref_seg = data[ref_i, i0:i1]

    max_lag_samples = None
    if max_lag_ms is not None:
        max_lag_samples = int(round(max_lag_ms * 1e-3 / dt))

    n = len(trace_ids)
    depth_m = daxis[trace_ids].copy()
    lag_samples = np.empty(n, dtype=int)
    t_cc = np.empty(n, dtype=float)
    cc_peak = np.empty(n, dtype=float)

    for k, i in enumerate(trace_ids):
        seg = data[i, i0:i1]
        lag = cc_lag_samples(seg, ref_seg, max_lag_samples=max_lag_samples)
        lag_samples[k] = lag
        t_cc[k] = t_center + lag * dt
        cc_peak[k] = cc_peak_after_lag(seg, ref_seg, lag)

    return {
        "depth_m": depth_m,
        "trace_i": trace_ids,
        "lag_samples": lag_samples,
        "t_cc": t_cc,
        "cc_peak": cc_peak,
        "ref_depth": float(daxis[ref_i]),
        "ref_i": ref_i,
        "t_center": float(t_center),
        "win_s": float(win_s),
    }

def apply_cc_lags(das_obj, trace_i, lag_samples):
    """
    Apply CC lags to selected traces only.

    Parameters
    ----------
    das_obj : Data2D-like
    trace_i : 1D array of trace indices (e.g. cc["trace_i"])
    lag_samples : 1D array, same length (e.g. cc["lag_samples"])

    Returns
    -------
    das_out : copy of das_obj with rolls applied on trace_i
    """
    trace_i = np.asarray(trace_i, int)
    lag_samples = np.asarray(lag_samples, int)

    if trace_i.size != lag_samples.size:
        raise ValueError("trace_i and lag_samples must have the same length")

    das_out = das_obj.copy()
    for j, i in enumerate(trace_i):
        lag = int(lag_samples[j])
        if lag != 0:
            das_out.data[i] = np.roll(das_obj.data[i], -lag)

    return das_out


def apply_cc_lags_from_gate(das_obj, cc_result):
    """
    Wrapper: apply lags from cc_lags_in_depth_gate() return dict.
    """
    return apply_cc_lags(das_obj, cc_result["trace_i"], cc_result["lag_samples"])


def stack_traces(das_obj, trace_ids, method="median", tbegin=None, tend=None):
    """
    Stack selected traces into one pilot trace.

    Parameters
    ----------
    das_obj : Data2D-like (.data, .taxis)
    trace_ids : 1D array of trace row indices
    method : 'median' or 'mean'
    tbegin, tend : float, optional
        If both set, stack only samples in [tbegin, tend) on das_obj.taxis.
        If None, stack full trace length.

    Returns
    -------
    stack_trace : 1D ndarray
    taxis_stack : 1D ndarray (time axis for stack_trace)
    """
    trace_ids = np.asarray(trace_ids, int)
    data = das_obj.data
    taxis = np.asarray(das_obj.taxis, float)

    if trace_ids.size == 0:
        raise ValueError("trace_ids is empty")

    if tbegin is not None and tend is not None:
        i0 = int(np.searchsorted(taxis, tbegin, side="left"))
        i1 = int(np.searchsorted(taxis, tend, side="right"))
        block = data[trace_ids, i0:i1]
        taxis_stack = taxis[i0:i1].copy()
    else:
        block = data[trace_ids, :]
        taxis_stack = taxis.copy()

    if method == "median":
        stack_trace = np.median(block, axis=0)
    elif method == "mean":
        stack_trace = np.mean(block, axis=0)
    else:
        raise ValueError("method must be 'median' or 'mean'")

    return stack_trace, taxis_stack

def stack_traces_mean(das_obj, trace_ids, method="median", tbegin=None, tend=None):
    """
    Stack selected traces into one pilot trace.

    Parameters
    ----------
    das_obj : Data2D-like (.data, .taxis, .daxis)
    trace_ids : 1D array of trace row indices
    method : 'median' or 'mean'
    tbegin, tend : float, optional
        If both set, stack only samples in [tbegin, tend) on das_obj.taxis.
        If None, stack full trace length.

    Returns
    -------
    dict with keys:
        trace_stack      : 1D ndarray (stacked pilot; median or mean)
        mean_trace_stack : 1D ndarray (trace_stack / number_traces)
        time             : 1D ndarray (time axis for the stack)
        number_traces    : int (number of traces stacked)
        depth_traces     : 1D ndarray (depths of stacked traces from .daxis)
    """
    trace_ids = np.asarray(trace_ids, int)
    data = das_obj.data
    taxis = np.asarray(das_obj.taxis, float)
    daxis = np.asarray(das_obj.daxis, float)

    if trace_ids.size == 0:
        raise ValueError("trace_ids is empty")

    if np.any(trace_ids < 0) or np.any(trace_ids >= data.shape[0]):
        raise ValueError("trace_ids out of range for das_obj.data")

    if tbegin is not None and tend is not None:
        i0 = int(np.searchsorted(taxis, tbegin, side="left"))
        i1 = int(np.searchsorted(taxis, tend, side="right"))
        block = data[trace_ids, i0:i1]
        taxis_stack = taxis[i0:i1].copy()
    else:
        block = data[trace_ids, :]
        taxis_stack = taxis.copy()

    if method == "median":
        stack_trace = np.median(block, axis=0)
    elif method == "mean":
        stack_trace = np.mean(block, axis=0)
    else:
        raise ValueError("method must be 'median' or 'mean'")

    number_traces = int(block.shape[0])
    depth_traces = daxis[trace_ids].copy()

    return {
        "trace_stack": stack_trace,
        "time": taxis_stack,
        "number_traces": number_traces,
        "depth_traces": depth_traces,
        "traces": np.asarray(block, float).copy(),  # before stack
    }

def _pilot_cc_segment(pilot, taxis, i0, i1):
    """Pilot window segment: full-length pilot[taxis] or window-length stack."""
    pilot = np.asarray(pilot, float).ravel()
    taxis = np.asarray(taxis, float)
    nwin = i1 - i0
    if pilot.size == taxis.size:
        return pilot[i0:i1]
    if pilot.size == nwin:
        return pilot
    raise ValueError(
        f"pilot length {pilot.size} must match len(taxis)={taxis.size} "
        f"or CC window length {nwin}"
    )


def cc_peak_after_lag_abs(seg, ref, lag_samples):
    """|correlation| at chosen lag (0–1 scale), for QC with abs-CC lags."""
    seg = np.asarray(seg, float) - np.mean(seg)
    ref = np.asarray(ref, float) - np.mean(ref)
    seg_al = np.roll(seg, -int(lag_samples))
    c = correlate(seg_al, ref, mode="valid")
    denom = np.linalg.norm(seg_al) * np.linalg.norm(ref) + 1e-12
    return float(np.abs(c).max() / denom)

def cc_lags_vs_pilot(
    das_obj,
    pilot,
    dbegin,
    dend,
    t_center,
    win_s,
    max_lag_ms=None,
):
    """
    CC traces in [dbegin, dend] to a pilot wavelet (|CC| peak for lag).

    Parameters
    ----------
    das_obj : Data2D-like (.data, .daxis, .taxis, .dt)
    pilot : 1D array
        Full-length on das_obj.taxis, or window-length for [t_center ± win_s/2].
    dbegin, dend : float
        Depth gate [m], same as cc_lags_in_depth_gate.
    t_center, win_s : float
        CC window [s].
    max_lag_ms : float or None
        Max |lag| when searching peak [ms].

    Returns
    -------
    dict with 1D arrays (one per trace in gate):
        depth_m, trace_i, lag_samples, t_cc, cc_peak
    Scalars: t_center, win_s
    """
    data = das_obj.data
    daxis = np.asarray(das_obj.daxis, float)
    taxis = np.asarray(das_obj.taxis, float)
    dt = float(das_obj.dt)

    i0, i1 = cc_window_indices(taxis, t_center, win_s)
    ref_seg = _pilot_cc_segment(pilot, taxis, i0, i1)

    trace_ids = traces_in_depth_interval(daxis, dbegin, dend)
    if trace_ids.size == 0:
        raise ValueError("no traces in [dbegin, dend]")

    max_lag_samples = None
    if max_lag_ms is not None:
        max_lag_samples = int(round(max_lag_ms * 1e-3 / dt))

    n = len(trace_ids)
    depth_m = daxis[trace_ids].copy()
    lag_samples = np.empty(n, dtype=int)
    t_cc = np.empty(n, dtype=float)
    cc_peak = np.empty(n, dtype=float)

    for k, i in enumerate(trace_ids):
        seg = data[i, i0:i1]
        lag =cc_lag_samples(seg, ref_seg, max_lag_samples=max_lag_samples)
        lag_samples[k] = lag
        t_cc[k] = t_center + lag * dt
        cc_peak[k] = cc_peak_after_lag_abs(seg, ref_seg, lag)

    return {
        "depth_m": depth_m,
        "trace_i": trace_ids,
        "lag_samples": lag_samples,
        "t_cc": t_cc,
        "cc_peak": cc_peak,
        "t_center": float(t_center),
        "win_s": float(win_s),
    }

def cc_lags_vs_pilot_tgate(
    das_obj,
    pilot,
    dbegin,
    dend,
    tref,
    tb,
    tend,
    max_lag_ms=None,
):
    """
    CC traces in [dbegin, dend] to a pilot wavelet (|CC| peak for lag).

    Same as cc_lags_vs_pilot, but the CC window is given explicitly as
    [tb, tend] instead of (t_center, win_s).

    Parameters
    ----------
    das_obj : Data2D-like (.data, .daxis, .taxis, .dt)
    pilot : 1D array
        Full-length on das_obj.taxis, or window-length for [tb, tend].
    dbegin, dend : float
        Depth gate [m].
    tref : float
        Reference time [s] used for t_cc = tref + lag * dt.
    tb, tend : float
        CC window start/end [s]. Require tend > tb.
    max_lag_ms : float or None
        Max |lag| when searching peak [ms].

    Returns
    -------
    dict with 1D arrays (one per trace in gate):
        depth_m, trace_i, lag_samples, t_cc, cc_peak
    Scalars: tref, tb, tend
    """
    data = das_obj.data
    daxis = np.asarray(das_obj.daxis, float)
    taxis = np.asarray(das_obj.taxis, float)
    dt = float(das_obj.dt)

    tref = float(tref)
    tb = float(tb)
    tend = float(tend)
    if tend <= tb:
        raise ValueError("require tend > tb")

    i0 = int(np.searchsorted(taxis, tb, side="left"))
    i1 = int(np.searchsorted(taxis, tend, side="right"))
    if i1 <= i0:
        raise ValueError("CC window empty: check tb, tend, taxis")

    ref_seg = _pilot_cc_segment(pilot, taxis, i0, i1)

    trace_ids = traces_in_depth_interval(daxis, dbegin, dend)
    if trace_ids.size == 0:
        raise ValueError("no traces in [dbegin, dend]")

    max_lag_samples = None
    if max_lag_ms is not None:
        max_lag_samples = int(round(max_lag_ms * 1e-3 / dt))

    n = len(trace_ids)
    depth_m = daxis[trace_ids].copy()
    lag_samples = np.empty(n, dtype=int)
    t_cc = np.empty(n, dtype=float)
    cc_peak = np.empty(n, dtype=float)

    for k, i in enumerate(trace_ids):
        seg = data[i, i0:i1]
        lag = cc_lag_samples(seg, ref_seg, max_lag_samples=max_lag_samples)
        lag_samples[k] = lag
        t_cc[k] = tref + lag * dt
        cc_peak[k] = cc_peak_after_lag_abs(seg, ref_seg, lag)

    return {
        "depth_m": depth_m,
        "trace_i": trace_ids,
        "lag_samples": lag_samples,
        "t_cc": t_cc,
        "cc_peak": cc_peak,
        "tref": tref,
        "tb": tb,
        "tend": tend,
    }

def reflection_arrival(source, xf, yf):
        """
        Calculate the arrival at the fiber given the source and the reflection points
        source: source location (xs,ys)
        fline: fracture line (m,b), m is slope and b is intercept
        xf : scalar or 1D array [m], x coordinate of the fracture line
        yf : scalar or 1D array [m], y coordinate of the fracture line
        """
        #source
        xs,ys = source
        
        #receiver
        yr= (ys-yf)*xf/(xs-xf)+yf
        return yr

def reflection_tt(source,xf,yf,v):
    """
    Calculate the travel time of a ray reflecting at the fracture plane
    source: source location (xs,ys)
    fline: fracture line (m,b), m is slope and b is intercept
    yr: receiver location
    v: velocity
    """


    yr=reflection_arrival(source,xf,yf)
    xs,ys = source
    x1=xs-xf
    y1=ys-yf
    y2=yr-yf

    d1 = np.hypot(x1, y1)   # (N,) — x1 array, y1 scalar OK
    d2 = np.hypot(xf, y2)
    tt=(d1+d2)/v

    return yr,tt

def file_reflection_tt(source,fline,xf,v,tfileb=0.0):
    """
    Calculate the travel time of a ray reflecting at the fracture plane
    with respect to the beginning of the cropped fileand the source time
    source: source time and location (ts,xs,ys)
    fline: fracture line (m,b), m is slope and b is intercept
    xf : scalar or 1D array [m], x coordinate of the fracture line
    v: velocity
    tfile_begin: beginning of cropped file

    """
    ts=source[0]
    source_pos=source[1:]

    #fracture
    xf = np.asarray(xf, dtype=float)
    m,b = fline
    yf=m*xf+b
    yr,tt=reflection_tt(source_pos,xf,yf,v)
    tt_file=ts+tt-tfileb
    return yr,tt_file

def fiber_reflection_tt(
    source,
    fline,
    xf,
    daxis,
    v,
    tfileb=0.0,
    max_map_dist=None,
):
    """
    SS reflection times on panel axis, one value per daxis channel.

    Computes (yr_calc, tt) for samples xf on y = m*xf + b, interpolates tt
    onto daxis, fills NaNs on the branch span [yr_min, yr_max], and returns
    branch endpoints on daxis.

    Parameters
    ----------
    source : (ts, xs, ys)
    fline : (m, b)
    xf : 1D array
        Fracture x samples [m].
    daxis : 1D array
        Receiver depths [m].
    v : float
        S velocity [m/s].
    tfileb : float
        Start of cropped panel [s].
    max_map_dist : float or None
        Unused (kept for API compatibility).

    Returns
    -------
    t_arrival : 1D array, len(daxis)
        Panel times; NaN outside branch span on daxis.
    (db, ib) : tuple
        Depth [m] and index on daxis at branch start (yr_min side).
    (de, ie) : tuple
        Depth [m] and index on daxis at branch end (yr_max side).
    """
    daxis = np.asarray(daxis, float)
    xf = np.asarray(xf, float)

    if daxis.ndim != 1 or xf.ndim != 1:
        raise ValueError("daxis and xf must be 1D")
    if xf.size == 0:
        raise ValueError("xf is empty")

    xs = source[1]
    if np.any(np.isclose(xf, xs)):
        raise ValueError("xf must not equal xs")

    yr_calc, tt_calc = file_reflection_tt(source, fline, xf, v, tfileb)
    yr_calc = np.asarray(yr_calc, float)
    tt_calc = np.asarray(tt_calc, float)

    ok = np.isfinite(yr_calc) & np.isfinite(tt_calc)
    yr_calc = yr_calc[ok]
    tt_calc = tt_calc[ok]
    if yr_calc.size < 2:
        raise ValueError("need at least 2 valid (yr, tt) samples")

    order = np.argsort(yr_calc)
    yr_s = yr_calc[order]
    tt_s = tt_calc[order]

    keep = np.concatenate([[True], np.diff(yr_s) > 1e-6])
    yr_s = yr_s[keep]
    tt_s = tt_s[keep]

    yr_min = float(yr_s[0])
    yr_max = float(yr_s[-1])

    # map model tt(yr) onto all daxis
    t_arrival = np.interp(
        daxis, yr_s, tt_s,
        left=np.nan, right=np.nan,
    )

    # branch span on daxis: channels inside [yr_min, yr_max]
    in_span = (daxis >= yr_min) & (daxis <= yr_max)
    if not np.any(in_span):
        raise ValueError("no daxis channels within model yr span")

    ib = int(np.flatnonzero(in_span)[0])
    ie = int(np.flatnonzero(in_span)[-1])
    db = float(daxis[ib])
    de = float(daxis[ie])

    # inside branch: fill NaN gaps by interpolating along daxis
    seg = t_arrival[ib:ie + 1].copy()
    d_seg = daxis[ib:ie + 1]
    m = np.isfinite(seg)
    if m.sum() < 2:
        raise ValueError("need at least 2 finite samples on branch span")
    if np.any(~m):
        seg[~m] = np.interp(d_seg[~m], d_seg[m], seg[m])
    t_arrival[ib:ie + 1] = seg

    # outside branch span -> NaN
    t_arrival[:ib] = np.nan
    t_arrival[ie + 1:] = np.nan

    return {
        "SStt": t_arrival,
        "depth_begin": db,
        "begin_id": ib,
        "depth_end": de,
        "end_id": ie
        }

def snap_sstt_to_positive_peak(
    das_flat,
    SStt,
    branch_ids,
    time_gate,
    peak_only=True,
):
    """
    Refine SStt on flattened data by snapping to max (positive) peak
    in a time window on each trace in the branch.

    Parameters
    ----------
    das_flat : Data2D-like
        Flattened panel (e.g. after flatten_s_arrivals).
    SStt : 1D array
        Model SS times [s], len(daxis); same frame as used for flattening.
    branch_ids : (begin_id, end_id)
        Inclusive trace indices on daxis where SStt branch is valid.
    time_gate : (tref, tbegin, tend)
        tref   : flatten reference time [s] (same t_ref used in flatten_s_arrivals)
        tbegin : start of search window on flat panel [s]
        tend   : end of search window on flat panel [s]
        Require tbegin < tref < tend.
    peak_only : bool
        If True, pick max positive sample; else max |amp|.

    Returns
    -------
    dict with keys:
        modified_SStt : 1D array (copy of SStt, refined on branch)
        time_shift    : 1D array (modified_SStt - SStt) [s]
        n_shifted     : int, traces with |time_shift| > 0.5*dt on branch
    """
    SStt = np.asarray(SStt, float).copy()
    daxis = np.asarray(das_flat.daxis, float)
    taxis = np.asarray(das_flat.taxis, float)
    dt = float(das_flat.dt)
    data = das_flat.data

    if SStt.size != daxis.size:
        raise ValueError("SStt length must match len(das_flat.daxis)")

    ib, ie = (int(branch_ids[0]), int(branch_ids[1]))
    if ib < 0 or ie >= len(daxis) or ib > ie:
        raise ValueError("invalid branch_ids")

    if len(time_gate) != 3:
        raise ValueError("time_gate must be (tref, tbegin, tend)")

    tref, tbegin, tend = (float(time_gate[0]), float(time_gate[1]), float(time_gate[2]))
    if not (tbegin < tref < tend):
        raise ValueError("require tbegin < tref < tend")
    if tend <= tbegin:
        raise ValueError("require tend > tbegin")

    i0 = int(np.searchsorted(taxis, tbegin))
    i1 = int(np.searchsorted(taxis, tend, side="right"))
    if i1 <= i0:
        raise ValueError("time gate empty on das_flat.taxis")

    modified_SStt = SStt.copy()
    time_shift = np.zeros_like(SStt)
    n_shifted = 0

    for i in range(ib, ie + 1):
        if not np.isfinite(SStt[i]):
            continue

        seg = data[i, i0:i1]
        if seg.size == 0:
            continue

        if peak_only:
            j = int(np.argmax(seg))
        else:
            j = int(np.argmax(np.abs(seg)))

        t_peak = float(taxis[i0 + j])
        shift = t_peak - tref
        modified_SStt[i] = SStt[i] + shift
        time_shift[i] = shift

        if abs(shift) > 0.5 * dt:
            n_shifted += 1

    return {
        "modified_SStt": modified_SStt,
        "time_shift": time_shift,
        "n_shifted": n_shifted,
    }

def tt_line(source, receiver, line, v, xr=0.0):
    """
    SS reflection travel times off a line y = m*x + b (image-source method).

    Parameters
    ----------
    source : (xs, ys)
        Source position [m].
    receiver : array-like
        Receiver y positions [m] along the fibre (depth).
    line : (m, b)
        Reflecting line: y = m*x + b.
    v : float
        Medium velocity [m/s].
    xr : float, optional
        Receiver x position [m]. Default 0.

    Returns
    -------
    dict with keys:
        mirror_source : (xs_p, ys_p)   # S'
        xp, yp          : 1D arrays      # reflection points on the line
        tt              : 1D arrays      # travel times [s]
    """
    xs, ys = map(float, source)
    m, b = map(float, line)
    yr = np.asarray(receiver, dtype=float)
    v = float(v)
    xr = float(xr)

    # --- Fig. 1: mirror source S' across line y = m*x + b ---
    denom = m * m + 1.0
    t = (m * xs - ys + b) / denom
    xs_p = xs - 2.0 * m * t
    ys_p = ys + 2.0 * t

    # --- Reflection point: intersection of line y=m*x+b and line S'->R ---
    # (equivalent to fig. 2 path-slope method)
    dx_sr = xr - xs_p
    dy_sr = yr - ys_p
    den = dy_sr - m * dx_sr

    lam = np.where(
        np.abs(den) > 1e-12,
        (m * xs_p + b - ys_p) / den,
        np.nan,
    )

    xp = xs_p + lam * dx_sr
    yp = m * xp + b

    # --- Fig. 3: travel time = distance S'R / v ---
    dist = np.hypot(xr - xs_p, yr - ys_p)
    tt = dist / v

    return {
        "mirror_source": (xs_p, ys_p),
        "xp": xp,
        "yp": yp,
        "tt": tt
    }

def _tt_from_line(source, receiver, line, v, xr=0.0):
    """Helper: travel times only from tt_line."""
    return tt_line(source, receiver, line, v, xr=xr)["tt"]


def _stacked_residuals(tt_obs, tt_calc):
    """Residual vector and sum of squares (valid points only)."""
    tt_obs = np.asarray(tt_obs, float)
    tt_calc = np.asarray(tt_calc, float)
    valid = np.isfinite(tt_obs) & np.isfinite(tt_calc)
    r = np.zeros_like(tt_obs, dtype=float)
    r[valid] = tt_obs[valid] - tt_calc[valid]
    r[~valid] = 0.0
    S = float(np.sum(r[valid] ** 2))
    return r, S, valid


def _jacobian_tt_line(source, receiver, m, b, v, xr, dm, db):
    """
    Numerical Jacobian of tt w.r.t. (m, b), step 3 in LM algorithm.
    J is (n x 2): columns are dt/dm, dt/db.
    """
    line0 = (m, b)
    tt0 = _tt_from_line(source, receiver, line0, v, xr)

    tt_m = _tt_from_line(source, receiver, (m + dm, b), v, xr)
    tt_b = _tt_from_line(source, receiver, (m, b + db), v, xr)

    J = np.column_stack([(tt_m - tt0) / dm, (tt_b - tt0) / db])
    return J, tt0


def estimate_best_line(
    source,
    receiver,
    line,
    v,
    tt_data,
    lam=0.01,
    epsilon=1e-10,
    xr=0.0,
    max_iter=1000,
    dm=None,
    db=None,
):
    """
    Levenberg-Marquardt fit of reflecting line y = m*x + b to travel-time data.

    Minimizes sum_i (t_obs,i - t_calc,i)^2 using tt_line forward model.

    Parameters
    ----------
    source : (xs, ys)
    receiver : array-like
        Receiver depths yr [m] (xr fixed, default 0).
    line : (m, b)
        Initial guess for reflecting line.
    v : float
        Medium velocity [m/s].
    tt_data : array-like
        Observed travel times [s], same length as receiver.
        Must be in same time domain as tt_line output (ray time).
    lam : float
        Initial damping factor (figure step 1).
    epsilon : float
        Convergence threshold on relative change in S.
    xr : float
        Receiver x position [m].
    max_iter : int
        Maximum LM iterations.
    dm, db : float or None
        Finite-difference steps for Jacobian. Auto if None.

    Returns
    -------
    dict with keys:
        line        : (m, b) best fit
        xp, yp      : reflection points
        tt          : modeled travel times at best fit
        mirror_source : mirrored source (xs_p, ys_p)
        residuals   : t_obs - t_calc
        S           : final sum of squares
        n_iter      : number of accepted iterations
        success     : bool
    """
    m, b = map(float, line)
    tt_obs = np.asarray(tt_data, float)
    yr = np.asarray(receiver, float)

    if tt_obs.shape != yr.shape:
        raise ValueError("tt_data and receiver must have the same shape")

    if dm is None:
        dm = max(1e-6, 1e-4 * max(abs(m), 1.0))
    if db is None:
        db = max(1e-6, 1e-2 * max(abs(b), 1.0))

    theta = np.array([m, b], dtype=float)
    damping = float(lam)

    # --- Step 2: initial model and residuals ---
    out0 = tt_line(source, yr, (theta[0], theta[1]), v, xr=xr)
    r, S_old, valid = _stacked_residuals(tt_obs, out0["tt"])
    if not np.any(valid):
        raise ValueError("No valid (finite) tt_data / tt_calc pairs")

    n_accept = 0

    for it in range(max_iter):
        m, b = theta
        J, tt_calc = _jacobian_tt_line(source, yr, m, b, v, xr, dm, db)

        # mask invalid rows in Jacobian/residuals consistently
        J = J[valid, :]
        r_valid = r[valid]

        # --- Step 4: damped normal equation ---
        JTJ = J.T @ J
        rhs = J.T @ r_valid
        A = JTJ + damping * np.eye(2)
        try:
            delta = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue

        # --- Step 5: trial update ---
        theta_trial = theta + delta
        out_trial = tt_line(source, yr, (theta_trial[0], theta_trial[1]), v, xr=xr)
        _, S_trial, _ = _stacked_residuals(tt_obs, out_trial["tt"])

        if S_trial < S_old:
            # accept
            theta = theta_trial
            r, S_new, valid = _stacked_residuals(tt_obs, out_trial["tt"])
            rel_change = abs(S_old - S_new) / max(S_old, 1e-30)
            S_old = S_new
            damping = max(damping / 10.0, 1e-12)
            n_accept += 1

            if rel_change < epsilon:
                break
        else:
            # reject
            damping = min(damping * 10.0, 1e12)

    # final forward model at best theta
    m_best, b_best = theta
    out = tt_line(source, yr, (m_best, b_best), v, xr=xr)
    r, S_final, _ = _stacked_residuals(tt_obs, out["tt"])

    return {
        "line": (m_best, b_best),
        "xp": out["xp"],
        "yp": out["yp"],
        "tt": out["tt"],
        "mirror_source": out["mirror_source"],
        "residuals": r,
        "S": S_final,
        "n_iter": n_accept,
        "success": n_accept > 0 or S_final < np.inf,
    }

def incident_reflection(source_pos, yr, xp, yp, m, xr=0.0):
    """
    Incident and reflected angles at reflection points on line y = m*x + b.

    Normal points toward increasing y (depth), same side as incident ray.
    This `normal` is the actual (possibly tilted) reflector geometry and is
    what theta_i/theta_r are measured against -- it is unaffected by the
    note below.

    Polarizations p_trans (from u_inc/u_trans) and p_ref (from u_ref) are
    each computed via _polarization_from_n_u(n, ...), which internally
    re-aligns n's sign to match each leg's own propagation sense before
    applying (n x u) x u (see that function's docstring). This matters
    because u_inc/u_trans and u_ref generally travel in opposite senses
    relative to `n` (down to the reflector, back up from it), and Aki &
    Richards' SV polarization has a different sign for a downgoing vs.
    upgoing leg -- so a single un-aligned n would only ever be correct for
    one of the two legs. With the alignment, both n_p_trans_dot and
    n_p_ref_dot are consistent with the same Aki & Richards convention that
    DAS_inversion.RPS_coef was derived under.

    Parameters
    ----------
    source_pos : (xs, ys)
        Source position [m].
    yr : array-like
        Receiver y positions [m].
    xp, yp : array-like
        Reflection point coordinates [m], same length as yr.
    m : float
        Slope of reflecting interface.
    xr : float, optional
        Receiver x position [m]. Default 0.

    Returns
    -------
    dict with keys:
        theta_i_deg, theta_r_deg : incident / reflected angles from normal [deg]
        theta_i, theta_r         : same angles [rad]
        u_inc, u_ref             : (N, 2) unit vectors [inc_x, inc_y], [ref_x, ref_y]
        normal                        : (2,) unit normal [n_x, n_y] (reflector geometry,
                                         not the polarization reference -- see above)
    """
    xs, ys = map(float, source_pos)
    yr = np.asarray(yr, dtype=float)
    xp = np.asarray(xp, dtype=float)
    yp = np.asarray(yp, dtype=float)
    m = float(m)
    xr = float(xr)

    if not (yr.shape == xp.shape == yp.shape):
        raise ValueError("yr, xp, and yp must have the same shape")

    # --- unit normal toward increasing y (figure 1) ---
    den = np.sqrt(1.0 + m * m)
    n = np.array([-m / den, 1.0 / den], dtype=float)   # points toward +y

    # --- unit incident vector: S -> P (figure 2) ---
    dx_inc = xp - xs
    dy_inc = yp - ys
    r_inc = np.hypot(dx_inc, dy_inc)
    u_inc_x = dx_inc / r_inc
    u_inc_y = dy_inc / r_inc
    u_inc = np.column_stack([u_inc_x, u_inc_y])

    # --- unit reflected vector: P -> R (figure 2) ---
    dx_ref = xr - xp
    dy_ref = yr - yp
    r_ref = np.hypot(dx_ref, dy_ref)
    u_ref_x = dx_ref / r_ref
    u_ref_y = dy_ref / r_ref
    u_ref = np.column_stack([u_ref_x, u_ref_y])

    #---distance from source to receiver (reflection leg)
    source_rr=r_inc + r_ref

    # --- angles relative to normal (figure 3) ---
    cos_i = np.clip(np.abs(u_inc_x * n[0] + u_inc_y * n[1]), 0.0, 1.0)
    cos_r = np.clip(np.abs(u_ref_x * n[0] + u_ref_y * n[1]), 0.0, 1.0)

    theta_i = np.arccos(cos_i)
    theta_r = np.arccos(cos_r)
    theta_i_deg = np.degrees(theta_i)
    theta_r_deg = np.degrees(theta_r)

    # transmission aliases (same medium: transmitted = incident)
    u_trans = u_inc
    theta_t = theta_i
    theta_t_deg = theta_i_deg

    # polarizations: p = (n × u) × u
    p_trans = _polarization_from_n_u(n, u_trans)
    p_ref = _polarization_from_n_u(n, u_ref)

    # f · p and f · u  (N,)
    # fiber unit tangent (parallel to fiber, +y)
    f = np.array([0.0, 1.0], dtype=float)
    n_p_trans_dot = p_trans @ f
    n_p_ref_dot = p_ref @ f
    n_u_trans_dot = u_trans @ f
    n_u_ref_dot = u_ref @ f

    return {
    "theta_i": theta_i,
    "theta_r": theta_r,
    "theta_t": theta_t,
    "theta_i_deg": theta_i_deg,
    "theta_r_deg": theta_r_deg,
    "theta_t_deg": theta_t_deg,
    "u_inc": u_inc,
    "u_trans": u_trans,
    "u_ref": u_ref,
    "p_trans": p_trans,
    "p_ref": p_ref,
    "normal": n,
    "source_pinc":r_inc,
    "pinc_rr": r_ref,
    "source_rr": source_rr,
    "n_p_trans_dot": n_p_trans_dot,
    "n_p_ref_dot": n_p_ref_dot,
    "n_u_trans_dot": n_u_trans_dot,
    "n_u_ref_dot": n_u_ref_dot,
    }

def transmission(source_pos, ui, v, yr, xt=0.0):
    """
    Straight-through transmission: ray from source along ui hits fibre at x = xt.

    Only keeps rays that:
      - point toward the fibre (for xs > xt: ux < 0)
      - hit forward along the ray (s > 0)
      - land within the depth range spanned by yr

    Parameters
    ----------
    source_pos : (xs, ys)
        Source position [m].
    ui : array-like, shape (N, 2)
        Unit incident vectors [ux, uy].
    v : float
        Medium velocity [m/s].
    yr : array-like
        Fibre depth guide [m] (e.g. daxis or drange).
        Used only for yr.min() and yr.max() to trim yt.
        Does not need same length as ui.
    xt : float, optional
        Fibre x position [m]. Default 0.

    Returns
    -------
    dict with keys:
        yt, tt, dist_tr, idx, ui, xt, n_good, n_total, yr_min, yr_max
    """
    xs, ys = map(float, source_pos)
    ui = np.asarray(ui, dtype=float)
    yr = np.asarray(yr, dtype=float)
    v = float(v)
    xt = float(xt)

    if ui.ndim == 1:
        ui = ui.reshape(1, 2)
    if ui.ndim != 2 or ui.shape[1] != 2:
        raise ValueError("ui must have shape (N, 2) or (2,)")

    yr_min = float(np.min(yr))
    yr_max = float(np.max(yr))

    ux = ui[:, 0]
    uy = ui[:, 1]
    dx_fiber = xt - xs

    toward_fiber = dx_fiber * ux > 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        s = dx_fiber / ux

    yt_all = ys + s * uy
    dist_all = np.abs(s)
    tt_all = dist_all / v

    good = (
        toward_fiber
        & np.isfinite(yt_all)
        & (np.abs(ux) > 1e-12)
        & (s > 0.0)
        & (yt_all >= yr_min)
        & (yt_all <= yr_max)
    )

    idx = np.where(good)[0]

    return {
        "yt": yt_all[idx],
        "tt": tt_all[idx],
        "dist_tr": dist_all[idx],
        "idx": idx,
        "ui": ui[idx],
        "n_good": idx.size,
        "n_total": ui.shape[0],
        "yr_min": yr_min,
        "yr_max": yr_max,
    }

def nearest_in_d(d, y):
    """
    For each value in y, return the closest element of d.

    Parameters
    ----------
    d : 1D array
        Reference grid (e.g. daxis), sorted ascending.
    y : 1D array
        Query values within the range of d.

    Returns
    -------
  dict with:
        d_nearest : 1D array, closest d value for each y
        idx       : 1D int array, index into d
        offset    : 1D array, d_nearest - y
    """
    d = np.asarray(d, float)
    y = np.asarray(y, float)

    if d.ndim != 1 or y.ndim != 1:
        raise ValueError("d and y must be 1D arrays")

    i1 = np.searchsorted(d, y)
    i1 = np.clip(i1, 1, len(d) - 1)
    i0 = i1 - 1

    pick_right = np.abs(y - d[i1]) < np.abs(y - d[i0])
    idx = np.where(pick_right, i1, i0)
    d_nearest = d[idx]

    return {
        "d_nearest": d_nearest,
        "idx": idx,
        "offset": d_nearest - y,
    }

def _tukey_gate(n_win, alpha=0.5, sym=True):
    """
    Tukey taper for a discrete time gate.
  Parameters match scipy.signal.windows.tukey:
    M     -> n_win  (number of samples inside [tbegin, tend))
    alpha -> fraction of each taper region (0 = boxcar, 1 = Hann)
    sym   -> True for finite/symmetric gate (recommended here)
    """
    n_win = int(n_win)
    if n_win <= 0:
        raise ValueError("time window is empty")
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    return tukey(n_win, alpha=alpha, sym=sym)

def windowed_das(das_obj, time_window, alpha=0.5, sym=True):
    """
    Extract, demean and taper a common time interval for all DAS traces.

    Processing:
      1) extract samples inside time_window
      2) calculate each subtrace mean
      3) demean each subtrace
      4) apply a Tukey taper
      5) place the subtraces back, with samples outside the window set to zero

    Parameters
    ----------
    das_obj : Data2D-like
        Must have .data, .taxis and .copy().
    time_window : (tbegin, tend)
        Time interval in seconds.
    alpha : float
        Tukey parameter: 0 = rectangular, 1 = Hann.
    sym : bool
        Use a symmetric Tukey window.

    Returns
    -------
    das_out : Data2D-like
        Copy containing the processed traces.
    info : dict
        Window indices, taper and removed means.
    """
    tbegin, tend = map(float, time_window)

    if tend <= tbegin:
        raise ValueError("require tend > tbegin")

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    taxis = np.asarray(das_obj.taxis, dtype=float)
    data = np.asarray(das_obj.data)

    if data.ndim != 2:
        raise ValueError("das_obj.data must be 2D")

    n_traces, n_samples = data.shape

    if taxis.ndim != 1 or taxis.size != n_samples:
        raise ValueError("das_obj.taxis must match the time dimension")

    i0 = int(np.searchsorted(taxis, tbegin, side="left"))
    i1 = int(np.searchsorted(taxis, tend, side="right"))

    # Constrain indices to the available time axis
    i0 = np.clip(i0, 0, n_samples)
    i1 = np.clip(i1, 0, n_samples)

    if i1 <= i0:
        raise ValueError("time window is empty on das_obj.taxis")
    # 1) Extract subtraces
    subtraces = data[:, i0:i1].astype(float, copy=True)

    # 2) Calculate Tukey window
    n_window = i1 - i0
    taper = _tukey_gate(n_window, alpha=alpha, sym=sym)

    taper_sum = taper.sum()
    if taper_sum <= 0:
       raise ValueError("Tukey window has zero total weight")

    # 3) Weighted mean for each subtrace
    weighted_mean = (
    np.sum(subtraces * taper[np.newaxis, :], axis=1, keepdims=True)
    / taper_sum)

   # 4) Remove weighted mean and apply taper
    subtraces = (
    subtraces - weighted_mean
    ) * taper[np.newaxis, :]

    # 5) Insert processed subtraces into a zero-filled panel
    data_out = np.zeros((n_traces, n_samples), dtype=float)
    data_out[:, i0:i1] = subtraces

    das_out = das_obj.copy()
    das_out.data = data_out

    info = {
        "i0": int(i0),
        "i1": int(i1),
        "tbegin": tbegin,
        "tend": tend,
        "n_window": n_window,
        "alpha": alpha,
        "sym": bool(sym),
        "removed_mean": weighted_mean.ravel().copy(),
        "taper": taper,
    }
    return das_out, info

def windowed_trace(trace, time_axis, time_window, alpha=0.5, sym=True, lag=0):
    """
    Like windowed_das_from_file, but for one 1D trace.

    Processing:
      1) extract samples inside time_window
      2) compute Tukey-weighted mean of the subtrace
      3) demean with that weighted mean
      4) apply Tukey taper
      5) place back on full time_axis; samples outside the window are zero
      6) optional time shift by `lag` samples (zero-padded, no wrap-around)

    Parameters
    ----------
    trace : 1D array
        Single trace samples.
    time_axis : 1D array
        Time axis [s], same length as trace.
    time_window : (tbegin, tend)
        Time gate [s].
    alpha : float
        Tukey parameter (0 = boxcar, 1 = Hann). Same as windowed_das_from_file.
    sym : bool
        Symmetric Tukey window.
    lag : int
        Time shift [samples] applied to the windowed trace, default 0 (no
        shift). Same convention as DAS_inversion.shift_zero_pad /
        np.roll(x, -lag): POSITIVE lag ADVANCES the trace (moves it to
        earlier time, out[n] = trace[n + lag]); negative lag delays it.
        Vacated samples are zero-filled. E.g. lag=20 at dt=0.1 ms advances
        the direct-wave STF by 2 ms, to reference it to the reflections'
        window start (0.193 s) instead of the direct's (0.195 s).

    Returns
    -------
    trace : dict
        "trace": windowed (and shifted) trace, same length as input;
        "time": copy of time_axis (unchanged -- only the samples move).
    info : dict
        i0, i1 (window indices BEFORE the shift), tbegin, tend, n_window,
        alpha, sym, removed_mean, taper, lag, i0_out, i1_out (window
        indices AFTER the shift, clipped to the trace)
    """
    trace = np.asarray(trace, float).ravel()
    time_axis = np.asarray(time_axis, float).ravel()

    if trace.ndim != 1:
        raise ValueError("trace must be 1D")
    if time_axis.ndim != 1:
        raise ValueError("time_axis must be 1D")
    if trace.size != time_axis.size:
        raise ValueError("trace and time_axis must have the same length")

    tbegin, tend = map(float, time_window)
    if tend <= tbegin:
        raise ValueError("require tend > tbegin")

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    n_samples = trace.size
    i0 = int(np.searchsorted(time_axis, tbegin, side="left"))
    i1 = int(np.searchsorted(time_axis, tend, side="right"))
    i0 = int(np.clip(i0, 0, n_samples))
    i1 = int(np.clip(i1, 0, n_samples))
    if i1 <= i0:
        raise ValueError("time window empty on time_axis")

    # 1) extract
    sub = trace[i0:i1].astype(float, copy=True)

    # 2) Tukey + weighted mean
    n_window = i1 - i0
    taper = _tukey_gate(n_window, alpha=alpha, sym=sym)
    taper_sum = float(taper.sum())
    if taper_sum <= 0:
        raise ValueError("Tukey window has zero total weight")

    wm = float(np.sum(sub * taper) / taper_sum)

    # 3–4) demean and taper
    sub = (sub - wm) * taper

    # 5) place back
    trace_out = np.zeros(n_samples, dtype=float)
    trace_out[i0:i1] = sub

    # 6) zero-padded shift: out[n] = trace_out[n + lag] (lag > 0 -> earlier)
    lag = int(lag)
    if abs(lag) >= n_samples:
        raise ValueError("|lag| must be smaller than the trace length")
    if lag != 0:
        shifted = np.zeros(n_samples, dtype=float)
        if lag > 0:
            shifted[:n_samples - lag] = trace_out[lag:]   # advance
        else:
            shifted[-lag:] = trace_out[:n_samples + lag]  # delay
        trace_out = shifted

    info = {
        "i0": i0,
        "i1": i1,
        "tbegin": tbegin,
        "tend": tend,
        "n_window": n_window,
        "alpha": alpha,
        "sym": bool(sym),
        "removed_mean": wm,
        "taper": taper,
        "lag": lag,
        # window position after the shift, clipped to the trace
        "i0_out": int(np.clip(i0 - lag, 0, n_samples)),
        "i1_out": int(np.clip(i1 - lag, 0, n_samples)),
    }

    trace = {
    "trace": trace_out,
    "time": time_axis.copy(),
    }
    return trace, info

def trace_FT(trace_dict, n=None):
    """
    Fourier transform of a windowed_trace() result.

    Parameters
    ----------
    trace_dict : dict
        First return of windowed_trace: keys "trace", "time".
        Also accepts (trace_dict, info) and uses only the first item.
    n : int or None
        FFT length. None -> len(trace).

    Returns
    -------
    dict with:
        trace : 1D ndarray (time-domain, unchanged)
        time  : 1D ndarray
        fft   : 1D complex ndarray
        freqs : 1D ndarray [Hz]
    """
    if isinstance(trace_dict, (tuple, list)) and len(trace_dict) >= 1:
        trace_dict = trace_dict[0]

    if not isinstance(trace_dict, dict):
        raise ValueError("trace_dict must be a dict from windowed_trace()")
    if "trace" not in trace_dict or "time" not in trace_dict:
        raise ValueError("trace_dict must contain 'trace' and 'time'")

    tr = np.asarray(trace_dict["trace"], float).ravel()
    taxis = np.asarray(trace_dict["time"], float).ravel()
    if tr.size != taxis.size:
        raise ValueError("trace and time must have the same length")
    if tr.size < 2:
        raise ValueError("need at least 2 samples for FFT")

    dt = float(np.median(np.diff(taxis)))
    if dt <= 0:
        raise ValueError("time axis must be increasing (dt > 0)")

    N = int(tr.size if n is None else n)
    Tr = fft(tr, n=N)
    freqs = fftfreq(N, d=dt)

    return {
        "trace": tr.copy(),
        "time": taxis.copy(),
        "fft": Tr,
        "freqs": freqs,
    }

def _to_3d_xz(v):
    """
    Embed 2D vectors into 3D with y=0:
        (v0, v1) -> (v0, 0, v1)
    v : (2,) or (N, 2)
    """
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        return np.array([v[0], 0.0, v[1]], dtype=float)
    return np.column_stack([v[:, 0], np.zeros(len(v)), v[:, 1]])


def _from_3d_xz(v3):
    """(x, 0, z) -> (x, z)"""
    v3 = np.asarray(v3, dtype=float)
    if v3.ndim == 1:
        return np.array([v3[0], v3[2]], dtype=float)
    return np.column_stack([v3[:, 0], v3[:, 2]])


def _polarization_from_n_u(n, u):
    """
    SV-type polarization p = (n × u) × u, matching the up-/down-going Aki &
    Richards SV convention that DAS_inversion.RPS_coef was derived under
    (RPS_coef's header: "Fracture is along x1, with normal: -x3", i.e. its
    reference normal points in the physically-UP direction -- and this
    codebase's y axis is positive-up, so that reference is (0, 1) here).

    Aki & Richards give a *different* polarization formula for a downgoing
    vs. an upgoing leg (see e.g. Quantitative Seismology Ch. 5): the two
    forms are related by flipping the sign of the reference normal used in
    (n x u) x u. A single fixed n therefore cannot be correct for two legs
    that travel in opposite vertical senses (e.g. an incident leg going down
    to a reflector and the reflected leg coming back up from it, or simply
    not knowing in advance whether a direct-arrival leg is up- or
    down-going). To stay correct for either sense with one call, n is
    re-aligned per row to the propagation direction it is paired with:

        n_row = sign(n · u) * n

    which reduces to the plain (n × u) × u formula whenever n already
    points the same general way as u, and flips to the matching
    up-/down-going form otherwise. This makes the result -- and in
    particular n_p_ref_dot / n_p_trans_dot / n_p_dir_dot downstream --
    independent of the arbitrary sign originally chosen for n, as long as
    n itself lies along the intended reference axis (the reflector's own
    normal for a reflected/transmitted leg, or the vertical axis (0, 1)
    for a direct leg with no reflector -- see stack_source_distance).

    n : (2,)
    u : (N, 2)

    Returns
    -------
    p : (N, 2)
    """
    n = np.asarray(n, dtype=float)
    u = np.asarray(u, dtype=float)

    single = u.ndim == 1
    if single:
        u = u.reshape(1, 2)

    # align n to each row's own propagation sense (see docstring)
    sign = np.where(u @ n >= 0.0, 1.0, -1.0)   # (N,)
    n_row = sign[:, None] * n[None, :]         # (N, 2)

    n_3d = _to_3d_xz(n_row)              # (N, 3): (nx, 0, ny), per row
    u3 = _to_3d_xz(u)                    # (N, 3): (ux, 0, uy)

    # p3 = (n × u) × u
    n_cross_u = np.cross(n_3d, u3)      # (N, 3), along ±y
    p3 = np.cross(n_cross_u, u3)        # (N, 3), back in x–z plane

    p = _from_3d_xz(p3)                 # (px, pz) -> 2D

    rn = np.linalg.norm(p, axis=1)
    rn = np.where(rn > 0.0, rn, np.nan)
    p = p / rn[:, None]
    return p[0] if single else p

def take_traces(das_obj, trace_ids):
    """
    Return a DAS object containing only the selected traces.

    Parameters
    ----------
    das_obj : Data2D-like
        Must have .data (n_traces, n_samples), .daxis, .copy().
    trace_ids : array-like
        Trace indices into das_obj (0-based).

    Returns
    -------
    das_out : Data2D-like
        Copy with data and daxis restricted to trace_ids.
        taxis / dt unchanged.
    """
    trace_ids = np.asarray(trace_ids, int)
    if trace_ids.ndim != 1:
        raise ValueError("trace_ids must be 1D")
    if trace_ids.size == 0:
        raise ValueError("trace_ids is empty")

    n_tr = das_obj.data.shape[0]
    if np.any(trace_ids < 0) or np.any(trace_ids >= n_tr):
        raise ValueError("trace_ids out of range")

    daxis = np.asarray(das_obj.daxis, float)

    das_out = das_obj.copy()
    das_out.data = np.asarray(das_obj.data[trace_ids, :]).copy()
    das_out.daxis = daxis[trace_ids].copy()
    return das_out

def stack_source_distance(stack_out, ny, source, xr=0.0):
    """
    Source–receiver distances for traces used in stack_traces_mean().

    Also builds the mean-depth direct ray (source -> (xr, yr_mean)) and
    fiber projections f·p, f·u with f = (0, 1).

    There is no reflector for a direct arrival, so unlike incident_reflection
    (which has an actual reflector normal to work with) there is no natural
    reference axis for the SV polarization decomposition other than the
    vertical itself: this codebase's y axis is positive-up, matching the
    reference (normal along -x3, i.e. physically up) that
    DAS_inversion.RPS_coef was derived under, so ny should stay along that
    vertical axis. Its *sign* does not actually matter: p_dir is computed
    via _polarization_from_n_u(ny, u_dir), which re-aligns ny's sign to
    u_dir's own propagation sense (up- or down-going) before use -- see
    that function's docstring -- so ny=(0,1) and ny=(0,-1) give the same
    p_dir. The parameter is kept for clarity/generality rather than
    hardcoding the axis inside the function.

    Parameters
    ----------
    stack_out : dict
        Output of stack_traces_mean (needs "depth_traces").
    ny : (2,), optional
        Vertical reference axis for the SV polarization decomposition
        (see above). Default (0, 1); any nonzero multiple of the vertical
        axis gives the same p_dir.
    source : (xs, ys)
        Source position [m].
    xr : float
        Receiver x (fibre) position [m]. Default 0.

    Returns
    -------
    dict with:
        distances            : (N,) |S -> R_i| [m]
        mean_distance_source : float, mean of distances
        depth_traces         : (N,) receiver y (from stack_out)
        yr_mean              : float, mean receiver y
        u_dir                : (2,) unit vector source -> (xr, yr_mean)
        p_dir                : (2,) polarization from u_dir (see above)
        n_p_dir_dot          : float, f · p_dir
        n_u_dir_dot          : float, f · u_dir
        source               : (xs, ys)
        xr                   : float
    """
    if "depth_traces" not in stack_out:
        raise ValueError("stack_out must contain 'depth_traces' from stack_traces_mean()")

    xs, ys = map(float, source)
    xr = float(xr)
    yr = np.asarray(stack_out["depth_traces"], float).ravel()
    if yr.size == 0:
        raise ValueError("depth_traces is empty")

    distances = np.hypot(xs - xr, ys - yr)
    mean_distance_source = float(np.mean(distances))

    # mean fibre depth for the mean-distance ray
    yr_mean = float(np.mean(yr))

    # unit ray direction: source -> (xr, yr_mean)
    dx = xr - xs
    dy = yr_mean - ys
    r = float(np.hypot(dx, dy))
    if r == 0.0:
        raise ValueError("source coincides with (xr, yr_mean); u_dir undefined")
    u_dir = np.array([dx / r, dy / r], dtype=float)

    # polarization with  normal ny = (0, -1)
    p_dir = _polarization_from_n_u(ny, u_dir)   # (2,)

    # fiber unit tangent
    f = np.array([0.0, 1.0], dtype=float)
    n_p_dir_dot = float(np.dot(f, p_dir))
    n_u_dir_dot = float(np.dot(f, u_dir))

    return {
        "distances": distances,
        "mean_distance_source": mean_distance_source,
        "depth_traces": yr.copy(),
        "yr_mean": yr_mean,
        "u_dir": u_dir,
        "p_dir": p_dir,
        "n_p_dir_dot": n_p_dir_dot,
        "n_u_dir_dot": n_u_dir_dot
    }


def tukey_right_freq_window(freqs, f_end, alpha=0.5):
    """
    Right-sided (high-frequency) Tukey window on a frequency axis.

    For |f|:
      - W = 1 for |f| <= f1
      - cosine taper for f1 < |f| < f_end
      - W = 0 for |f| >= f_end

    where f1 = f_end * (1 - alpha)

    alpha : float in [0, 1]
      0 → hard cut at f_end (boxcar)
      1 → no flat part; cosine from 0 to f_end
      (same meaning as scipy tukey, but only the high-frequency side)

    Returns
    -------
    W : ndarray, same shape as freqs (real, even in f → zero phase)
    """
    freqs = np.asarray(freqs, float)
    f_end = float(f_end)
    alpha = float(alpha)

    if f_end <= 0:
        raise ValueError("f_end must be > 0")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    f_abs = np.abs(freqs)
    W = np.zeros_like(f_abs, dtype=float)

    if alpha == 0.0:
        W[f_abs <= f_end] = 1.0
        return W

    f1 = f_end * (1.0 - alpha)          # end of flat passband
    W[f_abs <= f1] = 1.0

    mid = (f_abs > f1) & (f_abs < f_end)
    # half-cosine: 1 at f1, 0 at f_end
    W[mid] = 0.5 * (1.0 + np.cos(np.pi * (f_abs[mid] - f1) / (f_end - f1)))

    return W


def lowpass_tukey_zerophase(trace, dt, f_end, alpha=0.5):
    """
    Soft low-pass: multiply FFT by right-sided Tukey window.
    """
    x = np.asarray(trace, float)
    N = x.size
    freqs = fftfreq(N, d=dt)
    W = tukey_right_freq_window(freqs, f_end=f_end, alpha=alpha)
    y = ifft(fft(x) * W).real
    return y


def lowpass_tukey_zerophase_scat(scat_data, traces, f_end, alpha=0.5):
    """
    Purpose
    -------
    Scattering-data version of lowpass_tukey_zerophase (the
    zero-phase low-pass used for the source wavelet in DASrefData.ipynb), so
    all data share the same filter. Applied to every trace of one trace set:

        X(f)  = FFT(trace)
        W(f)  = tukey_right_freq_window(freqs, f_end, alpha)
                (1 for |f| <= f_end (1 - alpha), half-cosine to 0 at f_end,
                 0 above; real and even in f -> zero phase)
        .fft  = X(f) W(f)
        .data = real(IFFT(X(f) W(f)))

    The traces object's .fft and .data are OVERWRITTEN (make a deep copy of
    scat_data beforehand to keep the unfiltered data).

    Input
    -----
    scat_data : dict (data/scattering_data.pkl from DASscatData.ipynb)
        with the trace set `traces` (Data2D_XT: .data (n_tr, nt), .dt [s]).
        If scat_data['freqs'] exists (fft_traces_scat) it must match
        np.fft.fftfreq(nt, dt).
    traces : str
        'transmitted_traces' or 'reflection_traces'.
    f_end : float
        End of the taper [Hz]; W = 0 for |f| >= f_end (e.g. 320, as in
        DASrefData.ipynb).
    alpha : float
        Taper fraction in [0, 1]; flat passband up to f_end (1 - alpha).
        Default 0.5 (as in DASrefData.ipynb).

    Output
    ------
    scat_data : the same dict, updated in place (also returned), with
        scat_data[traces].fft  : complex (n_tr, nt), filtered spectra
        scat_data[traces].data : float (n_tr, nt), filtered traces
    """
    if traces not in ("transmitted_traces", "reflection_traces"):
        raise ValueError("traces must be 'transmitted_traces' or 'reflection_traces'")
    das = scat_data[traces]
    data = np.asarray(das.data, float)
    nt = data.shape[1]

    freqs = np.fft.fftfreq(nt, d=float(das.dt))           # frequency axis of these traces [Hz]
    if "freqs" in scat_data and not np.allclose(scat_data["freqs"], freqs):
        raise ValueError("scat_data['freqs'] does not match fftfreq(nt, dt) of the traces")

    W = tukey_right_freq_window(freqs, f_end=f_end, alpha=alpha)   # same window as lowpass_tukey_zerophase
    X = np.fft.fft(data, axis=1) * W                       # filtered spectra, all traces at once
    das.fft = X
    das.data = np.fft.ifft(X, axis=1).real                 # zero-phase filtered traces
    return scat_data


def preevent_noise(das_panel, trace_ids, noise_window):
    """
    Pre-event (ambient) noise power of selected traces.

    Purpose
    -------
    Noise level per trace, measured before the event on the UNSHIFTED
    processed panel (e.g. DAS_filter in DASrefData.ipynb), for the same
    traces later extracted from the flattened panel
    (DASrr = take_traces(DAS_SS2shifted, traces_ids)). The flattened
    traces cannot be used for this: each was shifted by a different amount
    with np.roll (circular), so a fixed pre-event window would contain event
    energy or wrapped-around samples for some traces. See
    tests/preevent_snr_notes.md for how it compares with DASref_modeling.lateral_snr.

        sigma2_k = mean(noise_k(t)^2),  t in noise_window

    Input
    -----
    das_panel : Data2D-like
        Processed, unshifted panel (same filtering as the signal); needs
        .data (n_traces, n_samples) and .taxis.
    trace_ids : array-like of int
        Row indices into das_panel (the traces_ids passed to take_traces;
        DAS_SS2shifted has the same rows as DAS_filter).
    noise_window : (t0, t1)
        Pre-event window [s] on das_panel.taxis, e.g. (0.02, 0.105): before
        the origin time, skipping the first samples, where an f-k filter
        (2D FFT) can wrap late energy.

    Output
    ------
    sigma2 : (n,) float array
        Noise power per sample for each trace, in trace_ids order.
    """
    trace_ids = np.asarray(trace_ids, int).ravel()
    data = np.asarray(das_panel.data, float)
    if np.any(trace_ids < 0) or np.any(trace_ids >= data.shape[0]):
        raise ValueError("trace_ids out of range for das_panel")
    tp = np.asarray(das_panel.taxis, float)
    nw = (tp >= noise_window[0]) & (tp <= noise_window[1])
    if not nw.any():
        raise ValueError(f"noise_window {noise_window} is empty on das_panel.taxis")
    return np.mean(data[trace_ids][:, nw] ** 2, axis=1)   # mean square in the window, per trace


def DASattributes(DASdata, lg=5.0):
    """
    Assign the standard header attributes and axes to a DAS panel
    (same as the "#DAS attributes" cell in DASrefData.ipynb).

    Input
    -----
    DASdata : Data2D_XT-like
        DAS panel with .data (nx, nt) and .attrs holding 'NewStartFile',
        'EventTime', 'StartDepth', 'EndDepth', 'Samplingrate', 'dx'.
    lg : float
        Gauge length [m] (default 5.0).

    Output
    ------
    DASdata : same object, modified in place, with start_time, name,
        attrs['Unit'], startdepth, enddepth, lg, dt, taxis [s], daxis [m].
    """
    attrs = DASdata.attrs
    DASdata.start_time = datetime.strptime(attrs['NewStartFile'], '%Y-%m-%d %H:%M:%S.%f')
    DASdata.name = attrs['EventTime']
    attrs['Unit'] = r'n$\epsilon$/s'
    DASdata.startdepth = attrs['StartDepth']
    DASdata.enddepth = attrs['EndDepth']
    DASdata.lg = lg

    nx, nt = DASdata.data.shape
    fs = attrs['Samplingrate'][0]                            # sampling rate [Hz]
    DASdata.dt = 1 / fs
    DASdata.taxis = np.arange(nt) / fs                       # time axis [s], t=0 at file start
    DASdata.daxis = attrs['StartDepth'] + np.arange(nx) * attrs['dx']   # depth axis [m]
    return DASdata


def DASpre_process(DASdata, params=None):
    """
    Pre-process a DAS record: bandpass -> Tukey taper -> f-k velocity filter
    -> (optional) median filter -> cut.

    The f-k filter is applied to a taper window larger than the cut window, tapered with a
    2D Tukey window (time x depth), so the 2D FFT edge effects stay outside the window that
    is finally cut (tests/ref_data/fk_filter_amplitude_notes.md). Defaults are the
    DASrefData.ipynb (Event A) values.

    Input
    -----
    DASdata : Data2D_XT
        Full record with attributes set (DASattributes); not modified.
    params : dict or None
        Single dictionary with any of the groups below; missing groups/keys use defaults.
        {"bp":    {"lowf": 5, "highf": 250},                 # bandpass corners [Hz]
         "taper": {"t_window": (1.6, 2.4),                    # taper/f-k window, file time [s]
                                                              # (None = no taper step: f-k on the cut window)
                   "d_window": None,                          # taper/f-k depths [m]; None = all channels
                   "alpha_t": 0.25, "alpha_z": 0.15},         # Tukey alpha (time, depth)
         "fk":    {"vmin": 1900, "vmax": 1e5, "filter": 3.0}, # accepted |f/k| [m/s], mask smoothing [bins]
         "cut":   {"t_window": (1.8, 2.2),                    # output window, file time [s]
                   "d_window": (3100, 4200)},                 # output depths [m]
         "median": {"kernel_size": None}}                     # (n_chan, n_time) median filter after
                                                              # the f-k filter; None = skip.
                                                              # (1, 5) = ref_stf_data_V2.pkl (Oct 7)
        Tukey alpha = fraction of the window inside the two cosine ramps (alpha/2 per side);
        0 = boxcar, 1 = Hann.

    Output
    ------
    DAS_filter : Data2D_XT
        Filtered panel on the cut window; same samples as select_time(*cut t_window) /
        select_depth(*cut d_window) of the record, taxis starts at 0 (= cut tbegin).
    """
    defaults = {"bp": {"lowf": 5, "highf": 250},
                "taper": {"t_window": (1.6, 2.4), "d_window": None, "alpha_t": 0.25, "alpha_z": 0.15},
                "fk": {"vmin": 1900, "vmax": 1e5, "filter": 3.0},
                "cut": {"t_window": (1.8, 2.2), "d_window": (3100, 4200)},
                "median": {"kernel_size": None}}
    params = params or {}
    no_taper = "taper" in params and params["taper"] is None        # skip the taper step
    if no_taper:                                                    # f-k window = cut window, Tukey weights = 1
        params = {**params, "taper": {"t_window": params.get("cut", {}).get("t_window", defaults["cut"]["t_window"]),
                                      "d_window": params.get("cut", {}).get("d_window", defaults["cut"]["d_window"]),
                                      "alpha_t": 0.0, "alpha_z": 0.0}}
    p = {g: {**d, **params.get(g, {})} for g, d in defaults.items()}   # merge user values per group
    bp, tp, fk, cut = p["bp"], p["taper"], p["fk"], p["cut"]
    (ttb, tte), (ctb, cte) = tp["t_window"], cut["t_window"]
    (cdb, cde) = cut["d_window"]
    if not (ttb <= ctb and tte >= cte):
        raise ValueError("taper t_window must contain cut t_window")
    if tp["d_window"] is not None and not (tp["d_window"][0] <= cdb and tp["d_window"][1] >= cde):
        raise ValueError("taper d_window must contain cut d_window")

    # 1) bandpass on the full record (filter transients stay away from the window)
    panel = DASdata.copy()
    panel.bp_filter(bp["lowf"], bp["highf"])

    # 2) taper window, then 2D Tukey taper on its edges
    panel = panel.select_time(ttb, tte, makecopy=True)              # taxis reset to 0 at ttb
    if tp["d_window"] is not None:
        panel = panel.select_depth(*tp["d_window"])
    if not no_taper:
        wz = tukey(panel.data.shape[0], tp["alpha_z"])             # depth taper (one weight per trace)
        wt = tukey(panel.data.shape[1], tp["alpha_t"])             # time taper (same for every trace)
        panel.data = panel.data * wz[:, None] * wt[None, :]

    # 3) f-k velocity filter (accept vmin <= |f/k| <= vmax, Gaussian-smoothed mask)
    ProcessUtil.Data2D_fft2(panel)
    mask = ProcessUtil.Data2D_fkfilter_maskgen2(panel, fk["vmin"], fk["vmax"], fk["filter"], accept=True)
    DAS_filter = ProcessUtil.Data2D_fkfilter_applymask(panel, mask)
    del DAS_filter.fftdata      # spectrum of the taper window (~0.5 GB); stale after the cut and
                                # copied into every panel/pickle derived from DAS_filter

    # 3b) optional median filter (jin_pylib Data2D_XT.median_filter = scipy medfilt2d), on the
    #     taper window so its edge samples stay outside the cut
    if p["median"]["kernel_size"] is not None:
        DAS_filter.median_filter(kernel_size=tuple(p["median"]["kernel_size"]))

    # 4) cut: same samples as select_time(ctb, cte) on the file axis, expressed on the
    #    taper-window axis (half-sample bounds avoid float round-off)
    tf = DASdata.taxis
    t_cut = tf[(tf >= ctb) & (tf < cte)]
    t0 = tf[tf >= ttb][0]                                           # file time of taper-window sample 0
    hdt = 0.5 * DASdata.dt
    DAS_filter = DAS_filter.select_time(t_cut[0] - t0 - hdt, t_cut[-1] - t0 + hdt, makecopy=True)
    return DAS_filter.select_depth(cdb, cde)


# ---------------------------------------------------------------------------
# DASrefData workflow functions (reflected S + source wavelet), called from
# DASrefData_test.ipynb. Each one groups a block of DASrefData.ipynb cells.
# Parameters are a single dictionary; missing keys take the defaults (Event A).
# ---------------------------------------------------------------------------

def _merge_params(defaults, params):
    """
    Merge user parameters into defaults, one level of nested dictionaries deep.

    Input : defaults (dict), params (dict or None)
    Output: new dict; a key holding a dict in both is merged, otherwise params wins.
    """
    params = params or {}
    out = {}
    for k, v in defaults.items():
        u = params.get(k, None)
        if isinstance(v, dict) and isinstance(u, dict):
            out[k] = {**v, **u}                     # merge one nested group
        else:
            out[k] = v if k not in params else u
    for k in params.keys() - defaults.keys():       # extra user keys pass through
        out[k] = params[k]
    return out


def reflection_tt_flatten(DAS_filter, event, params=None):
    """
    Reflected SS traveltimes on the fiber, snapped to the positive peak, and the
    flattened panel (DASrefData.ipynb "Initial reflection travel time" + "Move the tt
    picks to the max positive peak" cells).

    Steps: fiber_reflection_tt (initial line) -> flatten on SStt -> snap_sstt_to_positive_peak
           -> roll each branch trace by its peak shift.
    The notebook flattened a second time on the snapped picks; both use whole-sample np.roll,
    so on the branch traces the result is identical (tests/ref_data/test_flatten_once.py).
    Off-branch traces keep the first flattening (the notebook moved them by the end-pick shift).

    Input
    -----
    DAS_filter : Data2D_XT
        Pre-processed panel (DASpre_process output).
    event : dict
        {"vs": S velocity [m/s], "xs", "ys": source position [m], "ts": origin time [s, file],
         "t_window": (tfile_begin, tfile_end) cut window of DAS_filter [s, file]}
    params : dict or None
        {"fracture": {"m": 0.28, "b": 3610,          # initial fracture line y = m x + b
                      "x_range": (12, 67), "step": 0.1},   # fracture x samples [m]
         "t_ref": 0.2,                               # flatten reference time [s, panel]
         "peak_gate": (0.198, 0.206)}                # positive-peak search window [s, flat panel]

    Output
    ------
    dict: SStt (initial times, len(daxis)), SStt_peak (snapped), trace_range (branch trace ids),
          id_begin, id_end, DAS_flat (panel flattened on SStt_peak), sstt_out, params
    """
    defaults = {"fracture": {"m": 0.28, "b": 3610, "x_range": (12, 67), "step": 0.1},
                "t_ref": 0.2, "peak_gate": (0.198, 0.206)}
    p = _merge_params(defaults, params)
    fr, t_ref = p["fracture"], p["t_ref"]

    # initial SS traveltime from the assumed fracture line, on the panel time axis
    xf = np.arange(fr["x_range"][0], fr["x_range"][1] + fr["step"], fr["step"])
    source = (event["ts"], event["xs"], event["ys"])
    sstt_out = fiber_reflection_tt(source, (fr["m"], fr["b"]), xf, DAS_filter.daxis,
                                   event["vs"], event["t_window"][0])
    SStt = sstt_out["SStt"]
    id_begin, id_end = sstt_out["begin_id"], sstt_out["end_id"]

    # flatten on the model times, then find the max positive peak of each branch trace
    DAS_flat0, _ = flatten_s_arrivals(DAS_filter, SStt, t_ref, interpolate_picks=True)
    snap = snap_sstt_to_positive_peak(DAS_flat0, SStt, (id_begin, id_end),
                                      (t_ref, *p["peak_gate"]))
    SStt_peak = snap["modified_SStt"]

    # move the peaks to t_ref: roll each branch trace by its peak shift (whole samples)
    trace_range = np.arange(id_begin, id_end + 1)
    lag = np.rint(snap["time_shift"][trace_range] / DAS_filter.dt).astype(int)
    DAS_flat = apply_cc_lags(DAS_flat0, trace_range, lag)

    return {"SStt": SStt, "SStt_peak": SStt_peak,
            "trace_range": trace_range,
            "id_begin": id_begin, "id_end": id_end,
            "DAS_flat": DAS_flat, "sstt_out": sstt_out, "params": p}


def cc_pilot_align(DAS_flat, params=None):
    """
    Pilot wavelet by cc-aligned stacking of a few traces, then (optional) cc of every trace
    in a depth gate against the pilot and alignment by the cc lags. Same procedure as the
    positive-lobe, negative-lobe and direct-S cc cells of DASrefData.ipynb.

    Input
    -----
    DAS_flat : Data2D_XT
        Flattened panel.
    params : dict or None
        {"pilot": {"d_gate": (3690, 3710),   # traces stacked into the pilot [m]
                   "dref": 3700,             # reference trace of the pilot gate [m]
                   "tref": 0.2,              # centre of the pilot cc window [s]
                   "tw": 0.008},             # length of the pilot cc window [s]
         "align": {"d_gate": None,           # traces cc'd with the pilot [m]; None = all traces
                   "tref": 0.2,              # time assigned to zero lag [s]
                   "t_gate": (0.197, 0.203)},# cc window [s]
                  or None to return the pilot only
         "stack_method": "mean"}

    Output
    ------
    dict: stack (stack_traces_mean output: trace_stack, time, depth_traces, ...),
          cc_ref (pilot-gate cc), cc_all (cc vs pilot: lag_samples, cc_peak, depth_m, ...)
          or None, DAS_shifted (DAS_flat aligned by cc_all lags) or None, params
    """
    defaults = {"pilot": {"d_gate": (3690, 3710), "dref": 3700, "tref": 0.2, "tw": 0.008},
                "align": {"d_gate": None, "tref": 0.2, "t_gate": (0.197, 0.203)},
                "stack_method": "mean"}
    p = _merge_params(defaults, params)
    pl = p["pilot"]

    # 1) pilot: cc-align the traces in the pilot gate and stack them
    cc_ref = cc_lags_in_depth_gate(DAS_flat, pl["d_gate"][0], pl["d_gate"][1],
                                   pl["dref"], pl["tref"], pl["tw"])
    DAS_pilot = apply_cc_lags_from_gate(DAS_flat, cc_ref)
    stack = stack_traces_mean(DAS_pilot, cc_ref["trace_i"], method=p["stack_method"])

    out = {"stack": stack, "cc_ref": cc_ref, "cc_all": None, "DAS_shifted": None, "params": p}
    if p["align"] is None:                       # pilot only (e.g. source wavelet)
        return out

    # 2) cc every trace in the gate with the pilot and align them
    al = p["align"]
    d_gate = al["d_gate"] or (DAS_flat.daxis[0], DAS_flat.daxis[-1])
    cc_all = cc_lags_vs_pilot_tgate(DAS_flat, stack["trace_stack"], d_gate[0], d_gate[1],
                                    al["tref"], al["t_gate"][0], al["t_gate"][1])
    out["cc_all"] = cc_all
    out["DAS_shifted"] = apply_cc_lags_from_gate(DAS_flat, cc_all)
    return out


def cc_pilot_align_tgate(DAS_flat, params=None):
    """
    Same as cc_pilot_align, but the pilot cc window is a time gate (tb, tend), given
    the same way as the cc window of all traces ("align" t_gate), instead of a
    centre + length (tref, tw).

    Input
    -----
    DAS_flat : Data2D_XT
        Flattened panel.
    params : dict or None
        {"pilot": {"d_gate": (3690, 3710),   # traces stacked into the pilot [m]
                   "dref": 3700,             # reference trace of the pilot gate [m]
                   "t_gate": (0.196, 0.204)},# pilot cc window [s]
         "align": {"d_gate": None,           # traces cc'd with the pilot [m]; None = all traces
                   "tref": 0.2,              # time assigned to zero lag [s]
                   "t_gate": (0.197, 0.203)},# cc window of all traces [s]
                  or None to return the pilot only
         "stack_method": "mean"}

    Output
    ------
    dict: as cc_pilot_align (stack, cc_ref, cc_all, DAS_shifted), params (gate form)
    """
    defaults = {"pilot": {"d_gate": (3690, 3710), "dref": 3700, "t_gate": (0.196, 0.204)},
                "align": {"d_gate": None, "tref": 0.2, "t_gate": (0.197, 0.203)},
                "stack_method": "mean"}
    p = _merge_params(defaults, params)
    tb, tend = p["pilot"]["t_gate"]

    # gate -> centre + length used by cc_pilot_align (window = [tb, tend])
    pilot = {"d_gate": p["pilot"]["d_gate"], "dref": p["pilot"]["dref"],
             "tref": 0.5 * (tb + tend), "tw": tend - tb}
    out = cc_pilot_align(DAS_flat, {"pilot": pilot, "align": p["align"],
                                    "stack_method": p["stack_method"]})
    out["params"] = p
    return out


def fit_reflector(SStt_pick, drange, event, params=None):
    """
    Best fracture line from the picked reflection times, and the reflection geometry
    (DASrefData.ipynb "Fit a curve" + "Find the best fracture line" cells).

    Steps: polynomial fit of the picks -> times from origin -> estimate_best_line
           -> incident_reflection (rays and angles) -> best-line times on the panel axis.

    Input
    -----
    SStt_pick : (n,) picked SS times on the panel axis [s], one per trace of drange
    drange : (n,) depths of those traces [m]
    event : dict (see reflection_tt_flatten)
    params : dict or None
        {"line0": (0.28, 3610),   # starting fracture line (m, b)
         "poly_deg": 2,           # degree of the fit to the picks
         "d_fit": None,           # (dmin, dmax) depths used in the fit [m]; None = all
         "xr": 0.0}               # fiber x position [m]

    Output
    ------
    dict: coeff, SStt_fit (fit on drange, panel axis), best_line_out, angles_rays,
          tt_fit_fracture ({'tt_best_das', 'depth_best'} for DAS_plots), params
    """
    defaults = {"line0": (0.28, 3610), "poly_deg": 2, "d_fit": None, "xr": 0.0}
    p = _merge_params(defaults, params)
    drange = np.asarray(drange, float)
    SStt_pick = np.asarray(SStt_pick, float)

    # fit a smooth curve to the picks (optionally on a depth sub-range)
    m = np.ones(drange.size, bool)
    if p["d_fit"] is not None:
        m = (drange >= p["d_fit"][0]) & (drange <= p["d_fit"][1])
    coeff = np.polyfit(drange[m], SStt_pick[m], p["poly_deg"])
    SStt_fit = np.polyval(coeff, drange)

    # traveltime from origin time, then best line by ray tracing
    t0, ts = event["t_window"][0], event["ts"]
    source = (event["xs"], event["ys"])
    best_line_out = estimate_best_line(source, drange, p["line0"], event["vs"],
                                       SStt_fit + t0 - ts)
    angles_rays = incident_reflection(source, drange, best_line_out["xp"], best_line_out["yp"],
                                      best_line_out["line"][0], p["xr"])

    # best-line times back on the panel axis (for plots)
    tt_best_das = np.asarray(best_line_out["tt"], float) + ts - t0
    return {"coeff": coeff, "SStt_fit": SStt_fit, "best_line_out": best_line_out,
            "angles_rays": angles_rays,
            "tt_fit_fracture": {"tt_best_das": tt_best_das, "depth_best": drange},
            "params": p}


def extract_reflection_traces(DAS_flat, DAS_shifted, trace_range, cc_pos, params=None):
    """
    Extract all reflected branch traces and compute the max cc of the negative lobe
    (replaces the negative-lobe cc_pilot_align call + select_reflection_traces).
    No cc selection: every branch trace is kept, and cc_pos / cc_neg are stored so a
    selection can be made later.

    Input
    -----
    DAS_flat : Data2D_XT, flattened panel (negative-lobe cc is measured here)
    DAS_shifted : Data2D_XT, flattened and positive-lobe cc-aligned panel (traces taken here)
    trace_range : (n,) branch trace ids (reflection_tt_flatten "trace_range")
    cc_pos : (n,) max cc of the positive lobe, one per trace_range entry
    params : dict or None, negative-lobe cc (cc_pilot_align_tgate parameters)
        {"pilot": {"d_gate": (3690, 3710),   # traces stacked into the pilot [m]
                   "dref": 3700,             # reference trace of the pilot gate [m]
                   "t_gate": (0.201, 0.209)},# pilot cc window [s]
         "align": {"tref": 0.205,            # time assigned to zero lag [s]
                   "t_gate": (0.201, 0.209)}}# cc window of the branch traces [s]

    Output
    ------
    dict: trace_range, trace_ids (= trace_range), idx (= all positions), DASrr_all and
          DASrr (both = all branch traces; kept so window_reflection_traces works),
          cc_pos, cc_neg (max cc of each lobe, one per trace_range entry),
          neg (cc_pilot_align_tgate output of the negative lobe), params
    """
    defaults = {"pilot": {"d_gate": (3690, 3710), "dref": 3700, "t_gate": (0.201, 0.209)},
                "align": {"tref": 0.205, "t_gate": (0.201, 0.209)}}
    p = _merge_params(defaults, params)
    trace_range = np.asarray(trace_range, int)

    # negative-lobe cc of every branch trace (depth gate = first/last branch trace)
    d_branch = (DAS_flat.daxis[trace_range[0]], DAS_flat.daxis[trace_range[-1]])
    neg = cc_pilot_align_tgate(DAS_flat, {"pilot": p["pilot"],
                                          "align": {**p["align"], "d_gate": d_branch}})

    # all branch traces from the positive-lobe aligned panel
    DASrr_all = take_traces(DAS_shifted, trace_range)
    return {"trace_range": trace_range, "trace_ids": trace_range,
            "idx": np.arange(trace_range.size),
            "DASrr_all": DASrr_all, "DASrr": DASrr_all,
            "cc_pos": np.asarray(cc_pos, float),
            "cc_neg": np.asarray(neg["cc_all"]["cc_peak"], float),
            "neg": neg, "params": p}


def window_all_reflection_traces(DAS_filter, sel, params=None):
    """
    Tukey-window all reflected branch traces and measure their pre-event noise.
    Same as window_reflection_traces, for extract_reflection_traces output (no trace
    selection), so a single noise array is returned.

    Input
    -----
    DAS_filter : Data2D_XT, unshifted pre-processed panel (noise is measured here)
    sel : extract_reflection_traces output (needs DASrr_all, trace_range)
    params : dict or None
        {"time_window": (0.193, 0.218),   # reflection window [s, flat panel]
         "alpha": 0.4,                    # Tukey alpha of the window
         "noise_window": (0.02, 0.105)}   # pre-event noise window [s, DAS_filter]

    Output
    ------
    dict: DASrr_w (windowed branch traces), w_info (windowed_das info),
          noise (noise power of each branch trace), params
    """
    defaults = {"time_window": (0.193, 0.218), "alpha": 0.4, "noise_window": (0.02, 0.105)}
    p = _merge_params(defaults, params)

    DASrr_w, w_info = windowed_das(sel["DASrr_all"], p["time_window"], p["alpha"])
    # noise on the unshifted panel (see window_reflection_traces)
    noise = preevent_noise(DAS_filter, sel["trace_range"], p["noise_window"])
    return {"DASrr_w": DASrr_w, "w_info": w_info, "noise": noise, "params": p}


def build_ref_dict(DASrr, trace_ids, cc_pos, cc_neg, fit_out, noise, time_window,
                   idx=None, extra=None):
    """
    Reflected-data dictionary exported for the inversions (ref_data_all / ref_data of
    DASrefData.ipynb; both have the same keys, ref_data is ref_data_all on idx).

    Input
    -----
    DASrr : Data2D_XT, reflected traces (already subset to idx)
    trace_ids : (m,) their trace ids on DAS_filter
    cc_pos, cc_neg : (n,) max cc of positive / negative lobe on all branch traces;
                     cc_neg may be None (stored as None)
    fit_out : fit_reflector output (angles_rays, best_line_out on all branch traces)
    noise : (m,) pre-event noise power of DASrr traces
    time_window : (t0, t1) reflection window [s]
    idx : (m,) positions in the branch of the exported traces; None = all
    extra : dict of extra keys to add (e.g. {'fracture_m_b': ...})

    Output
    ------
    dict with 'reflection_traces', 'trace_ids', 'max_cc_positive', 'max_cc_negative',
    'ref_points', angles/unit vectors/dot products from incident_reflection, distances,
    'time_window', 'noise' (+ extra)
    """
    ar, bl = fit_out["angles_rays"], fit_out["best_line_out"]
    sel = slice(None) if idx is None else np.asarray(idx, int)

    def take(a):                                  # per-trace array -> exported traces
        return None if a is None else np.asarray(a)[sel]

    rr_points = np.column_stack((bl["xp"], bl["yp"]))   # reflection points on the fracture
    out = {"reflection_traces": DASrr,
           "trace_ids": np.asarray(trace_ids),
           "max_cc_positive": take(cc_pos),
           "max_cc_negative": take(cc_neg),
           "ref_points": take(rr_points)}
    for k in ("theta_i_deg", "theta_r_deg", "theta_t_deg", "u_inc", "u_ref", "p_ref",
              "u_trans", "p_trans", "n_p_ref_dot", "n_u_ref_dot", "n_p_trans_dot",
              "n_u_trans_dot"):
        out[k] = take(ar[k])
    out["normal"] = ar["normal"]                  # one fracture normal, not per trace
    out["dist_source_rr"] = take(ar["source_rr"])     # source -> reflection arrival
    out["dist_source_pinc"] = take(ar["source_pinc"]) # source -> reflection point
    out["dist_pinc_rr"] = take(ar["pinc_rr"])         # reflection point -> arrival
    out["time_window"] = tuple(time_window)
    out["noise"] = np.asarray(noise)
    out.update(extra or {})
    return out


def extract_stf(DAS_filter, pick_file, event, params=None):
    """
    Source time function from the direct S (DASrefData.ipynb "S-Direct trace" cells).

    Steps: load S picks -> flatten on them -> cc-aligned stack of a few traces (pilot)
           -> Tukey window -> polarity correction -> Fourier transform -> source distance.

    Input
    -----
    DAS_filter : Data2D_XT, pre-processed panel
    pick_file : str, pick file (e.g. './data/pickfiles/pickfile_EventA.csv')
    event : dict (see reflection_tt_flatten)
    params : dict or None
        {"t_ref": 0.2,                                   # flatten reference time [s]
         "pilot": {"d_gate": (3670, 3680), "dref": 3675, "tref": 0.2, "tw": 0.01},
         "window": (0.193, 0.214), "alpha": 0.3, "lag": 0.0,   # windowed_trace inputs
         "polarity": -1,         # empirical sign (tests/direct_wave_polarity_investigation_notes.md)
         "ny": (0.0, 1.0)}       # vertical reference axis for stack_source_distance

    Output
    ------
    dict: stf_data (dict exported with the reflection data), stack (stack_traces_mean
          output), stf_trace (windowed trace + FT), DAS_flat, cc_ref, params
    """
    defaults = {"t_ref": 0.2,
                "pilot": {"d_gate": (3670, 3680), "dref": 3675, "tref": 0.2, "tw": 0.01},
                "window": (0.193, 0.214), "alpha": 0.3, "lag": 0.0, "polarity": -1,
                "ny": (0.0, 1.0)}
    p = _merge_params(defaults, params)

    # flatten the direct S on the picks
    picks = load_picks_for_window(pick_file, event["t_window"][0], event["t_window"][1],
                                  DAS_filter.daxis)
    DAS_flat, _ = flatten_s_arrivals(DAS_filter, picks["S_on_trace"], p["t_ref"],
                                     interpolate_picks=True)

    # pilot stack = estimated wavelet
    pil = cc_pilot_align(DAS_flat, {"pilot": p["pilot"], "align": None})
    stack = pil["stack"]

    # window, polarity, spectrum
    tr_w, _ = windowed_trace(stack["trace_stack"], stack["time"], p["window"], p["alpha"],
                             sym=True, lag=p["lag"])
    tr_w["trace"] *= p["polarity"]
    tr_ft = trace_FT(tr_w)

    # mean source distance and polarization projections of the stacked traces
    dist = stack_source_distance(stack, np.asarray(p["ny"], float), (event["xs"], event["ys"]))
    stf_data = {"stf_trace": tr_ft,
                "mean_dist_source": dist["mean_distance_source"],
                "u_dir": dist["u_dir"], "p_dir": dist["p_dir"],
                "n_p_dir_dot": dist["n_p_dir_dot"], "n_u_dir_dot": dist["n_u_dir_dot"]}
    return {"stf_data": stf_data, "stack": stack, "stf_trace": tr_ft,
            "DAS_flat": DAS_flat, "cc_ref": pil["cc_ref"], "params": p}


def extract_stf_tgate(DAS_filter, pick_file, event, params=None):
    """
    Same as extract_stf, but the pilot cc window is a time gate (tb, tend), as in
    cc_pilot_align_tgate, instead of a centre + length (tref, tw).

    Input
    -----
    DAS_filter, pick_file, event : as extract_stf
    params : dict or None, as extract_stf except the pilot:
        "pilot": {"d_gate": (3670, 3680),   # traces stacked into the pilot [m]
                  "dref": 3675,             # reference trace of the pilot gate [m]
                  "t_gate": (0.195, 0.205)} # pilot cc window [s]

    Output
    ------
    dict: as extract_stf, params (gate form)
    """
    defaults = {"t_ref": 0.2,
                "pilot": {"d_gate": (3670, 3680), "dref": 3675, "t_gate": (0.195, 0.205)},
                "window": (0.193, 0.214), "alpha": 0.3, "lag": 0.0, "polarity": -1,
                "ny": (0.0, 1.0)}
    p = _merge_params(defaults, params)
    tb, tend = p["pilot"]["t_gate"]

    # gate -> centre + length used by extract_stf (window = [tb, tend])
    pilot = {"d_gate": p["pilot"]["d_gate"], "dref": p["pilot"]["dref"],
             "tref": 0.5 * (tb + tend), "tw": tend - tb}
    out = extract_stf(DAS_filter, pick_file, event, {**p, "pilot": pilot})
    out["params"] = p
    return out


def plot_wiggle_zoom(das, xlim, ylim, params=None, curves=None):
    """
    Wiggle plot of a panel zoomed to a depth-time box, with optional curves on top
    (replaces the repeated zoomed-wiggle cells of DASrefData.ipynb).

    Input
    -----
    das : Data2D_XT
    xlim : (dmin, dmax) depth [m]
    ylim : (t_bottom, t_top) time [s]; give it inverted for time increasing down
    params : {"scale": 0.1, "trace_step": 1, "linewidth": 0.5, "figsize": (8, 5), "xlabel": "depth [m]"}
    curves : list of (x, y, kwargs) drawn with plt.plot, e.g. [(drange, SStt_fit, {"color": "g"})]

    Output
    ------
    fig, ax
    """
    p = _merge_params({"scale": 0.1, "trace_step": 1, "linewidth": 0.5, "figsize": (8, 5),
                       "xlabel": "depth [m]"}, params)
    plt.close("all")
    fig = plt.figure(figsize=p["figsize"])
    das.plot_wiggle(scale=p["scale"], trace_step=p["trace_step"], linewidth=p["linewidth"])
    for x, y, kw in (curves or []):
        plt.plot(x, y, **kw)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel(p["xlabel"])     # set before plt.show(), so the label is in the displayed figure
    plt.ylabel("time [s]")
    ax = plt.gca()   # take the axes before plt.show(): the inline backend closes the figure, and a
                     # later plt.gca() would create (and return) a new empty figure
    plt.show()
    return fig, ax


def plot_trace_overlay_grid(DASrr_w, DASrr, idx, angles=None, figsize=(8, 8),
                            xlim=(0.185, 0.225), title_fs=8, suptitle_fs=8, show=True):
    """
    Original vs windowed traces, 10 consecutive traces in a 5 x 2 grid
    (moved from DASrefData.ipynb). Left column idx..idx+4, right idx+5..idx+9.

    Input : DASrr_w, DASrr (Data2D_XT, same traces), idx (first trace), angles ((n,) or None,
            shown in titles), figsize, xlim [s], font sizes, show (plt.show or not)
    Output: fig, axes
    """
    taxis, daxis = DASrr_w.taxis, DASrr_w.daxis
    n_tr = DASrr_w.data.shape[0]
    ids = np.arange(int(idx), int(idx) + 10)
    if ids[-1] >= n_tr or ids[0] < 0:
        raise ValueError(f"need indices {ids[0]}..{ids[-1]} within [0, {n_tr})")

    fig, axes = plt.subplots(5, 2, figsize=figsize, sharex=True, sharey=True)
    for k, ax in zip(ids, axes.T.ravel()):        # column-major: left column first
        ax.plot(taxis, DASrr.data[k], label="original", lw=0.8)
        ax.plot(taxis, DASrr_w.data[k], label="windowed", lw=0.8)
        title = f"idx={k}, depth={daxis[k]:.1f} m"
        if angles is not None:
            title += f", angle={float(angles[k]):.1f} deg"
        ax.set_title(title, fontsize=title_fs)
        ax.set_xlim(*xlim)
    axes[0, 0].legend(loc="upper right", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("amplitude")
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.suptitle(f"indices {ids[0]}–{ids[-1]}", y=1.01, fontsize=suptitle_fs)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def save_overlay_grids(DASrr_w, DASrr, angles=None, path="./plots/reflection_w{start}_new.png",
                       dpi=150):
    """
    Save plot_trace_overlay_grid for every block of 10 traces without drawing them on screen.

    Input : DASrr_w, DASrr, angles (see plot_trace_overlay_grid), path (with {start}), dpi
    Output: list of files written
    """
    import matplotlib
    prev = matplotlib.get_backend()
    matplotlib.use("Agg")                         # draw off-screen only
    files = []
    try:
        for start in range(0, DASrr_w.data.shape[0] - 9, 10):
            fig, _ = plot_trace_overlay_grid(DASrr_w, DASrr, start, angles=angles, show=False)
            files.append(path.format(start=start))
            fig.savefig(files[-1], dpi=dpi, bbox_inches="tight")
            plt.close(fig)
    finally:
        matplotlib.use(prev)
    return files


# ---------------------------------------------------------------------------
# DASscatData workflow functions (transmitted S paired with reflected S), called from
# DASscatData_test.ipynb. Each one groups a block of DASscatData.ipynb cells.
# Parameters are a single dictionary; missing keys take the defaults (Event A).
# ---------------------------------------------------------------------------

def transmission_receivers(DAS_filter, ref_data, event):
    """
    Step 1: straight transmitted ray of every reflected trace (same incident direction u_inc)
    and the DAS channel where it arrives.

    Input
    -----
    DAS_filter : Data2D_XT, pre-processed panel
    ref_data : dict, reflected data of all branch traces (ref_data_all pkl)
    event : dict, {"vs": S velocity [m/s], "xs", "ys": source position [m]}

    Output
    ------
    dict: transmission (transmission() output), idx_tr (ref_data rows with a valid
          transmitted ray), idx_tr_das (transmission channels on DAS_filter),
          idx_rr_das (paired reflection channels on DAS_filter)
    """
    daxis = DAS_filter.daxis
    trans = transmission((event["xs"], event["ys"]), ref_data["u_inc"], event["vs"], daxis)
    idx_tr = trans["idx"]                                      # rays that hit the fiber
    idx_tr_das = nearest_in_d(daxis, trans["yt"])["idx"]       # nearest channel to each arrival
    idx_rr_das = np.asarray(ref_data["trace_ids"])[idx_tr]     # paired reflection channels
    return {"transmission": trans, "idx_tr": idx_tr,
            "idx_tr_das": idx_tr_das, "idx_rr_das": idx_rr_das}


def transmission_tt_flatten(DAS_filter, pick_file, event, params=None):
    """
    Step 2: polynomial fit of the direct-S picks in a depth gate (transmitted S
    traveltime) and the panel flattened on that fit.

    Input
    -----
    DAS_filter : Data2D_XT, pre-processed panel
    pick_file : str, pick file (e.g. './data/pickfiles/pickfile_EventA.csv')
    event : dict, {"t_window": (tfile_begin, tfile_end)} cut window of DAS_filter [s, file]
    params : dict or None
        {"d_fit": (3480, 3632),   # depth gate of the fit [m]
         "poly_deg": 2,           # degree of the fit
         "t_ref": 0.2}            # flatten reference time [s]

    Output
    ------
    dict: S_pick (picks on every channel), SStr_fit_all (fit, NaN outside the gate),
          depth_gate, SStr_fit (fit in the gate), coeff, DAS_flat,
          transmission_tt (dict to save as transmission_tt pkl), params
    """
    defaults = {"d_fit": (3480, 3632), "poly_deg": 2, "t_ref": 0.2}
    p = _merge_params(defaults, params)
    daxis = DAS_filter.daxis

    picks = load_picks_for_window(pick_file, event["t_window"][0], event["t_window"][1], daxis)
    S_pick = picks["S_on_trace"]

    # fit the picks inside the depth gate
    idx = np.where((daxis >= p["d_fit"][0]) & (daxis <= p["d_fit"][1]))[0]
    coeff = np.polyfit(daxis[idx], S_pick[idx], p["poly_deg"])
    SStr_fit = np.polyval(coeff, daxis[idx])
    SStr_fit_all = np.full(len(S_pick), np.nan)               # NaN outside the gate
    SStr_fit_all[idx] = SStr_fit

    DAS_flat, _ = flatten_s_arrivals(DAS_filter, SStr_fit_all, p["t_ref"], interpolate_picks=True)

    transmission_tt = {"daxis": daxis,                        # depth of every channel [m]
                       "SStr_fit": SStr_fit_all,              # fitted time [s], NaN outside gate
                       "S_pick": S_pick,                      # original S picks [s]
                       "coeff": coeff,                        # fit coefficients
                       "depth_gate": tuple(p["d_fit"]),       # fit gate [m]
                       "tfile_begin": event["t_window"][0],   # file time = panel time + tfile_begin
                       "t_ref": p["t_ref"]}                   # flattening reference time [s]
    return {"S_pick": S_pick, "SStr_fit_all": SStr_fit_all, "depth_gate": daxis[idx],
            "SStr_fit": SStr_fit, "coeff": coeff, "DAS_flat": DAS_flat,
            "transmission_tt": transmission_tt, "params": p}


def transmission_cc_align(DAS_flat, params=None):
    """
    Step 3: pilot = cc-aligned stack of a few flattened transmitted traces; cc of every
    trace in the fit gate with the pilot (|cc| peak) and alignment by the cc lags.
    Same window for the pilot and for the traces, given as a time gate (tb, tend).

    Input
    -----
    DAS_flat : Data2D_XT, panel flattened on the transmitted S (transmission_tt_flatten)
    params : dict or None
        {"pilot": {"d_gate": (3587, 3610),   # traces stacked into the pilot [m]
                   "dref": 3600},            # reference trace of the pilot gate [m]
         "d_gate": (3480, 3632),             # traces cc'd with the pilot [m]
         "t_gate": (0.195, 0.21),            # cc window [s]
         "stack_method": "mean"}

    Output
    ------
    dict: cc_ref (pilot-gate cc), stack (pilot trace), time (its time axis),
          cc_all (cc vs pilot: trace_i, lag_samples, cc_peak, ...), DAS_shifted, params
    """
    defaults = {"pilot": {"d_gate": (3587, 3610), "dref": 3600},
                "d_gate": (3480, 3632), "t_gate": (0.195, 0.21), "stack_method": "mean"}
    p = _merge_params(defaults, params)
    tb, tend = p["t_gate"]
    tref, tw = 0.5 * (tb + tend), tend - tb                 # gate -> centre + length
    pl = p["pilot"]

    # pilot: cc-align the traces in the pilot gate and stack them
    cc_ref = cc_lags_in_depth_gate(DAS_flat, pl["d_gate"][0], pl["d_gate"][1], pl["dref"], tref, tw)
    DAS_pilot = apply_cc_lags_from_gate(DAS_flat, cc_ref)
    stack, time = stack_traces(DAS_pilot, cc_ref["trace_i"], method=p["stack_method"])

    # cc every trace in the gate with the pilot and align them
    cc_all = cc_lags_vs_pilot(DAS_flat, stack, p["d_gate"][0], p["d_gate"][1], tref, tw)
    DAS_shifted = apply_cc_lags_from_gate(DAS_flat, cc_all)
    return {"cc_ref": cc_ref, "stack": stack, "time": time, "cc_all": cc_all,
            "DAS_shifted": DAS_shifted, "params": p}


def pair_scattering_traces(DAS_shifted, ref_data, rec, cc_all):
    """
    Step 4: keep the transmitted/reflected pairs whose transmission channel is one of the
    cc-aligned transmitted traces, and extract both traces.

    Input
    -----
    DAS_shifted : Data2D_XT, aligned transmitted panel (transmission_cc_align)
    ref_data : dict, reflected data of all branch traces (ref_data_all pkl)
    rec : dict, transmission_receivers output
    cc_all : dict, transmission_cc_align()["cc_all"]

    Output
    ------
    dict: tr_kept (positions in rec arrays), rr_kept (ref_data rows),
          idx_tr_final / idx_rr_final (transmission / reflection channels on DAS_filter),
          cc_tr (max cc of the kept transmitted traces), DAStr, DASrr (paired traces)
    """
    cc_idx_tr = cc_all["trace_i"]
    tr_kept = np.where(np.isin(rec["idx_tr_das"], cc_idx_tr))[0]
    idx_tr_final = rec["idx_tr_das"][tr_kept]
    idx_rr_final = rec["idx_rr_das"][tr_kept]
    rr_kept = rec["idx_tr"][tr_kept]
    cc_tr = cc_all["cc_peak"][np.searchsorted(cc_idx_tr, idx_tr_final)]   # cc_idx_tr is sorted
    return {"tr_kept": tr_kept, "rr_kept": rr_kept,
            "idx_tr_final": idx_tr_final, "idx_rr_final": idx_rr_final, "cc_tr": cc_tr,
            "DAStr": take_traces(DAS_shifted, idx_tr_final),
            "DASrr": take_traces(ref_data["reflection_traces"], rr_kept)}


def window_scattering_traces(DAS_filter, pairs, params=None):
    """
    Step 4b: Tukey window of the paired transmitted and reflected traces and pre-event
    noise of the transmitted traces (reflected noise comes from ref_data).

    Input
    -----
    DAS_filter : Data2D_XT, pre-processed panel (unshifted, for the noise)
    pairs : dict, pair_scattering_traces output
    params : dict or None
        {"time_window": (0.193, 0.218),   # window [s, flat panel]
         "alpha": 0.4,                    # Tukey alpha
         "noise_window": (0.02, 0.105)}   # pre-event noise window [s, DAS_filter]

    Output
    ------
    dict: DAStr_w, DASrr_w (windowed traces), tw_info, rw_info (windowed_das info),
          noise_tr (noise power of each transmitted trace), params
    """
    defaults = {"time_window": (0.193, 0.218), "alpha": 0.4, "noise_window": (0.02, 0.105)}
    p = _merge_params(defaults, params)
    DASrr_w, rw_info = windowed_das(pairs["DASrr"], p["time_window"], p["alpha"])
    DAStr_w, tw_info = windowed_das(pairs["DAStr"], p["time_window"], p["alpha"])
    noise_tr = preevent_noise(DAS_filter, pairs["idx_tr_final"], p["noise_window"])
    return {"DAStr_w": DAStr_w, "DASrr_w": DASrr_w, "tw_info": tw_info, "rw_info": rw_info,
            "noise_tr": noise_tr, "params": p}


def build_scat_dict(ref_data, rec, pairs, win):
    """
    Step 5: scattering-data dictionary (paired transmitted + reflected traces and their
    geometry) exported for the transmission-reflection inversion.

    Input
    -----
    ref_data : dict, reflected data of all branch traces (ref_data_all pkl)
    rec, pairs, win : transmission_receivers, pair_scattering_traces,
                      window_scattering_traces outputs

    Output
    ------
    dict with the keys of DASscatData.ipynb scattering_data
    """
    rr = pairs["rr_kept"]
    out = {"transmitted_traces": win["DAStr_w"],
           "reflection_traces": win["DASrr_w"],
           "idx_trans_traces": pairs["idx_tr_final"],
           "idx_ref_traces": pairs["idx_rr_final"],
           "max_cc_ref": ref_data["max_cc_positive"][rr],
           "max_cc_tr": pairs["cc_tr"],
           "ref_points": ref_data["ref_points"][rr]}
    for k in ("theta_i_deg", "theta_r_deg", "theta_t_deg", "u_inc", "u_ref", "u_trans",
              "p_trans", "p_ref", "n_p_trans_dot", "n_p_ref_dot", "n_u_trans_dot", "n_u_ref_dot"):
        out[k] = ref_data[k][rr]
    out["normal"] = ref_data["normal"]                                      # one fracture normal
    out["dist_source_rr"] = ref_data["dist_source_rr"][rr]                  # source -> reflection arrival
    out["dist_source_tr"] = rec["transmission"]["dist_tr"][pairs["tr_kept"]]  # source -> transmission arrival
    out["time_window"] = ref_data["time_window"]
    out["noise_ref"] = ref_data["noise"][rr]
    out["noise_tr"] = win["noise_tr"]
    return out
