"""
DAS_model_inversion: forward modeling and compliance (cn, ct) inversion of reflected and
transmitted DAS waves. Combines the functions of DAS_inversion.py, DAS_modeling.py and
DASref_modeling.py that the top-level notebooks (and DAS_plots.py / cnct_search.py) call,
copied unchanged except:
  - calls between the three modules (DASinv. / DASmod. / DASrefmod.) are plain calls here;
  - DAS_inversion's C_spectrum_amplitude (cost / sum|D|^2) is C_spectrum_amplitude_norm;
    C_spectrum_amplitude is DASref_modeling's (raw sum |D - E|^2).
  - windowed_sinc is kept once (DAS_modeling and DASref_modeling copies are identical).
"""
import numpy as np
from scipy.signal.windows import tukey
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import correlate
import functools
import pickle
from pathlib import Path
import copy


# ---------------------------------------------------------------------------
# from DAS_inversion.py
# ---------------------------------------------------------------------------


def calculate_trace_weight(ref_data):
    """
    Per-trace weight from max positive/negative lobe correlations.

        cc_i = min(cc_pos_i, cc_neg_i)
        w_i  = cc_i^2 / (1 - cc_i^2)

    Parameters
    ----------
    ref_data : dict
        Must contain 1D arrays:
          - 'max_cc_positive'
          - 'max_cc_negative'

    Returns
    -------
    w : ndarray, shape (n,)
        Trace weights. Non-finite or |cc| >= 1 → np.nan.
    """
    if "max_cc_positive" not in ref_data or "max_cc_negative" not in ref_data:
        raise KeyError("ref_data needs 'max_cc_positive' and 'max_cc_negative'")

    cc_pos = np.asarray(ref_data["max_cc_positive"], float).ravel()
    cc_neg = np.asarray(ref_data["max_cc_negative"], float).ravel()
    if cc_pos.shape != cc_neg.shape:
        raise ValueError(
            f"max_cc_positive and max_cc_negative length mismatch: "
            f"{cc_pos.size} vs {cc_neg.size}"
        )

    cc = np.minimum(cc_pos, cc_neg)
    cc2 = cc * cc

    w = np.full_like(cc, np.nan, dtype=float)
    ok = np.isfinite(cc) & (cc2 < 1.0)
    w[ok] = cc2[ok] / (1.0 - cc2[ok])
    return w
  

def real_wave(Wave):   
    # Takes the real part of inverse fourier transform and shifts it to compensate the small offset from zero amplitude at t=0 (if any)
    rwave=ifft(Wave).real
    shift=np.mean([rwave[0],rwave[-1]])
    return rwave-shift

def shift_zero_pad(x, lag):
    """
    Shift 1-D array `x` by `-lag` samples, same convention as
    np.roll(x, -lag) (positive lag shifts x to EARLIER indices) but
    zero-padded instead of circular: samples that would wrap around from
    one end to the other are dropped, and zeros fill the vacated end
    instead. Same length as `x`.

    Use this instead of np.roll for shifting a finite time-domain trace:
    np.roll recycles whatever was at the far end of the array into the
    end being shifted into, which is only harmless if that content is
    already ~zero there (true for this dataset's small lags and
    naturally-decaying synthetics, but not guaranteed in general -- e.g.
    a larger max_lag, or a compliance regime whose coda doesn't decay
    within the record, could splice real signal across the wrap).
    """
    x = np.asarray(x)
    n = x.size
    lag = int(lag)
    out = np.zeros_like(x)
    if lag == 0:
        out[:] = x
    elif lag > 0:
        if lag < n:
            out[: n - lag] = x[lag:]
    else:
        shift = -lag
        if shift < n:
            out[shift:] = x[: n - shift]
    return out


def align_by_cc(trace, reference, max_lag=None):
    """
    Shift `trace` to align with `reference` via full-trace cross-correlation.
    Returns (aligned_trace, lag_samples). Same length/grid as inputs;
    use the same taxis for plotting.

    `max_lag` (samples, default None = unrestricted) caps the search to
    lags in [-max_lag, max_lag]. Cross-correlation alignment is only
    physically meaningful as a small correction for genuine sub-sample
    timing jitter between two traces of the SAME underlying shape/polarity
    -- it cannot represent "this candidate has the wrong polarity/shape",
    which is real, compliance-dependent physics for a fracture reflection
    coefficient (see tests/reflection_polarity_reversal_notes.md), not a
    timing error. Left unrestricted, a candidate trace that is genuinely a
    poor (e.g. anti-correlated) match at zero lag will still be "aligned"
    by chasing whatever large, physically implausible shift happens to
    maximize the raw dot product against some other part of the
    reference -- producing a misleadingly good-looking but physically
    meaningless overlay instead of an honestly poor, informative misfit.
    Set `max_lag` to a small value (a few samples/ms, matching the actual
    expected pick/timing uncertainty in the data) to keep the correction
    limited to genuine timing jitter.
    """
    x = np.asarray(trace, float).real
    y = np.asarray(reference, float).real
    x0 = x - x.mean()
    y0 = y - y.mean()
    n = y0.size
    cc = correlate(x0, y0, mode="full")
    lags = np.arange(-(n - 1), n)
    if max_lag is not None:
        mask = np.abs(lags) <= int(max_lag)
        lags = lags[mask]
        cc = cc[mask]
    lag = int(lags[np.argmax(cc)])
    return shift_zero_pad(x, lag), lag


def align_by_window_cc(trace, reference, taxis, time_window, max_lag=None, cc="signed"):
    """
    Cross-correlation alignment restricted to a fixed physical time
    window, instead of either the full trace (align_by_cc's failure mode:
    dominated by compliance-dependent coda far outside any physically
    meaningful window) or a single dominant sample (align_by_peak's
    failure mode: fragile when two lobes are comparably sized -- this
    dataset genuinely has a ~4-5 ms two-lobe ambiguity, confirmed by the
    picked single-sample peak flipping between two lobes from trace to
    trace with no smooth transition).

    `cc` selects between two ways of picking the winning lag from the
    normalized correlation curve rho(lag):
      - "signed" (default): lag = argmax(rho) -- only a SAME-polarity
        match can win; a candidate that is genuinely anti-correlated
        with the data at every lag gets a poor, near-arbitrary alignment
        (see cc_alignment_for_inversion_notes.md for why this trade-off
        was chosen: it avoids ever mis-aligning a same-polarity trace).
      - "abs": lag = argmax(|rho|) -- a strong negative correlation is
        treated as equally valid evidence of correct timing as a strong
        positive one (matches DAS_processing.cc_lag_samples_abs's
        convention); which sign actually gets applied is still decided
        separately by s_hat, not here.

    `time_window` should be the DATA's own fixed extraction window (e.g.
    (0.193, 0.218) from DASrefData.ipynb's windowed_das call) -- that is
    the one genuinely fixed time range in this pipeline. The candidate
    model trace's natural duration is NOT fixed the same way (R(omega)
    smears energy toward increasing time, by a compliance-dependent
    amount -- see notch_investigation_notes.md), so `trace` is not
    re-windowed/tapered, only EVALUATED on this same fixed range for the
    purpose of finding the alignment; its values outside the window
    (part of the returned aligned_trace) are left untouched.

    Both `trace` and `reference` are sliced to the same sample range
    (taxis in time_window) and cross-correlated over just that range --
    aggregating the whole windowed shape (both lobes, if within the
    window) rather than gambling on a single sample, while the window's
    own natural length caps how far the fit can wander even before
    `max_lag` is applied.

    The lag is picked by argmax(rho) within the window, where rho is the
    NORMALIZED cross-correlation (Pearson correlation coefficient) at
    each lag, bounded to [-1, 1]:

        rho[k] = sum_n (x[n]-xbar)(y[n+k]-ybar)
                 / sqrt(sum_n (x[n]-xbar)^2 * sum_n (y[n+k]-ybar)^2)

    with x = reference (real data), y = trace (candidate model), and
    both the sums and the local means (xbar, ybar) taken over only the
    OVERLAPPING samples at that lag (so rho stays correctly bounded to
    [-1,1] even for a partial-overlap lag near the edge of the window,
    unlike scipy.signal.correlate's raw, unnormalized dot product).
    The lag closest to rho=+1 wins -- this picks the SAME-polarity match
    only; unlike the earlier argmax(|correlation|) version, a strong
    negative correlation (rho near -1) is no longer treated as valid
    evidence of correct timing here.

    Returns (aligned_trace, lag_samples, cc_value): the full-length
    `trace` shifted (zero-padded, see shift_zero_pad) by the fitted lag
    (same lag convention as align_by_cc/align_by_peak), the lag itself,
    and cc_value = the SIGNED normalized correlation rho in [-1, 1] at the
    winning lag (for cc="abs" it can therefore be negative; nan if no
    lag was valid). NOTE: 3 return values, unlike the 2 of
    align_by_cc/align_by_peak -- callers must unpack all three.

    Lag choice: the lag with the highest score within the allowed range;
    but if that lag is on the EDGE of the range (+-max_lag, or +-nwin//2
    when max_lag is None/larger), the highest INTERIOR local maximum of
    the score is used instead -- an edge maximum is usually the rising
    flank of a stronger match outside the range, not a real alignment.
    The edge lag is kept only when the score has no interior peak.

    A lag is only ever considered if the resulting overlap is at least
    half the window (|lag| <= nwin//2), regardless of `max_lag`. This is
    a required safeguard, not an optional tightening: with a tiny
    overlap (e.g. 2 samples), the normalized correlation coefficient is
    trivially +-1 no matter what the values are (a line through 2 points
    always fits exactly), so without this floor a near-edge, essentially
    unconstrained lag can spuriously outscore every genuine, well-
    supported match computed over the full overlap.
    """
    x = np.asarray(trace, float).real
    y = np.asarray(reference, float).real
    taxis = np.asarray(taxis, float)
    t0, t1 = time_window
    mask = (taxis >= t0) & (taxis <= t1)
    if not np.any(mask):
        raise ValueError("time_window does not overlap taxis")

    xw = x[mask]
    yw = y[mask]
    nwin = yw.size
    lags = np.arange(-(nwin - 1), nwin)
    lags = lags[np.abs(lags) <= nwin // 2]
    if max_lag is not None:
        lags = lags[np.abs(lags) <= int(max_lag)]

    # Vectorized equivalent of the per-lag loop (yw=data="x" in the formula,
    # unshifted; xw=trace="y" in the formula, shifted by k), via cumulative
    # sums for the local (overlap-only) normalization terms and a single
    # scipy.signal.correlate call for the cross term -- verified to match
    # the straightforward per-lag loop to float precision, ~40x faster
    # (the loop was the actual cause of this function being slow/"hanging"
    # over a full grid search: ~n_grid*n_trace calls x ~nwin/2 lags each,
    # all in pure Python).
    L = (nwin - np.abs(lags)).astype(float)
    cs_d = np.concatenate([[0.0], np.cumsum(yw)])
    cs_d2 = np.concatenate([[0.0], np.cumsum(yw ** 2)])
    cs_s = np.concatenate([[0.0], np.cumsum(xw)])
    cs_s2 = np.concatenate([[0.0], np.cumsum(xw ** 2)])

    pos = lags >= 0
    Sd = np.where(pos, cs_d[np.clip(nwin - lags, 0, nwin)],
                  cs_d[nwin] - cs_d[np.clip(-lags, 0, nwin)])
    Sd2 = np.where(pos, cs_d2[np.clip(nwin - lags, 0, nwin)],
                   cs_d2[nwin] - cs_d2[np.clip(-lags, 0, nwin)])
    Ss = np.where(pos, cs_s[nwin] - cs_s[np.clip(lags, 0, nwin)],
                  cs_s[np.clip(nwin + lags, 0, nwin)])
    Ss2 = np.where(pos, cs_s2[nwin] - cs_s2[np.clip(lags, 0, nwin)],
                   cs_s2[np.clip(nwin + lags, 0, nwin)])

    cross_full = correlate(yw, xw, mode="full")
    full_lags = np.arange(-(nwin - 1), nwin)
    Sds = cross_full[np.searchsorted(full_lags, -lags)]

    num = Sds - Sd * Ss / L
    den = np.sqrt(np.clip(Sd2 - Sd ** 2 / L, 0, None) * np.clip(Ss2 - Ss ** 2 / L, 0, None))
    rho = np.full(lags.shape, -np.inf)
    ok = (L >= 2) & (den > 0)
    rho[ok] = num[ok] / den[ok]

    if cc == "abs":
        score = np.full(lags.shape, -np.inf)
        score[ok] = np.abs(rho[ok])
    elif cc == "signed":
        score = rho
    else:
        raise ValueError(f"cc must be 'signed' or 'abs', got {cc!r}")

    # Winning lag = global maximum of the score over the allowed lags,
    # except when it falls on the range edge (see edge rule below).
    # (A former "closer-to-zero of the top 2 local maxima" guard was removed:
    # it could prefer a much weaker peak purely by proximity to lag 0.
    # Cycle skipping is controlled via `max_lag` instead -- keep it below
    # ~half the dominant period so only one cycle fits in the search.)
    k_best = int(np.argmax(score))
    # Edge rule: if the global max sits on the EDGE of the allowed lags
    # (+-max_lag, or +-nwin//2), it is usually not a real peak but the
    # flank of a stronger match lying outside the range (e.g. a cycle
    # skip). Only in that case, switch to the highest INTERIOR local
    # maximum (score higher than the previous lag and >= the next one);
    # if there is none (score monotonic over the range), keep the edge.
    if k_best in (0, lags.size - 1) and lags.size >= 3:
        inner = np.arange(1, lags.size - 1)
        is_pk = (score[inner] > score[inner - 1]) & (score[inner] >= score[inner + 1])
        pk = inner[is_pk]
        if pk.size:
            k_best = int(pk[np.argmax(score[pk])])
    lag = int(lags[k_best])
    # signed rho at the winning lag (-inf -> nan if no valid lag)
    cc_value = float(rho[k_best]) if np.isfinite(rho[k_best]) else np.nan
    return shift_zero_pad(x, lag), lag, cc_value


def estimate_reflection(trans_data_fft, rr_data_trace, GR, GT,
                       vp, vs, ps, rho, omega, cn, ct, wave='SS', eta=1e-3):
    """
      Estimate reflected response from transmitted spectrum 
    """
    R, T = RPS_coef(vp, vs, ps, rho, omega, cn, ct, wave)

    F = GT * T
    Fabs=np.abs(F)
    eps = float(np.max(Fabs)) * float(eta)
    F2 = np.abs(F) ** 2
                                                                                                                                                                                                                                                                                                         
    # Reflection model in frequency domain
    ER_model = (GR * R * np.conj(F) * trans_data_fft) / (F2 + eps**2)
    # Reflection model in time domain
    ER_time = real_wave(ER_model)
    # Align reflection model in time domain with reflection traces in time domain
    ER_time_cc, lag = align_by_cc(ER_time, rr_data_trace)
    return  {   
        "reflection_model_fft": ER_model,
        "reflection_model_time": ER_time,
        "reflection_model_time_cc": ER_time_cc,    # after CC; same taxis
        "cc_lag_samples": lag,
    }

def C_spectrum_amplitude_norm(D, E):
    """
    Normalized spectral L2 misfit.

        C = Σ_f |D(f) - E(f)|² / Σ_f |D(f)|²
    """
    D = np.asarray(D)
    E = np.asarray(E)
    if D.shape != E.shape:
        raise ValueError(f"D and E shape mismatch: {D.shape} vs {E.shape}")

    denom = float(np.sum(np.abs(D) ** 2))
    if denom == 0.0:
        raise ValueError("Σ |D(f)|² is zero; cannot normalize")

    return float(np.sum(np.abs(D - E) ** 2) / denom)

def C_wavelet(d, e):
    """
    Normalized time-domain wavelet misfit.

        C = Σ_t (d(t) - e(t))² / Σ_t d(t)²
    """
    d = np.asarray(d, float).real
    e = np.asarray(e, float).real
    if d.shape != e.shape:
        raise ValueError(f"d and e shape mismatch: {d.shape} vs {e.shape}")

    denom = float(np.sum(d**2))
    if denom == 0.0:
        raise ValueError("Σ d(t)² is zero; cannot normalize")

    return float(np.sum((d - e) ** 2) / denom)

def RPS_coef(vp,vs, p,rho, omega, cn,ct, wave):
    # POST-CRITICAL FIX (2026-09-29; the previous version is kept as RPS_coef_deprecated).
    # Beyond the critical angle asin(vs/vp) (SS and SP only; PP/PS never), cos(theta_P) is
    # imaginary: the converted P wave is evanescent and must DECAY away from the fracture at
    # every frequency. In numpy's e^{+i w t} convention (the one the traction-sign
    # re-derivation below targets) that requires
    #     cos(theta_P) = -i sgn(omega) sqrt|cp|      (sgn(0) taken as +1)
    # and sin(2 theta_P) = 2 sin(theta_P) cos(theta_P), so it carries the same sign.
    # The deprecated version used +i for all omega (and sqrt(1-cos2th^2), always +i), which
    # is right only for omega < 0 and gave a non-Hermitian spectrum (spurious notches on
    # omega > 0, real(ifft) mixing the two branches). With the fix R(-w) = conj(R(w)) and
    # T(-w) = conj(T(w)) exactly. Only the lines marked "fix:" differ from the deprecated
    # version; PP/PS at all angles and SS/SP pre-critical are identical to it.
    # Check: tests/spectrum_inversion/test_RPS_coef_v2.py.
    # Calculates the complex and frquency-dependant reflection/Transmission (R/T) coefficients from a fracture modeled as a slip interface embedded in a whole space. 
    # Contribution: The resulting coefficients yield causal signals in time-domain.
    # To this end, formulas were re-derived but considering a negative sign in the tractions : T= -T1x1-T3x3. Fracture is along x1, with normal: -x3
    # Other aspects of the derivation follows the sign and coordinates convention of Aki and Richards as in  Chapter 5 of the Quantitative seismology Book.
    # However expressions are similar to the ones found in Pyrak-Nolte 1990: Transmission of seismic waves across single natural fractures
    # Input
    # vp, vs: P- and S-velocities, respectively. Floats
    # p: ray parameter p=sin(theta_i)/v where theta_i: incidence angle and v is the velocity of the medium
    # rho: density of the medium
    # omega: angular frequency. Float
    # cn,ct: normal and tangential compliance of the fracture. Floats
    #wave: Indicates the type of reflection/transmission. String. Possible strings are :'PP', 'PS', 'SS' or 'SP'
    # in
    cp=1-p**2*vp**2
    cs=1-p**2*vs**2
    # fix: evanescent branch follows the sign of omega (decays for w>0 and w<0 alike)
    sgn=np.where(np.asarray(omega)<0, -1.0, 1.0)
    costh=np.where(cp>0, np.sqrt(np.abs(cp))+0j, -1j*sgn*np.sqrt(np.abs(cp)))
    cosph=np.where(cs>0, np.sqrt(np.abs(cs))+0j, -1j*sgn*np.sqrt(np.abs(cs)))

    sinth=np.sqrt(1-costh**2+0j)
    sinph=np.sqrt(1-cosph**2+0j)

    cos2th=2*costh**2-1
    cos2ph=2*cosph**2-1

    # fix: 2 sin cos keeps the sign of the imaginary cosine (sqrt(1-cos2^2) always gave +i)
    sin2th=2*sinth*costh
    sin2ph=2*sinph*cosph


    kn=1/cn
    kt=1/ct
    zp=rho*vp
    zs=rho*vs

    dd=(-2*1j*kn*(cos2ph*costh*zp + sin2th*sinph*zs)+ (cos2ph**2*zp**2 + sin2th*sin2ph*zs**2)*omega)* \
        (-2*1j*kt*zp*(cos2ph*cosph*zp+sin2ph*sinth*zs)+zs*(cos2ph**2*zp**2+sin2th*sin2ph*zs**2)*omega)
    
    match wave:
        case 'PP':
            R = omega*(2*1j*cos2ph**3*cosph*kt*zp**4+2*1j*cos2ph**2*kt*sin2ph*sinth*zp**3*zs - \
                       2*1j*cos2ph*costh*kn*sin2th*sin2ph*zp*zs**3 - \
                       cos2ph**4*zp**4*zs*omega + sin2th**2*sin2ph*zs**4*(-2*1j*kn*sinph+sin2ph*zs*omega)) / dd
            
            T= (-4*kn*kt*zp*(cos2ph*cosph*zp+sin2ph*sinth*zs)*(cos2ph*costh*zp+sin2th*sinph*zs) -\
                2*1j*zp*zs*(cos2ph**3*costh*kn*zp**2 + cos2ph*sin2th*(cosph*kt*sin2ph + cos2ph*kn*sinph)*zp*zs + \
                kt*sin2th*sin2ph**2*sinth*zs**2 )*omega) / dd

        case 'PS':
            R = 2*cos2ph*sin2th*zp*zs*omega*(-1j*(cos2ph*zp*(cosph*kt*zp+costh*kn*zs) + \
                  zs*(kt*sin2ph*sinth*zp+kn*sin2th*sinph*zs))+\
                  zs*(cos2ph**2*zp**2 +sin2th*sin2ph*zs**2)*omega) /dd

            T= (-2*1j*cos2ph*sin2th*zp*zs*(cos2ph*zp*(cosph*kt*zp-costh*kn*zs)+ \
               zs*(kt*sin2ph*sinth*zp - kn*sin2th*sinph*zs))*omega) / dd
               
        case 'SS': 
            R = zs*omega*(2*1j*zp*(-cos2ph**3*costh*kn*zp**2 + \
                cos2ph*sin2th*(cosph*kt*sin2ph-cos2ph*kn*sinph)*zp*zs +\
                kt*sin2th*sin2ph**2*sinth*zs**2) + ( cos2ph**4*zp**4 - sin2th**2*sin2ph**2*zs**4)*omega) /dd
                           
            T=(-4*kn*kt*zp*(cos2ph*cosph*zp+sin2ph*sinth*zs)*(cos2ph*costh*zp+sin2th*sinph*zs) - \
                2*1j*(cos2ph**3*cosph*kt*zp**4 + cos2ph**2*kt*sin2ph*sinth*zp**3*zs + \
                cos2ph*costh*kn*sin2th*sin2ph*zp*zs**3 +\
                 kn*sin2th**2*sin2ph*sinph*zs**4)*omega) / dd
        case 'SP':
            R = 2*cos2ph*sin2ph*zp*zs*omega*(-1j*(cos2ph*zp*(cosph*kt*zp+costh*kn*zs) + \
                zs*(kt*sin2ph*sinth*zp +kn*sin2th*sinph*zs)) +\
                zs*(cos2ph**2*zp**2+sin2th*sin2ph*zs**2)*omega ) / dd

            T= (-2*1j*cos2ph*sin2ph*zp*zs*(cos2ph*zp*(cosph*kt*zp-costh*kn*zs)+\
               zs*(kt*sin2ph*sinth*zp-kn*sin2th*sinph*zs))*omega) /dd
    return R,T


# ---------------------------------------------------------------------------
# from DAS_modeling.py
# ---------------------------------------------------------------------------

def fft_traces_scat(scat_data):
    """
    Purpose
    -------
    Scattering-data version of fft_traces: FFT of every reflected AND
    transmitted trace, stored on each traces object, plus the common
    frequency axis stored in the dictionary:
        scat_data['reflection_traces'].fft  = np.fft.fft(data, axis=1)
        scat_data['transmitted_traces'].fft = np.fft.fft(data, axis=1)
        scat_data['freqs'] = np.fft.fftfreq(nt, dt)
    Full two-sided spectra of the stored (already windowed) traces, in
    np.fft order, so they match scat_data['freqs']. Both trace sets must be
    sampled identically (same dt and time axis), which is checked first.

    Input
    -----
    scat_data : dict (data/scattering_data.pkl from DASscatData.ipynb) with
        'reflection_traces', 'transmitted_traces' : Data2D_XT with
        .data (n_tr, nt), .taxis (nt,) and .dt [s].

    Output
    ------
    scat_data : the same dict (updated in place), with
        scat_data['reflection_traces'].fft  : complex (n_tr, nt)
        scat_data['transmitted_traces'].fft : complex (n_tr, nt)
        scat_data['freqs'] : (nt,) frequency axis [Hz], np.fft order.
    """
    rr = scat_data["reflection_traces"]
    tr = scat_data["transmitted_traces"]

    # --- both trace sets must share the same sampling ---
    dt_r, dt_t = float(rr.dt), float(tr.dt)
    if not np.isclose(dt_r, dt_t):
        raise ValueError(f"dt differs: reflection {dt_r} s, transmission {dt_t} s")
    t_r = np.asarray(rr.taxis, float)
    t_t = np.asarray(tr.taxis, float)
    if t_r.shape != t_t.shape or not np.allclose(t_r, t_t):
        raise ValueError("reflection and transmission time axes (taxis) differ")
    nt = t_r.size
    if np.asarray(rr.data).shape[1] != nt or np.asarray(tr.data).shape[1] != nt:
        raise ValueError("number of samples in .data does not match taxis")

    # --- FFT of all traces at once (time is axis 1) ---
    rr.fft = np.fft.fft(np.asarray(rr.data, float), axis=1)
    tr.fft = np.fft.fft(np.asarray(tr.data, float), axis=1)

    # --- common frequency axis [Hz], same order as the FFTs ---
    scat_data["freqs"] = np.fft.fftfreq(nt, d=dt_r)
    return scat_data


def add_noise_weights_scat(scat_data, taper_alpha=None):
    """
    Purpose
    -------
    Scattering-data version of DASref_modeling.add_noise_weights: SNR and
    noise weight of each reflected AND transmitted trace from its stored
    pre-event noise power, added to scat_data. Same method for both sets:
    the noise power sigma2 (preevent_noise, measured on the unshifted panel)
    is compared with the trace's power in the fit window. The fit window was
    tapered (windowed_das, Tukey), which also lowers the noise inside it, so
    the noise power is scaled by the taper's mean square:

        noise_win_k = sigma2_k * mean(w^2)
        SNR_k       = (mean(d_k^2) - noise_win_k) / noise_win_k   (>= 0)
        w_k         = 1 / sigma2_k   (maximum-likelihood weight)

    Input
    -----
    scat_data : dict (data/scattering_data.pkl from DASscatData.ipynb), needs
        'reflection_traces', 'transmitted_traces' : windowed traces (Data2D_XT)
        'time_window' : (tbegin, tend) [s], window used by windowed_das
        'noise_ref', 'noise_tr' : (n_tr,) pre-event sigma2 per trace, same
            trace order as reflection_traces / transmitted_traces.
    taper_alpha : float or None
        Tukey alpha windowed_das used on the fit window; None -> use
        scat_data['taper_alpha'] if present, else 0.4 (DASscatData.ipynb).

    Output
    ------
    scat_data : the same dict, updated in place (also returned) with (n_tr,):
        'snr_noise_ref', 'w_noise_ref' : reflected traces
        'snr_noise_tr',  'w_noise_tr'  : transmitted traces
    """
    if taper_alpha is None:
        taper_alpha = float(scat_data.get("taper_alpha", 0.4))
    tb, te = scat_data["time_window"]

    # (traces key, noise key, output suffix) for the two trace sets
    sets = (("reflection_traces", "noise_ref", "ref"),
            ("transmitted_traces", "noise_tr", "tr"))
    for traces_key, noise_key, tag in sets:
        if noise_key not in scat_data:
            raise KeyError(f"scat_data needs '{noise_key}' (pre-event sigma2, from preevent_noise)")
        das = scat_data[traces_key]
        t = np.asarray(das.taxis, float)
        win = (t >= tb) & (t <= te)                          # samples of the fit window
        D = np.asarray(das.data, float)[:, win]
        sigma2 = np.asarray(scat_data[noise_key], float).ravel()
        if sigma2.size != D.shape[0]:
            raise ValueError(f"'{noise_key}' has {sigma2.size} values for {D.shape[0]} {traces_key}")

        w = tukey(int(win.sum()), alpha=taper_alpha, sym=True)
        noise_win = sigma2 * np.mean(w**2)                   # noise power after the same taper
        snr = np.clip(np.mean(D**2, axis=1) - noise_win, 0, None) / noise_win

        scat_data[f"snr_noise_{tag}"] = snr
        scat_data[f"w_noise_{tag}"] = 1.0 / sigma2
    return scat_data


def RfromT(scat_data, idx, vp, vs, rho, cn, ct, eps, wave="SS"):
    """
    Purpose
    -------
    Model the reflected DAS spectrum of pair idx from its observed transmitted
    spectrum (stabilized deconvolution of the transmission path), for a
    fracture with compliances (cn, ct). Scattering-data analogue of
    DASref_modeling.R_wave_v2, with the transmitted trace in place of the
    source wavelet and F = G_T * T in place of G_d:

        F(ω)    = G_T(ω) T(ω)
        η       = max(|F(ω)|) * eps
        E_R(ω)  = E_T(ω) G_R(ω) R(ω) conj(F(ω)) / (|F(ω)|² + η²)

    R, T are the frequency-dependent reflection / transmission coefficients
    of the slip interface (DAS_inversion.RPS_coef) at the pair's incidence
    angle, ray parameter p = sin(theta_i) / vs.

    Input
    -----
    scat_data : dict (data/scattering_data.pkl, after fft_traces_scat,
        geometric_factor and the notebook's update), needs
        'transmitted_traces' : Data2D_XT with .fft (n_tr, nt) and .taxis
        'Gr', 'Gt' : (n_tr, nt) reflected / transmitted geometric factors
        'omega'    : (nt,) angular frequency [rad/s], np.fft order
        'freqs'    : (nt,) frequency [Hz], np.fft order
        'theta_i_deg' : (n_tr,) incidence angle [deg]
    idx : int
        Pair index (row of every per-pair array).
    vp, vs : float
        P and S velocity [m/s].
    rho : float
        Density [kg/m^3].
    cn, ct : float
        Normal and tangential fracture compliance [m/Pa].
    eps : float
        Stabilization fraction (>= 0); η = max(|F|) * eps.
    wave : str
        Wave type for RPS_coef ('SS', 'PP', 'PS', 'SP'). Default 'SS'.

    Output
    ------
    dict with
        'Er_omega' : (nt,) complex, modeled reflected spectrum E_R(ω)
        'Er_time'  : (nt,) real, modeled reflected trace (DAS_inversion.real_wave)
        'time'     : (nt,) time axis [s] (transmitted_traces.taxis)
        'freqs'    : (nt,) frequency axis [Hz]
        'idx'      : pair index
        'angle'    : incidence angle [deg]
        'F'        : (nt,) complex G_T * T
        'eps', 'eta' : stabilization fraction and value
    """
    req = ("transmitted_traces", "Gr", "Gt", "omega", "freqs", "theta_i_deg")
    missing = [k for k in req if k not in scat_data]
    if missing:
        raise ValueError(f"scat_data missing keys: {missing}")

    idx = int(idx)
    eps = float(eps)
    if eps < 0:
        raise ValueError("eps must be >= 0")

    DAS_tr = scat_data["transmitted_traces"]
    if not hasattr(DAS_tr, "fft"):
        raise ValueError("transmitted_traces has no .fft (run fft_traces_scat first)")

    Gr = np.asarray(scat_data["Gr"])
    Gt = np.asarray(scat_data["Gt"])
    omega = np.asarray(scat_data["omega"], float)
    freqs = np.asarray(scat_data["freqs"], float)
    theta_i_deg = np.asarray(scat_data["theta_i_deg"], float)
    ET_all = np.asarray(DAS_tr.fft)

    n_pairs = ET_all.shape[0]
    if not (0 <= idx < n_pairs):
        raise ValueError(f"idx={idx} out of range (n_pairs={n_pairs})")
    if Gr.shape != ET_all.shape or Gt.shape != ET_all.shape:
        raise ValueError(f"Gr {Gr.shape} / Gt {Gt.shape} must match transmitted .fft {ET_all.shape}")
    if omega.shape != freqs.shape or freqs.size != ET_all.shape[1]:
        raise ValueError("omega / freqs length must match the number of samples")

    ET = ET_all[idx]                                    # observed transmitted spectrum E_T(ω)
    GR = Gr[idx]                                        # reflected geometric factor G_R(ω)
    GT = Gt[idx]                                        # transmitted geometric factor G_T(ω)

    # R(ω), T(ω) of the fracture at this pair's incidence angle
    theta_i = np.deg2rad(theta_i_deg[idx])
    ps = np.sin(theta_i) / float(vs)                    # ray parameter [s/m]
    R, T = RPS_coef(vp, vs, ps, rho, omega, cn, ct, wave)

    # stabilized deconvolution of the transmission path F = G_T T
    F = GT * T
    F_abs = np.abs(F)
    eta = float(np.max(F_abs)) * eps
    if eta == 0.0 and np.any(F_abs == 0.0):
        # G_T is zero beyond its windowed-sinc lobe, so F = 0 there -> 0/0
        raise ValueError("F = G_T*T is zero at some frequencies (G_T sinc cut-off): use eps > 0")
    denom = F_abs**2 + eta**2

    Er_omega = ET * GR * R * np.conj(F) / denom
    Er_time = real_wave(Er_omega)

    return {
        "Er_omega": Er_omega,
        "Er_time": Er_time,
        "time": np.asarray(DAS_tr.taxis, float),
        "freqs": freqs,
        "idx": idx,
        "angle": float(theta_i_deg[idx]),
        "F": F,
        "eps": eps,
        "eta": eta,
    }


def _scat_noise_options(scat_data, cap=False, taper_alpha=None):
    """
    Purpose
    -------
    Per-pair noise weight and SNR for cnct_search_RfromT's trace_opts, from
    the pre-event noise powers of each pair's two channels: noise_ref
    (sigma_R^2, reflection) and noise_tr (sigma_T^2, transmission).
    With cap=True each noise power is first limited to its own median over
    all pairs of scat_data (reflection and transmission separately), because
    the pre-event window of some channels contains transient signal:
        sigma^2 -> min(sigma^2, median(sigma^2))
    Then, with the (capped or uncapped) noise:
        w_i   = 1 / (sigma_R^2 + sigma_T^2)                  combined weight
        SNR   = (mean(d^2) - noise_win) / noise_win          per trace, as
                add_noise_weights_scat (fit window, taper-scaled noise)
        snr_i = min(SNR_reflection, SNR_transmission)        pair SNR

    Input
    -----
    scat_data : dict with 'noise_ref', 'noise_tr', 'reflection_traces',
        'transmitted_traces' (windowed traces, .data and .taxis) and
        'time_window'.
    cap : bool
        True -> median-capped noise ("noise_cap"); False -> as stored ("noise").
    taper_alpha : float or None
        Tukey alpha of the fit window (windowed_das); None -> use
        scat_data['taper_alpha'] if present, else 0.4.

    Output
    ------
    dict of (n_pairs,) arrays: 'w' (weight), 'snr' (pair SNR),
        'snr_ref', 'snr_tr' (per trace), 'noise_ref', 'noise_tr' (noise used).
    """
    missing = [k for k in ("noise_ref", "noise_tr") if k not in scat_data]
    if missing:
        raise KeyError(f"noise-based trace_opts need scat_data keys {missing} (pre-event noise power)")
    if taper_alpha is None:
        taper_alpha = float(scat_data.get("taper_alpha", 0.4))
    sR = np.asarray(scat_data["noise_ref"], float).ravel()
    sT = np.asarray(scat_data["noise_tr"], float).ravel()
    if cap:                                                 # each channel set capped at its own median
        sR = np.minimum(sR, np.median(sR))
        sT = np.minimum(sT, np.median(sT))

    tb, te = scat_data["time_window"]
    out = {"w": 1.0 / (sR + sT), "noise_ref": sR, "noise_tr": sT}
    for traces_key, sig2, tag in (("reflection_traces", sR, "ref"),
                                  ("transmitted_traces", sT, "tr")):
        das = scat_data[traces_key]
        t = np.asarray(das.taxis, float)
        win = (t >= tb) & (t <= te)                         # samples of the fit window
        D = np.asarray(das.data, float)[:, win]
        w = tukey(int(win.sum()), alpha=taper_alpha, sym=True)
        noise_win = sig2 * np.mean(w ** 2)                  # noise power after the same taper
        out[f"snr_{tag}"] = np.clip(np.mean(D ** 2, axis=1) - noise_win, 0, None) / noise_win
    out["snr"] = np.minimum(out["snr_ref"], out["snr_tr"])  # a pair is as good as its weaker trace
    return out


def cnct_search_RfromT(scat_data, idx, medium, search_param):
    """
    Purpose
    -------
    Time-domain compliance grid search on the paired reflection/transmission
    data: at every (cn, ct) node the reflected trace of each pair is
    predicted from its OBSERVED transmitted trace (forward model RfromT) and
    compared with the observed reflected trace. Same structure as
    DASref_modeling.cnct_searchR_obs (grid types, alignment, trace selection
    and weights, output packs), but:
      - forward model RfromT instead of R_wave_v2 (no source wavelet);
      - NO amplitude scalar in the cost: RfromT predicts the absolute
        reflected amplitude (the source and the source-to-fracture leg are
        common to both legs), so per trace and node
            Csp_i = 0.5 * w_i * sum_t (d_i(t) - h_i(t))^2
        and the node cost is Csp_sum = sum_i Csp_i (minimized over the grid);
        s_hat = sum_i w_i d_i.h_i / sum_i w_i h_i.h_i is still computed at
        every node, as a DIAGNOSTIC only (~1 where amplitudes agree);
      - no attenuation (Q) option;
      - only search_mode "full" (unconstrained grid, as in cnct_searchR_obs)
        is implemented.

    Input
    -----
    scat_data : dict (data/scattering_data.pkl, prepared as in
        DASscatInv.ipynb), needs 'reflection_traces' (.data, .taxis),
        'transmitted_traces' (.fft, low-passed if desired), 'Gr', 'Gt',
        'omega', 'freqs', 'theta_i_deg', 'time_window', and 'noise_ref',
        'noise_tr' when trace_opts is used.
    idx : array-like of int
        Pair indices to fit (rows of the per-pair arrays).
    medium : (vp, vs, rho, wave)
        Velocities [m/s], density [kg/m^3] and wave type for RPS_coef.
    search_param : (search_mode, grid, eps, search_space, nsearch, cc
                    [, constraint[, trace_opts[, max_lag]]])
        search_mode : "full" (only option; "constrained" raises).
        grid : "ratio_ct"  -> (cn/ct ratio, ct) grid, cn = ratio*ct
                              (formerly "cn_ctratio");
               "ratio_cn"  -> (cn/ct ratio, cn) grid, ct = cn/ratio;
               "non_ratio" -> (cn, ct) grid.
        eps : float, stabilization fraction of RfromT (eta = max|F| eps).
        search_space : ((p_min, p_max), (q_min, q_max)): p = cn/ct for
            "ratio_ct" / "ratio_cn", p = cn [m/Pa] for "non_ratio"; q = ct
            [m/Pa] for "ratio_ct" / "non_ratio", q = cn [m/Pa] for
            "ratio_cn".
        nsearch : (n_p, n_ct), log-spaced samples per axis.
        cc : "signed" or "abs", lag rule of DAS_inversion.align_by_window_cc
            (model aligned to the data inside scat_data['time_window']).
        constraint : must be None (default).
        trace_opts : dict or None (default: all pairs, weights 1). Only these
            entries are accepted:
              "weight": "noise"     -> w_i = 1 / (sigma_R^2 + sigma_T^2), the
                                       combined pre-event noise power of the
                                       pair's reflection + transmission channel
                                       (noise_ref + noise_tr);
                        "noise_cap" -> same, with sigma_R^2 and sigma_T^2 each
                                       capped at its own median over all pairs
                                       (pre-event transients on some channels).
              "select": ("snr", min) -> drop pairs whose SNR is below min; the
                                       pair SNR is min(SNR_reflection,
                                       SNR_transmission), computed with the same
                                       noise as the weight (capped for
                                       "noise_cap", uncapped otherwise).
            e.g. {"weight": "noise_cap", "select": ("snr", 7.0)}.
            Weights are normalized to mean 1 over the kept pairs.
        max_lag : int or None (default), |lag| limit [samples] of the
            alignment.

    Output
    ------
    out : dict, the same fields as cnct_searchR_obs's full pack
        ("ratio_ct": _ratio_grid_pack fields, e.g. cn_ratio, ct, cn_grid,
        ratio_sum_best; "ratio_cn": the same relabelled by
        DASref_modeling._ratio_cn_pack, axes cn_ratio and cn, 2-D ct_grid;
        "non_ratio": cnct_search_pack fields, e.g. cn, ct, cn_sum_best;
        'grid' = "ratio_ct" / "ratio_cn" / "cn_ct"), including Csp (n_tr, n_p, n_ct), Csp_sum, s_hat
        (diagnostic), plus 'lag_samples', 'cc_max' (n_tr, n_p, n_ct),
        'data_energy', 'cost_norm' (= Csp_sum / data_energy), 'idx_input',
        'idx_dropped', 'trace_w', 'trace_opts', 'snr_pairs' (pair SNR of
        idx, None without trace_opts), 'search_mode', 'eps',
        'cost' = "0.5 w (d - h)^2, no s_hat".
    """
    vp, vs, rho, wave = medium

    # ---- search_param: (search_mode, grid, eps, space, nsearch, cc[, constraint[, trace_opts[, max_lag]]])
    if len(search_param) not in (6, 7, 8, 9):
        raise ValueError(
            "search_param must be (search_mode, grid, eps, search_space, nsearch, cc"
            "[, constraint[, trace_opts[, max_lag]]])"
        )
    search_mode, grid, eps, search_space, nsearch, cc = search_param[:6]
    constraint = search_param[6] if len(search_param) >= 7 else None
    trace_opts = search_param[7] if len(search_param) >= 8 else None
    max_lag = search_param[8] if len(search_param) >= 9 else None
    if search_mode != "full" or constraint is not None:
        raise NotImplementedError("only search_mode='full' (unconstrained, constraint=None) is available")
    if grid == "cn_ctratio":
        raise ValueError("grid 'cn_ctratio' was renamed 'ratio_ct'")
    if grid not in ("ratio_ct", "ratio_cn", "non_ratio"):
        raise ValueError("grid must be 'ratio_ct', 'ratio_cn' or 'non_ratio'")
    is_ratio = grid in ("ratio_ct", "ratio_cn")          # first axis is cn/ct
    is_rcn = grid == "ratio_cn"                         # second axis is cn (ct = cn / ratio)

    # ---- trace selection / weights (same rules as cnct_searchR_obs)
    idx_input = np.asarray(idx, int).ravel()
    if idx_input.size == 0:
        raise ValueError("idx is empty")
    opts = dict(trace_opts or {})
    unknown = set(opts) - {"weight", "select"}
    if unknown:
        raise ValueError(f"unknown trace_opts keys {sorted(unknown)} (use 'weight', 'select')")
    wopt, sel = opts.get("weight"), opts.get("select")
    if wopt not in (None, "noise", "noise_cap"):
        raise ValueError("trace_opts['weight'] must be 'noise' or 'noise_cap'")
    if sel is not None and (len(sel) != 2 or sel[0] != "snr"):
        raise ValueError("trace_opts['select'] must be ('snr', min_value)")

    keep = np.ones(idx_input.size, bool)
    w_keep, snr_pairs = None, None
    if wopt is not None or sel is not None:
        # weight and SNR from the same noise (median-capped for "noise_cap")
        nopt = _scat_noise_options(scat_data, cap=(wopt == "noise_cap"))
        snr_pairs = nopt["snr"][idx_input]
        if sel is not None:
            keep &= np.isfinite(snr_pairs) & (snr_pairs >= float(sel[1]))
        if wopt is not None:
            w_in = nopt["w"][idx_input]
            keep &= np.isfinite(w_in) & (w_in > 0)
    if not keep.any():
        raise ValueError(f"trace_opts {trace_opts} drops every pair in idx")
    idx_array, idx_dropped = idx_input[keep], idx_input[~keep]
    if wopt is not None:
        w_keep = w_in[keep]
    n_tr = idx_array.size
    trace_w = np.ones(n_tr) if w_keep is None else w_keep / w_keep.mean()   # mean 1

    if "time_window" not in scat_data:
        raise KeyError("scat_data needs 'time_window' (tbegin, tend)")
    D_all = np.asarray(scat_data["reflection_traces"].data, float)[idx_array]   # observed d_i
    taxis = np.asarray(scat_data["reflection_traces"].taxis, float)
    nt = D_all.shape[1]
    align_fn = functools.partial(
        align_by_window_cc, taxis=taxis, time_window=scat_data["time_window"], cc=cc
    )

    # ---- log-spaced axes (first axis p = cn/ct ratio or cn)
    axes = cnct_search_axes(search_space, nsearch)
    p_ax, ct = axes["cn"], axes["ct"]
    n_p, n_ct = axes["n_cn"], axes["n_ct"]

    Csp_all = np.empty((n_tr, n_p, n_ct), float)
    s_hat = np.empty((n_p, n_ct), float)
    lag_samples = np.empty((n_tr, n_p, n_ct), int)
    cc_max = np.empty((n_tr, n_p, n_ct), float)      # signed rho at the chosen lag
    H_all = np.empty((n_tr, nt), float)

    # ---- forward model + alignment + cost at every node
    for j, q_j in enumerate(ct):
        for i, p_i in enumerate(p_ax):
            # actual (cn, ct) at this node for the chosen grid
            if is_rcn:
                cn_i, ct_j = q_j, q_j / p_i
            elif is_ratio:
                cn_i, ct_j = p_i * q_j, q_j
            else:
                cn_i, ct_j = p_i, q_j
            for k, pair in enumerate(idx_array):
                h = RfromT(scat_data, int(pair), vp, vs, rho, cn_i, ct_j, eps, wave)["Er_time"]
                H_all[k], lag_samples[k, i, j], cc_max[k, i, j] = align_fn(
                    np.asarray(h, float), D_all[k], max_lag=max_lag)
            # cost with the predicted amplitude (no scalar), all traces at once
            Csp_all[:, i, j] = 0.5 * trace_w * np.sum((D_all - H_all) ** 2, axis=1)
            s_hat[i, j] = scale_hat_time(D_all, H_all, w=trace_w)   # diagnostic only

    # ---- pack (same field names as cnct_searchR_obs; a = 1, no scalar in the cost)
    if is_ratio:
        out = _ratio_grid_pack(
            Csp_all, s_hat, p_ax, ct,
            (np.broadcast_to(ct[None, :], (p_ax.size, ct.size)) if is_rcn
             else p_ax[:, None] * ct[None, :]), idx_array, 1.0,
            lag_samples=lag_samples,
        )
        if is_rcn:                                      # 'ct' slot held the cn axis: relabel
            out = _ratio_cn_pack(out)
    else:
        out = cnct_search_pack(
            Csp_all, s_hat, p_ax, ct, idx_array, 1.0,
            axes=axes, mode="R", domain="time", lag_samples=lag_samples, grid="cn_ct",
        )

    data_energy = float(0.5 * np.sum(trace_w * np.sum(D_all ** 2, axis=1)))
    out.update({
        "idx_input": idx_input,
        "idx_dropped": idx_dropped,
        "trace_w": trace_w,
        "trace_opts": trace_opts,
        "snr_pairs": snr_pairs,                         # pair SNR of idx_input (None if not used)
        "cc_max": cc_max,
        "data_energy": data_energy,
        "cost_norm": out["Csp_sum"] / data_energy,
        "search_mode": search_mode,
        "eps": float(eps),
        "cost": "0.5 w (d - h)^2, no s_hat",
    })
    return out


def _node_cnct(out):
    """
    Actual (cn, ct) at every node of a search result, for its grid type.
    Input: out (search result dict). Output: (CN, CT), each (n_p, n_q).
    """
    grid = out.get("grid")
    if grid == "ratio_cn":                                  # axes (cn/ct, cn)
        r = np.asarray(out["cn_ratio"], float)[:, None]
        CN = np.broadcast_to(np.asarray(out["cn"], float)[None, :], (r.size, len(out["cn"])))
        return CN, CN / r
    if "cn_ratio" in out:                                   # axes (cn/ct, ct)
        r = np.asarray(out["cn_ratio"], float)[:, None]
        CT = np.broadcast_to(np.asarray(out["ct"], float)[None, :], (r.size, len(out["ct"])))
        return r * CT, CT
    CN, CT = np.meshgrid(np.asarray(out["cn"], float), np.asarray(out["ct"], float), indexing="ij")
    return CN, CT                                           # axes (cn, ct)


def optimal_angle_groups(out, data, n_groups_max=4, min_size=3, min_gain=0.05, verbose=True):
    """
    Purpose
    -------
    Objective grouping of the fitted pairs (traces) of a saved compliance
    search by incidence angle, from the search's own per-pair costs
    (no re-run). Pairs are ordered by angle and split into K CONTIGUOUS
    groups; each group takes its own best node (the minimum of the sum of
    its pairs' cost maps, out['Csp']). For every K = 1..n_groups_max the
    split minimizing the total cost
        J(K) = sum_groups  min_node  sum_{i in group} Csp_i(node)
    is found exactly by dynamic programming. More groups always lower J, so
    the recommended K is the largest one whose improvement over K-1,
    gain(K) = (J(K-1) - J(K)) / J(K-1), is >= min_gain.

    Input
    -----
    out : dict, tuple or path
        Search result of cnct_search_RfromT or cnct_searchR_obs: the run
        folder (reads <folder>/<folder>.pkl), a .pkl path or the loaded
        object; for a constrained (out_full, out_constraint) tuple the full
        pack is used. Needs 'Csp', 'idx', 'trace_w' and the grid axes.
    data : dict
        scat_data (RfromT) or ref_data (reflection only) used for the run:
        needs 'theta_i_deg' and 'reflection_traces' (to normalize the costs
        by each pair's data energy 0.5 w sum d^2, as the search's cost_norm).
    n_groups_max : int
        Largest number of groups tried (default 4).
    min_size : int
        Minimum number of pairs per group (default 3).
    min_gain : float
        Minimum relative cost improvement for an extra group to be
        recommended (default 0.05 = 5%).
    verbose : bool
        Print one table per K and the recommended K.

    Output
    ------
    dict {'K': {K: {'groups': [pairs_1, ..., pairs_K]}}} for K = 1..n_groups_max
        pairs_g : array of the pair indices (as in out['idx']) of group g,
                  groups ordered from the largest angles.
        e.g. res['K'][2]['groups'][0] -> pairs of the first of 2 groups; its best
        node is read from the search result: out['Csp'][rows].sum(0), rows =
        positions of pairs_g in out['idx'].
    Costs, gains, best nodes and the recommended K are printed only (verbose).
    """
    import pickle as _pickle
    if isinstance(out, (str, bytes)) or hasattr(out, "__fspath__"):
        import os
        p = os.fspath(out)
        if os.path.isdir(p):                               # run folder -> <folder>/<folder>.pkl
            p = os.path.join(p, os.path.basename(os.path.normpath(p)) + ".pkl")
        with open(p, "rb") as f:
            out = _pickle.load(f)
    if isinstance(out, tuple):                             # constrained run: use the full pack
        out = out[0]

    idx = np.asarray(out["idx"], int)
    theta = np.asarray(data["theta_i_deg"], float)[idx]
    D = np.asarray(data["reflection_traces"].data, float)[idx]
    E = 0.5 * np.asarray(out["trace_w"], float) * np.sum(D ** 2, axis=1)   # per-pair data energy
    Csp = np.asarray(out["Csp"], float)                   # (n_pairs, n_p, n_q)
    CN, CT = _node_cnct(out)

    order = np.argsort(-theta)                             # largest angle first
    n = order.size
    if n_groups_max * min_size > n:
        n_groups_max = n // min_size
    # prefix sums along the angle order -> any contiguous group's cost map in O(1)
    S = np.concatenate([np.zeros((1,) + Csp.shape[1:]), np.cumsum(Csp[order], axis=0)])
    SE = np.concatenate([[0.0], np.cumsum(E[order])])
    seg = np.full((n + 1, n + 1), np.inf)                  # best cost of group order[a:b]
    for a in range(n):
        for b in range(a + min_size, n + 1):
            seg[a, b] = (S[b] - S[a]).min()

    # dynamic programming: best[k, b] = min_a best[k-1, a] + seg[a, b]
    best = np.full((n_groups_max + 1, n + 1), np.inf)
    arg = np.zeros((n_groups_max + 1, n + 1), int)
    best[0, 0] = 0.0
    for k in range(1, n_groups_max + 1):
        for b in range(k * min_size, n + 1):
            cand = best[k - 1, :b] + seg[:b, b]
            arg[k, b] = int(np.argmin(cand))
            best[k, b] = cand[arg[k, b]]

    result, prev = {}, None
    for k in range(1, n_groups_max + 1):
        if not np.isfinite(best[k, n]):
            break
        cuts, b = [], n                                    # back-track the group boundaries
        for kk in range(k, 0, -1):
            a = arg[kk, b]
            cuts.append((a, b))
            b = a
        groups = []
        for a, b in reversed(cuts):
            C = S[b] - S[a]                                # group cost map
            i, j = np.unravel_index(np.argmin(C), C.shape)
            near = C <= 1.10 * C[i, j]                     # nodes within 10% of the minimum
            rows = order[a:b]
            groups.append({
                "pairs": idx[rows],
                "angle": (float(theta[rows].min()), float(theta[rows].max())),
                "cn": float(CN[i, j]), "ct": float(CT[i, j]), "ratio": float(CN[i, j] / CT[i, j]),
                "cost_norm": float(C[i, j] / (SE[b] - SE[a])),
                "cn_range": (float(CN[near].min()), float(CN[near].max())),
            })
        J = best[k, n]
        gain = np.nan if prev is None else (prev - J) / prev
        result[k] = {"cost_norm": float(J / SE[n]), "gain": float(gain), "groups": groups}
        prev = J

    rec = 1
    for k in sorted(result)[1:]:                           # largest K with gain >= min_gain
        if result[k]["gain"] >= min_gain:
            rec = k
        else:
            break

    if verbose:
        for k, r in result.items():
            g_txt = "" if k == 1 else f", gain vs K={k - 1}: {r['gain']:.1%}"
            print(f"K={k}: total cost_norm {r['cost_norm']:.3f}{g_txt}"
                  + ("   <- recommended" if k == rec else ""))
            for g in r["groups"]:
                print(f"   angle {g['angle'][1]:4.1f}-{g['angle'][0]:4.1f} deg  pairs {g['pairs'].min()}-"
                      f"{g['pairs'].max()} (n={g['pairs'].size})  best cn {g['cn']:.2e} "
                      f"(10%: {g['cn_range'][0]:.2e}-{g['cn_range'][1]:.2e})  ct {g['ct']:.2e}  "
                      f"cn/ct {g['ratio']:.2f}  cost_norm {g['cost_norm']:.2f}")
    return {"K": {k: {"groups": [g["pairs"] for g in r["groups"]]} for k, r in result.items()}}


def extracting_search(run_dir, groups, data):
    """
    Purpose
    -------
    Recreate the global (full, unconstrained) search result of a saved
    cnct_search_RfromT / cnct_searchR_obs run for subsets (groups) of its
    pairs, WITHOUT re-running the search: every pair's cost map out['Csp']
    is saved, so a group's cost map is the sum over its pairs. Each group is
    saved as a search result with the same keys/format as the global one, so
    plot_cnct_search / pick_ratiosearch read it as any run.

    Input
    -----
    run_dir : str or path
        Run folder <out_root>/<name>/ holding <name>.pkl (as written by
        cnct_search_scat / cnct_search). A constrained (full, constraint)
        result: only the full pack is used.
    groups : list of array-like
        Pair indices (as in out['idx']) of each group, e.g.
        optimal_angle_groups(...)['K'][2]['groups']. Every index must be a
        pair kept in the search.
    data : dict
        scat_data (RfromT) or ref_data used for the run; needs
        'reflection_traces' (per-pair data energy 0.5 w sum d^2 for the
        group's cost_norm).

    Output
    ------
    paths : list of str
        Saved files <run_dir>/group<g>_<name>.pkl, g = 0, 1, ... in the order
        of `groups`. Each holds the global-format dict restricted to the
        group:
          per-pair fields (Csp, cn_best, ct_best, ratio_best, Csp_min, i_best,
            j_best, sub100, lag_samples, cc_max, trace_w, idx, geom_idx):
            the group's rows; index = 0..n_g-1
          recomputed: Csp_sum, Csp_mean, best node (i_sum_best, j_sum_best,
            cn_sum_best, ct_sum_best, ratio_sum_best, Csp_sum_min),
            data_energy, cost_norm, idx_input (group pairs), idx_dropped
            (empty), snr_pairs (group pairs)
          unchanged: grid axes and run settings (cn, ct / cn_ratio, cn_grid,
            ct_grid, grid, trace_opts, eps, cost, ...)
          added: 'group' (g), 'source_run' (<name>)
          removed: s_hat, s_hat_best (diagnostic, needs the models)
    """
    import os
    import copy
    import pickle as _pickle
    run_dir = os.path.normpath(os.fspath(run_dir))
    name = os.path.basename(run_dir)
    with open(os.path.join(run_dir, name + ".pkl"), "rb") as f:
        out = _pickle.load(f)
    if isinstance(out, tuple):                             # constrained run: keep the global pack
        out = out[0]

    idx = np.asarray(out["idx"], int)
    n_pairs = idx.size
    pos = {int(p): r for r, p in enumerate(idx)}           # pair index -> row in the per-pair arrays
    D = np.asarray(data["reflection_traces"].data, float)
    CN, CT = _node_cnct(out)                               # actual (cn, ct) at every node
    per_pair = ("Csp", "cn_best", "ct_best", "ratio_best", "Csp_min", "i_best", "j_best",
                "sub100", "lag_samples", "cc_max", "trace_w", "idx", "geom_idx")

    paths = []
    for g, pairs in enumerate(groups):
        pairs = np.asarray(pairs, int).ravel()
        missing = [int(p) for p in pairs if int(p) not in pos]
        if missing:
            raise ValueError(f"group {g}: pairs {missing} are not in the search (out['idx'])")
        rows = np.array([pos[int(p)] for p in pairs])

        o = {k: copy.deepcopy(v) for k, v in out.items() if k not in ("s_hat", "s_hat_best")}
        for k in per_pair:                                 # keep the group's rows only
            if o.get(k) is not None and np.ndim(o[k]) >= 1 and np.shape(o[k])[0] == n_pairs:
                o[k] = np.asarray(out[k])[rows]
        o["index"] = np.arange(rows.size)

        # group cost map and its best node
        C = o["Csp"].sum(0)
        i, j = np.unravel_index(np.argmin(C), C.shape)
        o.update({"Csp_sum": C, "Csp_mean": o["Csp"].mean(0), "Csp_sum_min": float(C[i, j]),
                  "i_sum_best": int(i), "j_sum_best": int(j),
                  "cn_sum_best": float(CN[i, j]), "ct_sum_best": float(CT[i, j])})
        if "ratio_sum_best" in out:
            o["ratio_sum_best"] = float(CN[i, j] / CT[i, j])
        if out.get("i_true") is not None and "Csp_sum_true" in out:
            o["Csp_sum_true"] = float(C[out["i_true"], out["j_true"]])

        # normalization by the group's data energy (same definition as the search)
        energy = float(0.5 * np.sum(o["trace_w"] * np.sum(D[pairs] ** 2, axis=1)))
        o.update({"data_energy": energy, "cost_norm": C / energy,
                  "idx_input": pairs.copy(), "idx_dropped": np.array([], int),
                  "group": g, "source_run": name})
        if out.get("snr_pairs") is not None:               # snr_pairs follows idx_input
            snr_in = dict(zip(np.asarray(out["idx_input"], int).tolist(), np.asarray(out["snr_pairs"])))
            o["snr_pairs"] = np.array([snr_in[int(p)] for p in pairs])

        path = os.path.join(run_dir, f"group{g}_{name}.pkl")
        with open(path, "wb") as f:
            _pickle.dump(o, f)
        paths.append(path)
    return paths


def waveform_fitness(run_dir, scat_data, groups=None, medium=None, search_param=None):
    """
    Purpose
    -------
    Waveform similarity of the best fits of a saved cnct_search_scat /
    cnct_search_RfromT run, per group of pairs (no re-run of the search).
    Each group's best node is the argmin of the sum of its pairs' cost maps,
    out['Csp'][rows].sum(0) (same node as extracting_search). At that node
    the modeled reflection h_i of every pair is rebuilt (RfromT, aligned to
    the data d_i with align_by_window_cc, as in the search) and compared
    with d_i over the whole trace (the samples the cost uses):
        rho_i = d_i.h_i / (|d_i| |h_i|)                shape match (cosine)
        amp_i = |h_i| / |d_i|                          model / data amplitude
        S_i   = 2 d_i.h_i / (|d_i|^2 + |h_i|^2)        shape AND amplitude
              = rho_i * 2 amp_i / (1 + amp_i^2)        (Lin's concordance
                                                        coefficient, zero-mean)
        S_mean = mean_i S_i                            every pair counts equally
        S_pool = 2 sum_i d_i.h_i / sum_i (|d_i|^2 + |h_i|^2)
                                                       energy-weighted, like cost_norm
    S = 1: same shape and amplitude; under- and over-prediction by the same
    factor give the same S (unlike the L2 cost).

    Input
    -----
    run_dir : str or path
        Run folder <out_root>/<name>/ holding <name>.pkl and <name>_data.pkl
        (as written by cnct_search_scat).
    scat_data : dict
        scat_data prepared EXACTLY as for the run (fft_traces_scat, Gr, Gt,
        omega, the same low-pass of the transmitted traces).
    groups : list of array-like or None
        Pair indices (as in out['idx']) of each group, e.g.
        optimal_angle_groups(...)['K'][4]['groups']. None = one group with
        all the run's pairs (the global best node).
    medium : (vp, vs, rho, wave) or None
        None = the run's own medium (from <name>_data.pkl).
    search_param : tuple or None
        None = the run's own search_param (eps = [2], cc = [5], max_lag = [8]).

    Output
    ------
    dict
        'run' : run name
        'groups' : list, one dict per group (order of `groups`):
            'pairs' (n,), 'angle' (min, max) [deg], best node 'cn', 'ct',
            'ratio' (cn/ct), 'cost_norm' (group cost at its best node),
            per pair (n,): 'rho', 'amp', 'S', 'E' (|d|^2 + |h|^2),
            'misfit2' (|d - h|^2), 'w' (trace weight of the run),
            group: 'S_mean', 'S_pool', 'rho_mean', 'amp_mean'
    """
    import os
    import pickle as _pickle
    run_dir = os.path.normpath(os.fspath(run_dir))
    name = os.path.basename(run_dir)
    with open(os.path.join(run_dir, name + ".pkl"), "rb") as f:
        out = _pickle.load(f)
    if isinstance(out, tuple):                             # constrained pack: use the full one
        out = out[0]
    with open(os.path.join(run_dir, name + "_data.pkl"), "rb") as f:
        run_in = _pickle.load(f)
    vp, vs, rho_m, wave = run_in["medium"] if medium is None else medium
    sp = run_in["search_param"] if search_param is None else search_param
    eps, cc = float(sp[2]), sp[5]
    max_lag = sp[8] if len(sp) >= 9 else None

    idx = np.asarray(out["idx"], int)
    pos = {int(p): r for r, p in enumerate(idx)}           # pair index -> row of out['Csp']
    if groups is None:
        groups = [idx]
    CN, CT = _node_cnct(out)                               # actual (cn, ct) at every node
    rr = scat_data["reflection_traces"]
    D = np.asarray(rr.data, float)
    taxis = np.asarray(rr.taxis, float)
    tw = scat_data["time_window"]
    theta = np.asarray(scat_data["theta_i_deg"], float)
    Csp = np.asarray(out["Csp"], float)
    w_all = np.asarray(out["trace_w"], float)

    res = []
    for g, pairs in enumerate(groups):
        pairs = np.asarray(pairs, int).ravel()
        missing = [int(p) for p in pairs if int(p) not in pos]
        if missing:
            raise ValueError(f"group {g}: pairs {missing} are not in the search (out['idx'])")
        rows = np.array([pos[int(p)] for p in pairs])
        C = Csp[rows].sum(0)                               # group cost map
        i, j = np.unravel_index(np.argmin(C), C.shape)     # group best node
        cn, ct = float(CN[i, j]), float(CT[i, j])

        # modeled reflection of every pair at the best node, aligned as in the search
        Dg = D[pairs]
        H = np.array([align_by_window_cc(
            np.asarray(RfromT(scat_data, int(p), vp, vs, rho_m, cn, ct, eps, wave)["Er_time"], float),
            Dg[k], taxis=taxis, time_window=tw, cc=cc, max_lag=max_lag)[0]
            for k, p in enumerate(pairs)])

        dh = np.sum(Dg * H, 1)                             # d.h per pair
        dd, hh = np.sum(Dg ** 2, 1), np.sum(H ** 2, 1)     # |d|^2, |h|^2
        rho = dh / np.sqrt(dd * hh)
        amp = np.sqrt(hh / dd)
        S = 2 * dh / (dd + hh)
        w = w_all[rows]
        res.append({
            "pairs": pairs, "angle": (float(theta[pairs].min()), float(theta[pairs].max())),
            "cn": cn, "ct": ct, "ratio": cn / ct,
            "cost_norm": float(C[i, j] / np.sum(0.5 * w * dd)),   # as the search's cost_norm
            "rho": rho, "amp": amp, "S": S, "E": dd + hh,
            "misfit2": np.sum((Dg - H) ** 2, 1), "w": w,
            "S_mean": float(S.mean()), "S_pool": float(2 * dh.sum() / (dd + hh).sum()),
            "rho_mean": float(rho.mean()), "amp_mean": float(amp.mean()),
        })
    return {"run": name, "groups": res}


def geometric_factor(scattering_data, gauge_length, velocity, alpha=0.5):
    """
    Frequency-dependent geometric factors G_R(ω) and G_T(ω).

        G(ω) = (n·p)(n·k̂) * sinc[(ω g / 2c) (n·k̂)] / d

    Parameters
    ----------
    scattering_data : dict
        Needs reflection_traces, transmitted_traces,
        n_p_ref_dot, n_u_ref_dot, dist_source_rr,
        n_p_trans_dot, n_u_trans_dot, dist_source_tr
        (keys of data/scattering_data.pkl from DASscatData.ipynb), and
        freqs: frequency axis [Hz], FFT order (np.fft.fftfreq), added by
        fft_traces_scat. Used for Gr, Gt, and omega = 2π freqs.
    gauge_length : float
        DAS gauge length g [m].
    velocity : float
        Wave speed c [m/s] (e.g. vs for SS).
    alpha : float
        Tukey taper for windowed_sinc.
    """
    g = float(gauge_length)
    c = float(velocity)
    if g <= 0 or c <= 0:
        raise ValueError("gauge_length and velocity must be > 0")

    if "freqs" not in scattering_data:
        raise KeyError("scattering_data needs 'freqs' (run fft_traces_scat first)")
    freqs = np.asarray(scattering_data["freqs"], float)   # common frequency axis [Hz]
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("freqs must be a 1-D non-empty array [Hz]")

    DAS_rr = scattering_data["reflection_traces"]
    DAS_tr = scattering_data["transmitted_traces"]

    n_p_r = np.asarray(scattering_data["n_p_ref_dot"], float).ravel()
    n_u_r = np.asarray(scattering_data["n_u_ref_dot"], float).ravel()
    d_r = np.asarray(scattering_data["dist_source_rr"], float).ravel()   # source -> fracture -> reflection receiver [m]

    n_p_t = np.asarray(scattering_data["n_p_trans_dot"], float).ravel()
    n_u_t = np.asarray(scattering_data["n_u_trans_dot"], float).ravel()
    d_t = np.asarray(scattering_data["dist_source_tr"], float).ravel()   # source -> transmission receiver [m]

    n_rr = int(np.asarray(DAS_rr.data).shape[0])
    n_tr = int(np.asarray(DAS_tr.data).shape[0])

    if not (n_p_r.size == n_u_r.size == d_r.size):
        raise ValueError(
            "reflection geometry sizes differ: "
            f"n_p_ref_dot={n_p_r.size}, n_u_ref_dot={n_u_r.size}, dist_source_rr={d_r.size}"
        )
    if not (n_p_t.size == n_u_t.size == d_t.size):
        raise ValueError(
            "transmission geometry sizes differ: "
            f"n_p_trans_dot={n_p_t.size}, n_u_trans_dot={n_u_t.size}, dist_source_tr={d_t.size}"
        )
    if n_p_r.size != n_p_t.size:
        raise ValueError(
            f"geometry length mismatch: reflection={n_p_r.size}, transmission={n_p_t.size}"
        )
    n_geom = n_p_r.size

    if n_rr != n_tr:
        raise ValueError(f"n_reflection_DAS={n_rr} != n_transmission_DAS={n_tr}")
    if n_rr > n_geom:
        raise ValueError(f"DAS traces ({n_rr}) exceed geometry length ({n_geom})")

    Gr = np.zeros((n_geom, freqs.size), dtype=float)
    Gt = np.zeros((n_geom, freqs.size), dtype=float)
    arg_r = np.zeros(n_geom, dtype=float)
    arg_t = np.zeros(n_geom, dtype=float)

    for i in range(n_geom):
        if d_r[i] == 0.0:
            raise ValueError(f"dist_source_rr[{i}] is zero")
        if n_u_r[i] == 0.0:
            raise ValueError(f"n_u_ref_dot[{i}] is zero (sinc arg undefined)")
        arg_r[i] = g * n_u_r[i] / (2.0 * c)
        sinc_r = windowed_sinc(freqs, arg_r[i], alpha=alpha)
        Gr[i, :] = (n_p_r[i] * n_u_r[i] * sinc_r) / d_r[i]

        if d_t[i] == 0.0:
            raise ValueError(f"dist_source_tr[{i}] is zero")
        if n_u_t[i] == 0.0:
            raise ValueError(f"n_u_trans_dot[{i}] is zero (sinc arg undefined)")
        arg_t[i] = g * n_u_t[i] / (2.0 * c)
        sinc_t = windowed_sinc(freqs, arg_t[i], alpha=alpha)
        Gt[i, :] = (n_p_t[i] * n_u_t[i] * sinc_t) / d_t[i]

    return {
        "Gr": Gr,
        "Gt": Gt,
        "freqs": freqs,
        "omega": 2.0 * np.pi * freqs,
        "arg_r": arg_r,
        "arg_t": arg_t,
        "gauge_length": g,
        "velocity": c,
        "alpha": float(alpha),
        "n_pairs": n_geom,
        "n_das": n_rr,
    }

def RT_wave(
    scattering_data,
    inc_wave,
    idx,
    vp,
    vs,
    rho,
    cn,
    ct,
    wave="SS",
):
    """
    Forward transmitted / reflected DAS spectra from incident wavelet S(ω).

        ET(ω) = (ω² / vs) * Gt(ω) * S(ω) * T(ω)
        ER(ω) = (ω² / vs) * Gr(ω) * S(ω) * R(ω)

    R, T from RPS_coef (same as estimate_inc_wavelet).
    """
    req1 = ("Gt", "Gr", "omega", "theta_t_deg")
    missing = [k for k in req1 if k not in scattering_data]
    if missing:
        raise ValueError(f"scattering_data missing keys: {missing}")

    req2 = ("S_omega", "time", "freqs")
    missing = [k for k in req2 if k not in inc_wave]
    if missing:
        raise ValueError(f"inc_wave missing keys: {missing}")

    idx = int(idx)
    Gt = np.asarray(scattering_data["Gt"])
    Gr = np.asarray(scattering_data["Gr"])
    omega = np.asarray(scattering_data["omega"], float)
    theta_t = np.asarray(scattering_data["theta_t_deg"], float)

    if not (0 <= idx < Gt.shape[0]):
        raise ValueError(f"idx={idx} out of range for Gt (rows={Gt.shape[0]})")
    if idx >= Gr.shape[0]:
        raise ValueError(f"idx={idx} out of range for Gr (rows={Gr.shape[0]})")
    if idx >= theta_t.size:
        raise ValueError(f"idx={idx} out of range for theta_t_deg (n={theta_t.size})")

    S = np.asarray(inc_wave["S_omega"])
    freqs = np.asarray(inc_wave["freqs"], float)
    taxis = np.asarray(inc_wave["time"], float)
    GT = np.asarray(Gt[idx], float)
    GR = np.asarray(Gr[idx], float)

    if S.shape != freqs.shape or GT.shape != freqs.shape or GR.shape != freqs.shape:
        raise ValueError(
            "length mismatch among S / Gt / Gr / freqs: "
            f"{S.shape}, {GT.shape}, {GR.shape}, {freqs.shape}"
        )
    if omega.shape != freqs.shape:
        raise ValueError("scattering_data['omega'] length must match inc_wave['freqs']")

    theta_i = np.deg2rad(theta_t[idx])
    ps = np.sin(theta_i) / float(vs)

    R, T = RPS_coef(vp, vs, ps, rho, omega, cn, ct, wave)

    scale = (omega**2) / float(vs)
    Et_omega = scale * GT * S * T
    Er_omega = scale * GR * S * R

    Et_time = real_wave(Et_omega)
    Er_time = real_wave(Er_omega)

    return {
        "Et_omega": Et_omega,
        "Et_time": Et_time,
        "Er_omega": Er_omega,
        "Er_time": Er_time,
        "time": taxis,
        "freqs": freqs,
        "idx": idx,
        "angle": theta_t[idx],
    }


def RT_wave_batch(
    scattering_data,
    inc_wave,
    idx_array,
    vp,
    vs,
    rho,
    cn,
    ct,
    wave="SS",
):
    """
    Run RT_wave for many indices with a shared incident wavelet.

    Returns dict of arrays with leading axis = len(idx_array).
    """
    idx_array = np.asarray(idx_array, int).ravel()
    n = idx_array.size
    if n == 0:
        raise ValueError("idx_array is empty")

    Et_omega = Er_omega = Et_time = Er_time = None
    angles = np.empty(n, float)

    for k, idx in enumerate(idx_array):
        out = RT_wave(
            scattering_data, inc_wave, idx, vp, vs, rho, cn, ct, wave=wave,
        )
        if k == 0:
            nf = out["Et_omega"].size
            nt = out["Et_time"].size
            Et_omega = np.empty((n, nf), dtype=complex)
            Er_omega = np.empty((n, nf), dtype=complex)
            Et_time = np.empty((n, nt), dtype=float)
            Er_time = np.empty((n, nt), dtype=float)
            freqs = np.asarray(out["freqs"])
            taxis = np.asarray(out["time"])

        Et_omega[k] = out["Et_omega"]
        Er_omega[k] = out["Er_omega"]
        Et_time[k] = out["Et_time"]
        Er_time[k] = out["Er_time"]
        angles[k] = out["angle"]

    return {
        "Et_omega": Et_omega,
        "Et_time": Et_time,
        "Er_omega": Er_omega,
        "Er_time": Er_time,
        "freqs": freqs,
        "time": taxis,
        "idx": idx_array,
        "angle": angles,
    }
def reflection_from_RTmodel(
    rt_batch,
    scattering_data,
    vp,
    vs,
    rho,
    cn,
    ct,
    wave="SS",
    eta=1e-2,
):
    """
    Apply estimate_reflection to every trace in an RT_wave_batch result.

    Input data (from rt_batch):
        trans_data_fft <- Et_omega[k]
        rr_data_trace  <- Er_time[k]

    Returns estimate_reflection fields (stacked) plus the input rt_batch arrays.
    """
    req_rt = ("Et_omega", "Er_time", "idx", "freqs", "time")
    missing = [k for k in req_rt if k not in rt_batch]
    if missing:
        raise ValueError(f"rt_batch missing keys: {missing}")

    req_sd = ("Gr", "Gt", "omega", "theta_t_deg")
    missing = [k for k in req_sd if k not in scattering_data]
    if missing:
        raise ValueError(f"scattering_data missing keys: {missing}")

    idx_array = np.asarray(rt_batch["idx"], int).ravel()
    Et_omega = np.asarray(rt_batch["Et_omega"])
    Er_time = np.asarray(rt_batch["Er_time"])
    n = idx_array.size

    if Et_omega.shape[0] != n or Er_time.shape[0] != n:
        raise ValueError("rt_batch Et_omega / Er_time leading size must match idx")

    Gr = np.asarray(scattering_data["Gr"])
    Gt = np.asarray(scattering_data["Gt"])
    omega = np.asarray(scattering_data["omega"], float)
    theta_t = np.asarray(scattering_data["theta_t_deg"], float)
    freqs = np.asarray(rt_batch["freqs"], float)
    taxis = np.asarray(rt_batch["time"], float)

    ER_fft = np.empty_like(Et_omega, dtype=complex)
    ER_time = np.empty_like(Er_time, dtype=float)
    ER_time_cc = np.empty_like(Er_time, dtype=float)

    lags = np.empty(n, dtype=int)
    angles = np.empty(n, dtype=float)

    for k, idx in enumerate(idx_array):
        if not (0 <= idx < Gr.shape[0] and idx < Gt.shape[0]):
            raise ValueError(f"idx[{k}]={idx} out of range for Gr/Gt")
        if idx >= theta_t.size:
            raise ValueError(f"idx[{k}]={idx} out of range for theta_t_deg")

        GR = np.asarray(Gr[idx], float)
        GT = np.asarray(Gt[idx], float)
        ps = np.sin(np.deg2rad(theta_t[idx])) / float(vs)

        est = estimate_reflection(
            Et_omega[k],
            Er_time[k],
            GR,
            GT,
            vp,
            vs,
            ps,
            rho,
            omega,
            cn,
            ct,
            wave=wave,
            eta=eta,
        )

        ER_fft[k] = est["reflection_model_fft"]
        ER_time[k] = est["reflection_model_time"]
        ER_time_cc[k] = est["reflection_model_time_cc"]
        lags[k] = est["cc_lag_samples"]
        angles[k] = float(theta_t[idx])

    return {
        # estimate_reflection outputs (stacked)
        "reflection_model_fft": ER_fft,
        "reflection_model_time": ER_time,
        "reflection_model_time_cc": ER_time_cc,
        "cc_lag_samples": lags,
        "angle": angles,
        "idx": idx_array,
        "freqs": freqs,
        "time": taxis,
        # input data from RT_wave_batch
        "Et_omega": Et_omega,
        "Et_time": np.asarray(rt_batch["Et_time"]),
        "Er_omega": np.asarray(rt_batch["Er_omega"]),
        "Er_time": Er_time,
    }

def grid_seach_helper(
    rt_batch,
    index,
    search_space=((1e-14, 1e-8), (0.01, 3.0)),
    nsearch=(30, 30),
    true_cnct=None,
):
    """
    search_space : [(cn_min, cn_max), (cn_ratio_min, cn_ratio_max)]
    nsearch : (n_cn, n_ratio)
    true_cnct : (cn, ct) or None
        Inserts cn and ratio=cn/ct into the axes.

    At each grid node: ct = cn / cn_ratio  (cn_ratio = cn/ct).
    Cost arrays use Cw[i, j] <-> cn_ratio[i], cn[j].
    """
    try:
        n_cn, n_r = int(nsearch[0]), int(nsearch[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("nsearch must be a tuple (n_cn, n_ratio)") from exc
    if n_cn < 2 or n_r < 2:
        raise ValueError("n_cn and n_ratio must be >= 2")

    try:
        (cn_min, cn_max), (r_min, r_max) = search_space
        cn_min, cn_max = float(cn_min), float(cn_max)
        r_min, r_max = float(r_min), float(r_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "search_space must be [(cn_min, cn_max), (cn_ratio_min, cn_ratio_max)]"
        ) from exc
    if not (cn_min > 0 and cn_max > cn_min):
        raise ValueError("need 0 < cn_min < cn_max")
    if not (r_min > 0 and r_max > r_min):
        raise ValueError("need 0 < cn_ratio_min < cn_ratio_max")

    Et_omega = np.asarray(rt_batch["Et_omega"])
    Er_omega = np.asarray(rt_batch["Er_omega"])
    Er_time = np.asarray(rt_batch["Er_time"])
    idx_array = np.asarray(rt_batch["idx"], int).ravel()
    n_tr = idx_array.size

    index = int(index)
    if not (0 <= index < n_tr):
        raise ValueError(f"index={index} out of range [0, {n_tr})")

    rt_k = {
        "Et_omega": Et_omega[index:index + 1],
        "Et_time": np.asarray(rt_batch["Et_time"])[index:index + 1],
        "Er_omega": Er_omega[index:index + 1],
        "Er_time": Er_time[index:index + 1],
        "idx": idx_array[index:index + 1],
        "freqs": rt_batch["freqs"],
        "time": rt_batch["time"],
    }

    cn = np.logspace(np.log10(cn_min), np.log10(cn_max), n_cn)
    cn_ratio = np.logspace(np.log10(r_min), np.log10(r_max), n_r)

    i_true = j_true = None
    true_pair = None
    ratio_true = None
    if true_cnct is not None:
        cn_t, ct_t = map(float, true_cnct)
        if cn_t <= 0 or ct_t <= 0:
            raise ValueError("true_cnct values must be > 0")
        ratio_true = cn_t / ct_t
        cn = np.unique(np.sort(np.append(cn, cn_t)))
        cn_ratio = np.unique(np.sort(np.append(cn_ratio, ratio_true)))
        true_pair = (cn_t, ct_t)
        i_true = int(np.argmin(np.abs(np.log10(cn_ratio) - np.log10(ratio_true))))
        j_true = int(np.argmin(np.abs(np.log10(cn) - np.log10(cn_t))))

    return {
        "rt_k": rt_k,
        "cn": cn,
        "cn_ratio": cn_ratio,
        "true_cnct": true_pair,
        "ratio_true": ratio_true,
        "i_true": i_true,
        "j_true": j_true,
    }

def cnct_ratio_search_time(
    rt_batch,
    scattering_data,
    vp,
    vs,
    rho,
    search_space=((1e-14, 1e-8), (0.01, 3.0)),
    nsearch=(30, 30),
    true_cnct=None,
    wave="SS",
    eta=1e-2,
):
    """
    Time-domain (cn, cn/ct) compliance search for every trace in rt_batch.

    search_space : [(cn_min, cn_max), (cn_ratio_min, cn_ratio_max)]
    nsearch : (n_cn, n_ratio)
    At each node: ct = cn / cn_ratio

    true_cnct : (cn, ct) or None → inserted via grid_seach_helper
    """
    idx_array = np.asarray(rt_batch["idx"], int).ravel()
    n_tr = idx_array.size
    if n_tr == 0:
        raise ValueError("rt_batch has no traces")

    Er_time = np.asarray(rt_batch["Er_time"])

    Cw_all = None
    ct_best = np.empty(n_tr, float)
    cn_best = np.empty(n_tr, float)
    Cw_min = np.empty(n_tr, float)
    i_best = np.empty(n_tr, int)
    j_best = np.empty(n_tr, int)
    geom_idx = np.empty(n_tr, int)
    index_arr = np.arange(n_tr, dtype=int)
    sub100_all = None
    cn = cn_ratio = None
    i_true = j_true = None
    true_pair = None
    ratio_true = None

    for k in range(n_tr):
        axes = grid_seach_helper(
            rt_batch, k, search_space, nsearch, true_cnct=true_cnct,
        )
        rt_k = axes["rt_k"]
        cn = axes["cn"]
        cn_ratio = axes["cn_ratio"]
        i_true = axes["i_true"]
        j_true = axes["j_true"]
        true_pair = axes["true_cnct"]
        ratio_true = axes["ratio_true"]

        n_cn = int(cn.size)
        n_r = int(cn_ratio.size)
        Cw = np.empty((n_r, n_cn), dtype=float)
        d = Er_time[k]

        for j, cn_j in enumerate(cn):
            for i, r in enumerate(cn_ratio):
                ct_j = cn_j / r
                est = reflection_from_RTmodel(
                    rt_k, scattering_data, vp, vs, rho, cn_j, ct_j,
                    wave=wave, eta=eta,
                )
                Cw[i, j] = C_wavelet(
                    d, est["reflection_model_time_cc"][0],
                )

        ii0, jj0 = np.unravel_index(np.argmin(Cw), Cw.shape)
        cn_b = float(cn[jj0])
        r_b = float(cn_ratio[ii0])
        ct_b = cn_b / r_b

        flat = Cw.ravel()
        n_top = min(100, flat.size)
        idx_flat = np.argpartition(flat, n_top - 1)[:n_top]
        idx_flat = idx_flat[np.argsort(flat[idx_flat])]
        ii, jj = np.unravel_index(idx_flat, Cw.shape)
        sub100 = np.column_stack([
            cn[jj],
            cn[jj] / cn_ratio[ii],
            flat[idx_flat],
        ])

        if k == 0:
            Cw_all = np.empty((n_tr, n_r, n_cn), float)
            sub100_all = np.empty((n_tr, n_top, 3), float)

        Cw_all[k] = Cw
        ct_best[k] = ct_b
        cn_best[k] = cn_b
        Cw_min[k] = float(Cw[ii0, jj0])
        i_best[k] = int(ii0)
        j_best[k] = int(jj0)
        geom_idx[k] = int(rt_k["idx"][0])
        sub100_all[k] = sub100

    Cw_sum = np.sum(Cw_all, axis=0)
    Cw_mean = np.mean(Cw_all, axis=0)

    i_sum_best, j_sum_best = np.unravel_index(np.argmin(Cw_sum), Cw_sum.shape)
    cn_sum_best = float(cn[j_sum_best])
    ratio_sum_best = float(cn_ratio[i_sum_best])
    ct_sum_best = cn_sum_best / ratio_sum_best
    Cw_sum_min = float(Cw_sum[i_sum_best, j_sum_best])

    out = {
        "Cw": Cw_all,                 # (n_tr, n_ratio, n_cn)
        "Cw_sum": Cw_sum,             # (n_ratio, n_cn)
        "Cw_mean": Cw_mean,
        "cn": cn,
        "cn_ratio": cn_ratio,
        "ct_sum_best": ct_sum_best,
        "cn_sum_best": cn_sum_best,
        "ratio_sum_best": ratio_sum_best,
        "Cw_sum_min": Cw_sum_min,
        "i_sum_best": int(i_sum_best),
        "j_sum_best": int(j_sum_best),
        "ct_best": ct_best,
        "cn_best": cn_best,
        "Cw_min": Cw_min,
        "i_best": i_best,
        "j_best": j_best,
        "sub100": sub100_all,         # [cn, ct, Cw]
        "index": index_arr,
        "geom_idx": geom_idx,
        "idx": idx_array,
        "true_cnct": true_pair,
        "ratio_true": ratio_true,
        "i_true": i_true,
        "j_true": j_true,
        "Cw_sum_true": None,
        "axes": ("cn_ratio", "cn"),
    }
    if true_pair is not None:
        out["Cw_sum_true"] = float(Cw_sum[i_true, j_true])
    return out

def cnct_search_RTomega(
    rt_batch,
    scattering_data,
    inc_wave,
    vp,
    vs,
    rho,
    mode="R",
    search_space=((1e-14, 1e-8), (0.01, 3.0)),
    nsearch=(30, 30),
    wave="SS",
):
    """
    Array (cn, ct) search fitting reflection or transmission spectra.

    Like array_compliance_search_frequency, but:
      - estimated waveforms from RT_wave
      - cost = C_spectrum_amplitude(D, E)

    mode : {"R", "T"}
      "R" — fit Er_omega (data: rt_batch["Er_omega"], model: syn["Er_omega"])
      "T" — fit Et_omega (data: rt_batch["Et_omega"], model: syn["Et_omega"])

    Observed D is the synthetic batch field for each row k.
    """
    mode = str(mode).upper()
    if mode not in ("R", "T"):
        raise ValueError('mode must be "R" or "T"')

    data_key = "Er_omega" if mode == "R" else "Et_omega"
    model_key = data_key

    if data_key not in rt_batch:
        raise ValueError(f"rt_batch missing '{data_key}' (required for mode={mode})")

    idx_array = np.asarray(rt_batch["idx"], int).ravel()
    n_tr = idx_array.size
    if n_tr == 0:
        raise ValueError("rt_batch has no traces")

    D_all = np.asarray(rt_batch[data_key])
    if D_all.shape[0] != n_tr:
        raise ValueError(f"rt_batch {data_key} leading size must match idx")

    _, ct, cn_ratio = grid_seach_helper(rt_batch, 0, search_space, nsearch)
    n_ct, n_cn = int(nsearch[0]), int(nsearch[1])

    Csp_all = np.empty((n_tr, n_cn, n_ct), float)
    ct_best = np.empty(n_tr, float)
    cn_best = np.empty(n_tr, float)
    Csp_min = np.empty(n_tr, float)
    i_best = np.empty(n_tr, int)
    j_best = np.empty(n_tr, int)
    geom_idx = np.empty(n_tr, int)
    index_arr = np.arange(n_tr, dtype=int)
    n_top = min(100, n_cn * n_ct)
    sub100_all = np.empty((n_tr, n_top, 3), float)

    for k in range(n_tr):
        D = D_all[k]
        geom = int(idx_array[k])
        Csp = np.empty((n_cn, n_ct), float)

        for j, ct_j in enumerate(ct):
            for i, r in enumerate(cn_ratio):
                syn = RT_wave(
                    scattering_data, inc_wave, geom,
                    vp, vs, rho, r * ct_j, ct_j, wave=wave,
                )
                Csp[i, j] = C_spectrum_amplitude_norm(D, syn[model_key])

        ii0, jj0 = np.unravel_index(np.argmin(Csp), Csp.shape)
        Csp_all[k] = Csp
        ct_best[k] = float(ct[jj0])
        cn_best[k] = float(cn_ratio[ii0] * ct[jj0])
        Csp_min[k] = float(Csp[ii0, jj0])
        i_best[k] = int(ii0)
        j_best[k] = int(jj0)
        geom_idx[k] = geom

        flat = Csp.ravel()
        idx_flat = np.argpartition(flat, n_top - 1)[:n_top]
        idx_flat = idx_flat[np.argsort(flat[idx_flat])]
        ii, jj = np.unravel_index(idx_flat, Csp.shape)
        sub100_all[k] = np.column_stack([
            cn_ratio[ii] * ct[jj],
            ct[jj],
            flat[idx_flat],
        ])

    Csp_sum = np.sum(Csp_all, axis=0)
    Csp_mean = np.mean(Csp_all, axis=0)
    i_sum_best, j_sum_best = np.unravel_index(np.argmin(Csp_sum), Csp_sum.shape)
    ct_sum_best = float(ct[j_sum_best])
    cn_sum_best = float(cn_ratio[i_sum_best] * ct_sum_best)
    Csp_sum_min = float(Csp_sum[i_sum_best, j_sum_best])

    return {
        "mode": mode,
        "Csp": Csp_all,              # (n_tr, n_cn, n_ct)
        "Csp_sum": Csp_sum,
        "Csp_mean": Csp_mean,
        "ct": ct,
        "cn_ratio": cn_ratio,
        "ct_sum_best": ct_sum_best,
        "cn_sum_best": cn_sum_best,
        "Csp_sum_min": Csp_sum_min,
        "ct_best": ct_best,
        "cn_best": cn_best,
        "Csp_min": Csp_min,
        "i_best": i_best,
        "j_best": j_best,
        "sub100": sub100_all,        # (n_tr, n_top, 3) [cn, ct, C]
        "index": index_arr,
        "geom_idx": geom_idx,
        "idx": idx_array,
    }

def cnct_search_RTomegaV2(
    rt_batch,
    scattering_data,
    inc_wave,
    vp,
    vs,
    rho,
    mode="R",
    search_space=((1e-14, 1e-8), (1e-14, 1e-8)),
    nsearch=(30, 30),
    wave="SS",
):
    """
    Array (cn, ct) search on a 2D log-spaced grid (independent cn and ct).

    Like cnct_search_RTomega, but:
      search_space = [(cn_min, cn_max), (ct_min, ct_max)]
      nsearch      = (n_cn, n_ct)
      cn = logspace(cn_min → cn_max, n_cn)
      ct = logspace(ct_min → ct_max, n_ct)
      Csp[i, j] <-> cn[i], ct[j]

    mode : {"R", "T"}
      "R" — fit Er_omega
      "T" — fit Et_omega
    """
    mode = str(mode).upper()
    if mode not in ("R", "T"):
        raise ValueError('mode must be "R" or "T"')

    data_key = "Er_omega" if mode == "R" else "Et_omega"
    model_key = data_key

    if data_key not in rt_batch:
        raise ValueError(f"rt_batch missing '{data_key}' (required for mode={mode})")

    idx_array = np.asarray(rt_batch["idx"], int).ravel()
    n_tr = idx_array.size
    if n_tr == 0:
        raise ValueError("rt_batch has no traces")

    D_all = np.asarray(rt_batch[data_key])
    if D_all.shape[0] != n_tr:
        raise ValueError(f"rt_batch {data_key} leading size must match idx")

    try:
        n_cn, n_ct = int(nsearch[0]), int(nsearch[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("nsearch must be (n_cn, n_ct)") from exc
    if n_cn < 2 or n_ct < 2:
        raise ValueError("n_cn and n_ct must be >= 2")

    try:
        (cn_min, cn_max), (ct_min, ct_max) = search_space
        cn_min, cn_max = float(cn_min), float(cn_max)
        ct_min, ct_max = float(ct_min), float(ct_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "search_space must be [(cn_min, cn_max), (ct_min, ct_max)]"
        ) from exc
    if not (cn_min > 0 and cn_max > cn_min):
        raise ValueError("need 0 < cn_min < cn_max")
    if not (ct_min > 0 and ct_max > ct_min):
        raise ValueError("need 0 < ct_min < ct_max")

    cn = np.logspace(np.log10(cn_min), np.log10(cn_max), n_cn)
    ct = np.logspace(np.log10(ct_min), np.log10(ct_max), n_ct)

    Csp_all = np.empty((n_tr, n_cn, n_ct), float)
    ct_best = np.empty(n_tr, float)
    cn_best = np.empty(n_tr, float)
    Csp_min = np.empty(n_tr, float)
    i_best = np.empty(n_tr, int)
    j_best = np.empty(n_tr, int)
    geom_idx = np.empty(n_tr, int)
    index_arr = np.arange(n_tr, dtype=int)
    n_top = min(100, n_cn * n_ct)
    sub100_all = np.empty((n_tr, n_top, 3), float)

    for k in range(n_tr):
        D = D_all[k]
        geom = int(idx_array[k])
        Csp = np.empty((n_cn, n_ct), float)

        for j, ct_j in enumerate(ct):
            for i, cn_i in enumerate(cn):
                syn = RT_wave(
                    scattering_data, inc_wave, geom,
                    vp, vs, rho, cn_i, ct_j, wave=wave,
                )
                Csp[i, j] = C_spectrum_amplitude_norm(D, syn[model_key])

        ii0, jj0 = np.unravel_index(np.argmin(Csp), Csp.shape)
        Csp_all[k] = Csp
        cn_best[k] = float(cn[ii0])
        ct_best[k] = float(ct[jj0])
        Csp_min[k] = float(Csp[ii0, jj0])
        i_best[k] = int(ii0)
        j_best[k] = int(jj0)
        geom_idx[k] = geom

        flat = Csp.ravel()
        idx_flat = np.argpartition(flat, n_top - 1)[:n_top]
        idx_flat = idx_flat[np.argsort(flat[idx_flat])]
        ii, jj = np.unravel_index(idx_flat, Csp.shape)
        sub100_all[k] = np.column_stack([
            cn[ii],
            ct[jj],
            flat[idx_flat],
        ])

    Csp_sum = np.sum(Csp_all, axis=0)
    Csp_mean = np.mean(Csp_all, axis=0)
    i_sum_best, j_sum_best = np.unravel_index(np.argmin(Csp_sum), Csp_sum.shape)
    cn_sum_best = float(cn[i_sum_best])
    ct_sum_best = float(ct[j_sum_best])
    Csp_sum_min = float(Csp_sum[i_sum_best, j_sum_best])

    return {
        "mode": mode,
        "Csp": Csp_all,              # (n_tr, n_cn, n_ct)
        "Csp_sum": Csp_sum,          # (n_cn, n_ct)
        "Csp_mean": Csp_mean,
        "cn": cn,                    # log-spaced cn axis
        "ct": ct,                    # log-spaced ct axis
        "cn_sum_best": cn_sum_best,
        "ct_sum_best": ct_sum_best,
        "Csp_sum_min": Csp_sum_min,
        "cn_best": cn_best,
        "ct_best": ct_best,
        "Csp_min": Csp_min,
        "i_best": i_best,
        "j_best": j_best,
        "sub100": sub100_all,        # (n_tr, n_top, 3) [cn, ct, C]
        "index": index_arr,
        "geom_idx": geom_idx,
        "idx": idx_array,
    }

def RT_wave_angle(
    Gt,
    Gr,
    angle,
    inc_wave,
    vp,
    vs,
    rho,
    cn,
    ct,
    wave="SS",
):
    """
    Forward transmitted / reflected DAS spectra from incident wavelet S(ω).

    Same as RT_wave, but geometry is passed directly:
        Gt(ω), Gr(ω) : 1D geometric factors (same length as freqs)
        angle        : incidence angle in degrees (θ_i)

        ET(ω) = (ω² / vs) * Gt(ω) * S(ω) * T(ω)
        ER(ω) = (ω² / vs) * Gr(ω) * S(ω) * R(ω)

    freqs / S(ω) / time from inc_wave; omega = 2π * freqs.
    """
    req = ("S_omega", "time", "freqs")
    missing = [k for k in req if k not in inc_wave]
    if missing:
        raise ValueError(f"inc_wave missing keys: {missing}")

    S = np.asarray(inc_wave["S_omega"])
    freqs = np.asarray(inc_wave["freqs"], float)
    taxis = np.asarray(inc_wave["time"], float)
    omega = 2.0 * np.pi * freqs

    GT = np.asarray(Gt, float).ravel()
    GR = np.asarray(Gr, float).ravel()

    if S.shape != freqs.shape or GT.shape != freqs.shape or GR.shape != freqs.shape:
        raise ValueError(
            "length mismatch among S / Gt / Gr / freqs: "
            f"{S.shape}, {GT.shape}, {GR.shape}, {freqs.shape}"
        )

    theta_i = np.deg2rad(float(angle))
    ps = np.sin(theta_i) / float(vs)

    R, T = RPS_coef(vp, vs, ps, rho, omega, cn, ct, wave)

    scale = (omega ** 2) / float(vs)
    Et_omega = scale * GT * S * T
    Er_omega = scale * GR * S * R

    Et_time = real_wave(Et_omega)
    Er_time = real_wave(Er_omega)

    return {
        "Et_omega": Et_omega,
        "Et_time": Et_time,
        "Er_omega": Er_omega,
        "Er_time": Er_time,
        "time": taxis,
        "freqs": freqs,
        "angle": float(angle),
    }

def cnct_anglesearch_omega(
    rt_angle,
    Gt,
    Gr,
    inc_wave,
    vp,
    vs,
    rho,
    mode="R",
    search_space=((1e-14, 1e-8), (1e-14, 1e-8)),
    nsearch=(30, 30),
    wave="SS",
):
    """
    (cn, ct) log-space search for one angle (result of RT_wave_angle).

    Like cnct_search_RTomegaV2, but:
      - observed data from rt_angle (RT_wave_angle output)
      - forward model via RT_wave_angle(Gt, Gr, angle, ...)
      - no scattering_data / idx loop

    search_space = [(cn_min, cn_max), (ct_min, ct_max)]
    nsearch      = (n_cn, n_ct)
    Csp[i, j] <-> cn[i], ct[j]

    mode : {"R", "T"}
      "R" — fit Er_omega
      "T" — fit Et_omega
    """
    mode = str(mode).upper()
    if mode not in ("R", "T"):
        raise ValueError('mode must be "R" or "T"')

    data_key = "Er_omega" if mode == "R" else "Et_omega"
    model_key = data_key

    if data_key not in rt_angle:
        raise ValueError(f"rt_angle missing '{data_key}' (required for mode={mode})")
    if "angle" not in rt_angle:
        raise ValueError("rt_angle missing 'angle'")

    D = np.asarray(rt_angle[data_key])
    angle = float(rt_angle["angle"])

    try:
        n_cn, n_ct = int(nsearch[0]), int(nsearch[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("nsearch must be (n_cn, n_ct)") from exc
    if n_cn < 2 or n_ct < 2:
        raise ValueError("n_cn and n_ct must be >= 2")

    try:
        (cn_min, cn_max), (ct_min, ct_max) = search_space
        cn_min, cn_max = float(cn_min), float(cn_max)
        ct_min, ct_max = float(ct_min), float(ct_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "search_space must be [(cn_min, cn_max), (ct_min, ct_max)]"
        ) from exc
    if not (cn_min > 0 and cn_max > cn_min):
        raise ValueError("need 0 < cn_min < cn_max")
    if not (ct_min > 0 and ct_max > ct_min):
        raise ValueError("need 0 < ct_min < ct_max")

    cn = np.logspace(np.log10(cn_min), np.log10(cn_max), n_cn)
    ct = np.logspace(np.log10(ct_min), np.log10(ct_max), n_ct)

    Csp = np.empty((n_cn, n_ct), float)
    for j, ct_j in enumerate(ct):
        for i, cn_i in enumerate(cn):
            syn = RT_wave_angle(
                Gt, Gr, angle, inc_wave,
                vp, vs, rho, cn_i, ct_j, wave=wave,
            )
            Csp[i, j] = C_spectrum_amplitude_norm(D, syn[model_key])

    ii0, jj0 = np.unravel_index(np.argmin(Csp), Csp.shape)
    cn_best = float(cn[ii0])
    ct_best = float(ct[jj0])
    Csp_min = float(Csp[ii0, jj0])

    n_top = min(100, n_cn * n_ct)
    flat = Csp.ravel()
    idx_flat = np.argpartition(flat, n_top - 1)[:n_top]
    idx_flat = idx_flat[np.argsort(flat[idx_flat])]
    ii, jj = np.unravel_index(idx_flat, Csp.shape)
    sub100 = np.column_stack([cn[ii], ct[jj], flat[idx_flat]])

    return {
        "mode": mode,
        "angle": angle,
        "Csp": Csp,              # (n_cn, n_ct)
        "cn": cn,
        "ct": ct,
        "cn_best": cn_best,
        "ct_best": ct_best,
        "Csp_min": Csp_min,
        "i_best": int(ii0),
        "j_best": int(jj0),
        "sub100": sub100,        # (n_top, 3) [cn, ct, C]
    }


# ---------------------------------------------------------------------------
# from DASref_modeling.py
# ---------------------------------------------------------------------------
def windowed_sinc(freqs, arg, alpha=0.3):
    """
    Windowed sinc on a Fourier frequency axis.

    sinc(f) = sin(2*pi*f*arg) / (2*pi*f*arg)
    Limit at f=0 is 1.

    First zeros at |f| = 1/(2*|arg|).
    Apply a Tukey taper on the main lobe and set everything
    outside to zero (no side lobes).

    Parameters
    ----------
    freqs : 1D array
        Fourier frequencies [Hz], may include negatives and DC.
    arg : float
        Scale in the sinc argument (seconds if freqs in Hz).
    alpha : float
        Tukey taper fraction (0 = boxcar, 1 = Hann).
        Smaller alpha preserves the sinc shape near f=0 more.

    Returns
    -------
    sinc_t : 1D array, same shape as freqs
    """
    freqs = np.asarray(freqs, dtype=float)
    arg = float(arg)
    alpha = float(alpha)

    if arg == 0.0:
        raise ValueError("arg must be nonzero")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    omega = 2.0 * np.pi * freqs * arg
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sin(omega) / omega
    s = np.where(np.abs(omega) < 1e-15, 1.0, s)

    f0 = 0.5 / abs(arg)
    mask = np.abs(freqs) < f0

    w = np.zeros_like(freqs)
    n = int(np.count_nonzero(mask))
    if n == 0:
        return w
    if n == 1:
        w[mask] = 1.0
    else:
        # tukey expects samples ordered along the lobe;
        # freqs may be fft-shifted or unordered — build on sorted lobe
        idx = np.where(mask)[0]
        order = np.argsort(freqs[idx])
        idx_sorted = idx[order]

        w_lobe = tukey(n, alpha=alpha)
        w[idx_sorted] = w_lobe

    return s * w
    
def geometric_factor_dir(dict_dir, gauge_length, velocity, alpha=0.3):
    """
    Frequency-dependent direct-ray geometric factor G_d(ω).

        G_d(ω) = (f·p)(f·û) * sinc[(ω g / 2c)(f·û)] / d

    with
        (f·p)  = dict_dir['n_p_dir_dot']
        (f·û)  = dict_dir['n_u_dir_dot']
        d      = dict_dir['mean_dist_source']
        freqs  = dict_dir['stf_trace']['freqs']
        omega  = 2π freqs
        arg    = g (f·û) / (2c)
        sinc   = windowed_sinc(freqs, arg, alpha=alpha)

    Parameters
    ----------
    dict_dir : dict
        Needs keys:
          'stf_trace' (with 'freqs'),
          'mean_dist_source',
          'n_p_dir_dot',
          'n_u_dir_dot'
        e.g. output of stack_source_distance() merged into stf_data.
    gauge_length : float
        DAS gauge length g [m].
    velocity : float
        Wave speed c [m/s] (e.g. vs for SS).
    alpha : float
        Tukey taper for windowed_sinc. Default 0.3.

    Returns
    -------
    dict with:
        Gd       : (n_freq,) direct geometric factor
        freqs    : (n_freq,)
        omega    : (n_freq,)
        d        : float, mean_dist_source
        arg_d    : float, g * n_u_dir_dot / (2c)
        d_factor : float, 1/d
    """
    g = float(gauge_length)
    c = float(velocity)
    if g <= 0 or c <= 0:
        raise ValueError("gauge_length and velocity must be > 0")

    req = ("stf_trace", "mean_dist_source", "n_p_dir_dot", "n_u_dir_dot")
    missing = [k for k in req if k not in dict_dir]
    if missing:
        raise ValueError(f"dict_dir missing keys: {missing}")

    stf = dict_dir["stf_trace"]
    if "freqs" not in stf:
        raise KeyError("dict_dir['stf_trace'] needs 'freqs'")

    freqs = np.asarray(stf["freqs"], float)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("stf_trace['freqs'] must be a 1-D non-empty array [Hz]")

    d = float(dict_dir["mean_dist_source"])
    if d <= 0.0:
        raise ValueError("mean_dist_source must be > 0")

    n_p = float(dict_dir["n_p_dir_dot"])
    n_u = float(dict_dir["n_u_dir_dot"])
    if n_u == 0.0:
        raise ValueError("n_u_dir_dot is zero (sinc arg undefined)")

    arg_d = g * n_u / (2.0 * c)
    sinc_d = windowed_sinc(freqs, arg_d, alpha=alpha)
    Gd = (n_p * n_u * sinc_d) / d

    return {
        "Gd": Gd,
        "freqs": freqs,
        "omega": 2.0 * np.pi * freqs,
        "d": d,
        "arg_d": arg_d,
        "d_factor": 1.0 / d,
        "n_p_dir_dot": n_p,
        "n_u_dir_dot": n_u,
    }

def geometric_factor_ref(ref_data, freqs, gauge_length, velocity, alpha=0.5):
    """
    Frequency-dependent reflected geometric factor G_R(ω) only.

        G_R(ω) = (n·p)(n·û) * sinc[(ω g / 2c) (n·û)] / d

    Parameters
    ----------
    ref_data : dict
        Either ``ref_stf_data["ref_data"]`` or the full ``ref_stf_data``
        pickle (must contain key ``ref_data``). Needs:
          reflection_traces, n_p_ref_dot, n_u_ref_dot,
          dist_source_pinc, dist_pinc_rr
        d = dist_source_pinc + dist_pinc_rr (source -> fracture -> receiver).
    freqs : array-like
        Frequency axis [Hz], same order as the STF FFT (np.fft.fftfreq).
        Used for Gr and omega = 2π freqs.
    gauge_length : float
        DAS gauge length g [m].
    velocity : float
        Wave speed c [m/s] (e.g. vs for SS).
    alpha : float
        Tukey taper for windowed_sinc.

    Returns
    -------
    dict with Gr (n_traces, n_freqs), d_factor (1/d per trace), freqs,
    omega, arg_r.
    """
    g = float(gauge_length)
    c = float(velocity)
    if g <= 0 or c <= 0:
        raise ValueError("gauge_length and velocity must be > 0")

    freqs = np.asarray(freqs, float)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("freqs must be a 1-D non-empty array [Hz]")

    if "ref_data" in ref_data and "n_p_ref_dot" not in ref_data:
        ref_data = ref_data["ref_data"]

    DAS_rr = ref_data["reflection_traces"]

    n_p_r = np.asarray(ref_data["n_p_ref_dot"], float).ravel()
    n_u_r = np.asarray(ref_data["n_u_ref_dot"], float).ravel()

    d_r1 = np.asarray(ref_data["dist_source_pinc"], float).ravel()
    d_r2 = np.asarray(ref_data["dist_pinc_rr"], float).ravel()
    d_r = d_r1 + d_r2

    if not (n_p_r.size == n_u_r.size == d_r.size):
        raise ValueError(
            "reflection geometry sizes differ: "
            f"n_p_ref_dot={n_p_r.size}, n_u_ref_dot={n_u_r.size}, "
            f"dist_source_pinc+dist_pinc_rr={d_r.size}"
        )

    n_geom = n_p_r.size
    n_rr = int(np.asarray(DAS_rr.data).shape[0])
    if n_rr > n_geom:
        raise ValueError(f"DAS traces ({n_rr}) exceed geometry length ({n_geom})")

    Gr = np.zeros((n_geom, freqs.size), dtype=float)
    arg_r = np.zeros(n_geom, dtype=float)

    for i in range(n_geom):
        if d_r[i] == 0.0:
            raise ValueError(f"dist_source_pinc+dist_pinc_rr [{i}] is zero")
        if n_u_r[i] == 0.0:
            raise ValueError(f"n_u_ref_dot[{i}] is zero (sinc arg undefined)")
        arg_r[i] = g * n_u_r[i] / (2.0 * c)
        sinc_r = windowed_sinc(freqs, arg_r[i], alpha=alpha)
        Gr[i, :] = (n_p_r[i] * n_u_r[i] * sinc_r) /d_r[i]

    return {
        "Gr": Gr,
        "d_factor": 1.0 / d_r,
        "freqs": freqs,
        "omega": 2.0 * np.pi * freqs,
        "arg_r": arg_r,
    }

def _as_inc_wave(inc_wave):
    """
    Normalize incident-wave dict to S_omega / freqs / time.

    Expected:
      inc_wave keys: trace, time, fft, freqs
        (e.g. ref_stf_data["stf_data"]["stf_trace"])

    Also unwraps nested stf_data / stf_trace if passed by accident,
    and accepts already-normalized {"S_omega", "freqs", "time"}.
    """
    if "stf_data" in inc_wave and "fft" not in inc_wave:
        inc_wave = inc_wave["stf_data"]
    if "stf_trace" in inc_wave and "fft" not in inc_wave:
        inc_wave = inc_wave["stf_trace"]

    if all(k in inc_wave for k in ("S_omega", "freqs", "time")):
        return {
            "S_omega": np.asarray(inc_wave["S_omega"]),
            "freqs": np.asarray(inc_wave["freqs"], float),
            "time": np.asarray(inc_wave["time"], float),
        }

    req = ("trace", "time", "fft", "freqs")
    missing = [k for k in req if k not in inc_wave]
    if missing:
        raise ValueError(
            "inc_wave must have keys ['trace', 'time', 'fft', 'freqs']; "
            f"missing: {missing}"
        )

    return {
        "S_omega": np.asarray(inc_wave["fft"]),
        "freqs": np.asarray(inc_wave["freqs"], float),
        "time": np.asarray(inc_wave["time"], float),
        "trace": np.asarray(inc_wave["trace"], float),
    }   
 
def R_wave(
    ref_data,
    inc_wave,
    idx,
    a,
    vp,
    vs,
    rho,
    cn,
    ct,
    wave="SS",
):
    """
    Forward reflected DAS spectrum.

        S_a(ω) = a * S(ω)
        ER(ω)  = (ω² / vs) * Gr(ω) * S_a(ω) * R(ω)

    Parameters
    ----------
    ref_data : dict
        ref_stf_data["ref_data"], after geometric_factor_ref update.
        Required keys: Gr, omega, theta_i_deg
    inc_wave : dict
        ref_stf_data["stf_data"]  (uses stf_trace['fft','freqs','time'])
    idx : int
        Row into Gr / theta_i_deg
    a : float
        Real scalar; st_scale = a * st
    """
    # allow accidental full pickle
    if "ref_data" in ref_data and "Gr" not in ref_data:
        ref_data = ref_data["ref_data"]

    req = ("Gr", "omega", "theta_i_deg")
    missing = [k for k in req if k not in ref_data]
    if missing:
        raise ValueError(f"ref_data missing keys: {missing}")

    iw = _as_inc_wave(inc_wave)
    a = float(a)
    idx = int(idx)

    Gr = np.asarray(ref_data["Gr"])
    omega = np.asarray(ref_data["omega"], float)
    theta_i_deg = np.asarray(ref_data["theta_i_deg"], float)

    if not (0 <= idx < Gr.shape[0]):
        raise ValueError(f"idx={idx} out of range for Gr (rows={Gr.shape[0]})")
    if idx >= theta_i_deg.size:
        raise ValueError(
            f"idx={idx} out of range for theta_i_deg (n={theta_i_deg.size})"
        )

    S = a * np.asarray(iw["S_omega"])   # a * stf_trace['fft']
    freqs = np.asarray(iw["freqs"], float)
    taxis = np.asarray(iw["time"], float)
    GR = np.asarray(Gr[idx], float)

    if S.shape != freqs.shape or GR.shape != freqs.shape:
        raise ValueError(
            "length mismatch among S / Gr / freqs: "
            f"{S.shape}, {GR.shape}, {freqs.shape}"
        )
    if omega.shape != freqs.shape:
        raise ValueError("ref_data['omega'] length must match stf_trace['freqs']")

    theta_i = np.deg2rad(theta_i_deg[idx])
    ps = np.sin(theta_i) / float(vs)

    R, _ = RPS_coef(vp, vs, ps, rho, omega, cn, ct, wave)

    scale = (omega**2) / float(vs)
    Er_omega = scale * GR * S * R
    Er_time = real_wave(Er_omega)

    return {
        "Er_omega": Er_omega,
        "Er_time": Er_time,
        "time": taxis,
        "freqs": freqs,
        "idx": idx,
        "angle": float(theta_i_deg[idx]),
        "a": a,
    }

def R_wave_v2(
    ref_data,
    inc_wave,
    idx,
    a,
    vp,
    vs,
    rho,
    cn,
    ct,
    Gd,
    eps,
    wave="SS",
):
    """
    Forward reflected DAS spectrum (stabilized by G_d).

        S_a(ω) = a * S(ω)
        ε_val  = max(|G_d(ω)|) * eps
        ER(ω)  = Gr(ω) * S_a(ω) * R(ω) * conj(G_d)
                 / (|G_d(ω)|² + ε_val²)

    Parameters
    ----------
    ref_data : dict
        Needs Gr, omega, theta_i_deg.
    inc_wave : dict
        STF dict (fft/freqs/time or S_omega/freqs/time).
    idx : int
        Row into Gr / theta_i_deg.
    a : float
        Scalar on S.
    Gd : array-like
        Direct geometric factor G_d(ω), same length as freqs.
        (e.g. geometric_factor_dir(...)['Gd'])
    eps : float
        Stabilization fraction; ε_val = max(|Gd|) * eps.
    """
    if "ref_data" in ref_data and "Gr" not in ref_data:
        ref_data = ref_data["ref_data"]

    req = ("Gr", "omega", "theta_i_deg")
    missing = [k for k in req if k not in ref_data]
    if missing:
        raise ValueError(f"ref_data missing keys: {missing}")

    iw = _as_inc_wave(inc_wave)
    a = float(a)
    idx = int(idx)
    eps = float(eps)
    if eps < 0:
        raise ValueError("eps must be >= 0")

    Gr = np.asarray(ref_data["Gr"])
    omega = np.asarray(ref_data["omega"], float)
    theta_i_deg = np.asarray(ref_data["theta_i_deg"], float)

    if not (0 <= idx < Gr.shape[0]):
        raise ValueError(f"idx={idx} out of range for Gr (rows={Gr.shape[0]})")
    if idx >= theta_i_deg.size:
        raise ValueError(
            f"idx={idx} out of range for theta_i_deg (n={theta_i_deg.size})"
        )

    S = a * np.asarray(iw["S_omega"])
    freqs = np.asarray(iw["freqs"], float)
    taxis = np.asarray(iw["time"], float)
    GR = np.asarray(Gr[idx], float)
    Gd = np.asarray(Gd, complex).ravel()

    if S.shape != freqs.shape or GR.shape != freqs.shape:
        raise ValueError(
            "length mismatch among S / Gr / freqs: "
            f"{S.shape}, {GR.shape}, {freqs.shape}"
        )
    if omega.shape != freqs.shape:
        raise ValueError("ref_data['omega'] length must match stf freqs")
    if Gd.shape != freqs.shape:
        raise ValueError(
            f"Gd length {Gd.shape} != freqs length {freqs.shape}"
        )

    theta_i = np.deg2rad(theta_i_deg[idx])
    ps = np.sin(theta_i) / float(vs)

    R, _ = RPS_coef(vp, vs, ps, rho, omega, cn, ct, wave)

    Gd_abs = np.abs(Gd)
    eps_val = float(np.max(Gd_abs)) * eps
    denom = Gd_abs**2 + eps_val**2

    Er_omega = GR * S * R * np.conj(Gd) / denom
    Er_time = real_wave(Er_omega)

    return {
        "Er_omega": Er_omega,
        "Er_time": Er_time,
        "time": taxis,
        "freqs": freqs,
        "idx": idx,
        "angle": float(theta_i_deg[idx]),
        "a": a,
        "Gd": Gd,
        "eps": eps,
        "eps_val": eps_val,
    }

def R_wave_batch(
    ref_data,
    inc_wave,
    idx_array,
    a,
    vp,
    vs,
    rho,
    cn,
    ct,
    wave="SS",
):
    """Reflection-only batch; same API idea as RT_wave_batch + scalar a."""
    idx_array = np.asarray(idx_array, int).ravel()
    n = idx_array.size
    if n == 0:
        raise ValueError("idx_array is empty")

    Er_omega = Er_time = None
    angles = np.empty(n, float)

    for k, idx in enumerate(idx_array):
        out = R_wave(
            ref_data, inc_wave, idx, a, vp, vs, rho, cn, ct, wave=wave,
        )
        if k == 0:
            nf = out["Er_omega"].size
            nt = out["Er_time"].size
            Er_omega = np.empty((n, nf), dtype=complex)
            Er_time = np.empty((n, nt), dtype=float)
            freqs = np.asarray(out["freqs"])
            taxis = np.asarray(out["time"])

        Er_omega[k] = out["Er_omega"]
        Er_time[k] = out["Er_time"]
        angles[k] = out["angle"]

    return {
        "Er_omega": Er_omega,
        "Er_time": Er_time,
        "freqs": freqs,
        "time": taxis,
        "idx": idx_array,
        "angle": angles,
        "a": float(a),
    }


def R_wave_batch_v2(
    ref_data,
    inc_wave,
    idx_array,
    a,
    vp,
    vs,
    rho,
    cn,
    ct,
    Gd,
    eps,
    wave="SS",
):
    """
    Batch version of R_wave_v2 for many geometry indices.

        S_a(ω) = a * S(ω)
        ε_val  = max(|G_d|) * eps
        ER(ω)  = Gr * S_a * R * conj(G_d) / (|G_d|² + ε_val²)

    Parameters
    ----------
    idx_array : array-like
        Trace / geometry indices into Gr, theta_i_deg.
    Gd : array-like
        Direct geometric factor G_d(ω), shared for all traces.
    eps : float
        Stabilization fraction.

    Returns
    -------
    dict with Er_omega (n, nf), Er_time (n, nt), freqs, time, idx, angle, a, Gd, eps, eps_val
    """
    idx_array = np.asarray(idx_array, int).ravel()
    n = idx_array.size
    if n == 0:
        raise ValueError("idx_array is empty")

    Er_omega = Er_time = None
    angles = np.empty(n, float)
    eps_val = None

    for k, idx in enumerate(idx_array):
        out = R_wave_v2(
            ref_data, inc_wave, idx, a, vp, vs, rho, cn, ct,
            Gd=Gd, eps=eps, wave=wave,
        )
        if k == 0:
            nf = out["Er_omega"].size
            nt = out["Er_time"].size
            Er_omega = np.empty((n, nf), dtype=complex)
            Er_time = np.empty((n, nt), dtype=float)
            freqs = np.asarray(out["freqs"])
            taxis = np.asarray(out["time"])
            Gd_out = np.asarray(out["Gd"])
            eps_val = float(out["eps_val"])

        Er_omega[k] = out["Er_omega"]
        Er_time[k] = out["Er_time"]
        angles[k] = out["angle"]

    return {
        "Er_omega": Er_omega,
        "Er_time": Er_time,
        "freqs": freqs,
        "time": taxis,
        "idx": idx_array,
        "angle": angles,
        "a": float(a),
        "Gd": Gd_out,
        "eps": float(eps),
        "eps_val": eps_val,
    }

def scale_hat_time(D, H, w=None):
    """
    Optimal scalar, optionally trace-weighted:

        s_hat(m) = sum_i w_i sum_k d_i(t_k) h_i(t_k;m)
                   / sum_i w_i sum_k [h_i(t_k;m)]^2

    D, H : (N, nt)
    w    : (N,) per-trace weights, or None (all 1 -> the unweighted form)
    """
    D = np.asarray(D, float)
    H = np.asarray(H, float)
    if D.shape != H.shape:
        raise ValueError(f"D and H shape mismatch: {D.shape} vs {H.shape}")
    if D.ndim != 2:
        raise ValueError("D and H must be 2D (n_tr, n_t)")

    if w is None:
        num = float(np.sum(D * H))
        den = float(np.sum(H * H))
    else:
        w = np.asarray(w, float).ravel()
        if w.size != D.shape[0]:
            raise ValueError(f"w length {w.size} != n_tr {D.shape[0]}")
        num = float(w @ np.sum(D * H, axis=1))   # per-trace dot products, then weighted sum
        den = float(w @ np.sum(H * H, axis=1))
    if den == 0.0:
        raise ValueError("sum h^2 is zero; cannot compute s_hat")
    return num / den

def C_spectrum_amplitude(D, E):
    """
    Normalized spectral L2 misfit.

        C = Σ_f |D(f) - E(f)|² / Σ_f |D(f)|²
    """
    D = np.asarray(D)
    E = np.asarray(E)
    if D.shape != E.shape:
        raise ValueError(f"D and E shape mismatch: {D.shape} vs {E.shape}")

    denom = float(np.sum(np.abs(D) ** 2))
    if denom == 0.0:
        raise ValueError("Σ |D(f)|² is zero; cannot normalize")

    return float(np.sum(np.abs(D - E) ** 2) )

def C_time_unnorm(d, e):
    """
    Time-domain misfit like C_wavelet but NOT normalized (and no 1/2):

        C = sum_k [d(t_k) - e(t_k)]^2
    """
    d = np.asarray(d, float).ravel()
    e = np.asarray(e, float).ravel()
    if d.shape != e.shape:
        raise ValueError(f"d and e shape mismatch: {d.shape} vs {e.shape}")
    return float(np.sum((d - e) ** 2) )

def cnct_search_Rtime_ratio(
    r_batch,
    ref_data,
    inc_wave,
    a,
    vp,
    vs,
    rho,
    search_space=((0.01, 3.0), (1e-14, 1e-8)),
    nsearch=(30, 30),
    true_cnct=None,
    wave="SS",
):
    """
    Time-domain search on (cn/ct, ct) with cn = ratio * ct.

    search_space = [(ratio_min, ratio_max), (ct_min, ct_max)]
    nsearch      = (n_ratio, n_ct)
    ratio, ct    = logspace(...)

    Same cost / s_hat logic as cnct_search_Rtime.
    Csp[i, j] <-> ratio[i], ct[j],  cn = ratio[i] * ct[j]

    true_cnct : (cn, ct) or None
        If given, inserts ratio_true = cn/ct into the ratio axis and
        ct into the ct axis so the true model is an exact grid node.
    """
    if "Er_time" not in r_batch:
        raise ValueError("r_batch missing 'Er_time'")

    idx_array = np.asarray(r_batch["idx"], int).ravel()
    n_tr = idx_array.size
    if n_tr == 0:
        raise ValueError("r_batch has no traces")

    D_all = np.asarray(r_batch["Er_time"], float)
    if D_all.ndim != 2 or D_all.shape[0] != n_tr:
        raise ValueError("r_batch['Er_time'] must be (n_tr, n_t)")

    try:
        n_ratio, n_ct = int(nsearch[0]), int(nsearch[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("nsearch must be (n_ratio, n_ct)") from exc
    if n_ratio < 2 or n_ct < 2:
        raise ValueError("n_ratio and n_ct must be >= 2")

    try:
        (r_min, r_max), (ct_min, ct_max) = search_space
        r_min, r_max = float(r_min), float(r_max)
        ct_min, ct_max = float(ct_min), float(ct_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "search_space must be [(ratio_min, ratio_max), (ct_min, ct_max)]"
        ) from exc
    if not (r_min > 0 and r_max > r_min):
        raise ValueError("need 0 < ratio_min < ratio_max")
    if not (ct_min > 0 and ct_max > ct_min):
        raise ValueError("need 0 < ct_min < ct_max")

    cn_ratio = np.logspace(np.log10(r_min), np.log10(r_max), n_ratio)
    ct = np.logspace(np.log10(ct_min), np.log10(ct_max), n_ct)

    i_true = j_true = None
    true_pair = None
    ratio_true = None
    if true_cnct is not None:
        cn_t, ct_t = map(float, true_cnct)
        if cn_t <= 0 or ct_t <= 0:
            raise ValueError("true_cnct values must be > 0")
        ratio_true = cn_t / ct_t
        cn_ratio = np.unique(np.sort(np.append(cn_ratio, ratio_true)))
        ct = np.unique(np.sort(np.append(ct, ct_t)))
        true_pair = (cn_t, ct_t)
        i_true = int(np.argmin(np.abs(np.log10(cn_ratio) - np.log10(ratio_true))))
        j_true = int(np.argmin(np.abs(np.log10(ct) - np.log10(ct_t))))

    n_ratio, n_ct = int(cn_ratio.size), int(ct.size)
    a = float(a)

    Csp_all = np.empty((n_tr, n_ratio, n_ct), float)
    s_hat = np.empty((n_ratio, n_ct), float)
    cn_grid = np.empty((n_ratio, n_ct), float)  # cn[i,j] = ratio[i]*ct[j]

    nt = D_all.shape[1]
    H_all = np.empty((n_tr, nt), float)

    for j, ct_j in enumerate(ct):
        for i, r in enumerate(cn_ratio):
            cn_i = r * ct_j
            cn_grid[i, j] = cn_i
            for k, geom in enumerate(idx_array):
                syn = R_wave(
                    ref_data, inc_wave, int(geom), a,
                    vp, vs, rho, cn_i, ct_j, wave=wave,
                )
                h = np.asarray(syn["Er_time"], float).ravel()
                if h.size != nt:
                    raise ValueError(
                        f"Er_time length {h.size} != observed nt={nt}"
                    )
                H_all[k] = h

            s = scale_hat_time(D_all, H_all)
            s_hat[i, j] = s
            for k in range(n_tr):
                Csp_all[k, i, j] = C_time_unnorm(D_all[k], s * H_all[k])

    ct_best = np.empty(n_tr, float)
    cn_best = np.empty(n_tr, float)
    ratio_best = np.empty(n_tr, float)
    Csp_min = np.empty(n_tr, float)
    i_best = np.empty(n_tr, int)
    j_best = np.empty(n_tr, int)
    geom_idx = np.asarray(idx_array, int).copy()
    index_arr = np.arange(n_tr, dtype=int)
    n_top = min(100, n_ratio * n_ct)
    sub100_all = np.empty((n_tr, n_top, 3), float)

    for k in range(n_tr):
        Csp = Csp_all[k]
        ii0, jj0 = np.unravel_index(np.argmin(Csp), Csp.shape)
        ratio_best[k] = float(cn_ratio[ii0])
        ct_best[k] = float(ct[jj0])
        cn_best[k] = float(cn_ratio[ii0] * ct[jj0])
        Csp_min[k] = float(Csp[ii0, jj0])
        i_best[k] = int(ii0)
        j_best[k] = int(jj0)

        flat = Csp.ravel()
        idx_flat = np.argpartition(flat, n_top - 1)[:n_top]
        idx_flat = idx_flat[np.argsort(flat[idx_flat])]
        ii, jj = np.unravel_index(idx_flat, Csp.shape)
        sub100_all[k] = np.column_stack([
            cn_ratio[ii] * ct[jj],  # cn
            ct[jj],
            flat[idx_flat],
        ])

    Csp_sum = np.sum(Csp_all, axis=0)
    Csp_mean = np.mean(Csp_all, axis=0)
    i_sum_best, j_sum_best = np.unravel_index(np.argmin(Csp_sum), Csp_sum.shape)
    ratio_sum_best = float(cn_ratio[i_sum_best])
    ct_sum_best = float(ct[j_sum_best])
    cn_sum_best = float(ratio_sum_best * ct_sum_best)
    Csp_sum_min = float(Csp_sum[i_sum_best, j_sum_best])
    s_hat_best = float(s_hat[i_sum_best, j_sum_best])

    out = {
        "mode": "R",
        "domain": "time",
        "grid": "ratio_ct",
        "a": a,
        "Csp": Csp_all,                 # (n_tr, n_ratio, n_ct)
        "Csp_sum": Csp_sum,
        "Csp_mean": Csp_mean,
        "s_hat": s_hat,
        "s_hat_best": s_hat_best,
        "cn_ratio": cn_ratio,           # 1D ratio axis
        "ct": ct,
        "cn_grid": cn_grid,             # (n_ratio, n_ct) = ratio*ct
        "cn_sum_best": cn_sum_best,
        "ct_sum_best": ct_sum_best,
        "ratio_sum_best": ratio_sum_best,
        "Csp_sum_min": Csp_sum_min,
        "cn_best": cn_best,
        "ct_best": ct_best,
        "ratio_best": ratio_best,
        "Csp_min": Csp_min,
        "i_best": i_best,
        "j_best": j_best,
        "i_sum_best": int(i_sum_best),
        "j_sum_best": int(j_sum_best),
        "sub100": sub100_all,
        "index": index_arr,
        "geom_idx": geom_idx,
        "idx": idx_array,
        "true_cnct": true_pair,
        "ratio_true": ratio_true,
        "i_true": i_true,
        "j_true": j_true,
        "s_hat_true": None,
        "Csp_sum_true": None,
    }
    if true_pair is not None:
        out["s_hat_true"] = float(s_hat[i_true, j_true])
        out["Csp_sum_true"] = float(Csp_sum[i_true, j_true])
    return out

def cnct_search_axes(search_space, nsearch, true_cnct=None):
    """
    Build log-spaced (cn, ct) axes for a compliance search.

    Parameters
    ----------
    search_space : [(cn_min, cn_max), (ct_min, ct_max)]
    nsearch : (n_cn, n_ct)
        Number of logspace samples before inserting true_cnct.
    true_cnct : (cn, ct) or None
        True compliance [m/Pa]. If given, both values are inserted
        into the axes (np.unique keeps order).

    Returns
    -------
    dict
        cn, ct (1-D float), n_cn, n_ct,
        true_cnct, i_true, j_true  (None if true_cnct is None)
    """
    try:
        n_cn, n_ct = int(nsearch[0]), int(nsearch[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("nsearch must be (n_cn, n_ct)") from exc
    if n_cn < 2 or n_ct < 2:
        raise ValueError("n_cn and n_ct must be >= 2")

    try:
        (cn_min, cn_max), (ct_min, ct_max) = search_space
        cn_min, cn_max = float(cn_min), float(cn_max)
        ct_min, ct_max = float(ct_min), float(ct_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "search_space must be [(cn_min, cn_max), (ct_min, ct_max)]"
        ) from exc
    if not (cn_min > 0 and cn_max > cn_min):
        raise ValueError("need 0 < cn_min < cn_max")
    if not (ct_min > 0 and ct_max > ct_min):
        raise ValueError("need 0 < ct_min < ct_max")

    cn = np.logspace(np.log10(cn_min), np.log10(cn_max), n_cn)
    ct = np.logspace(np.log10(ct_min), np.log10(ct_max), n_ct)

    i_true = j_true = None
    true_pair = None
    if true_cnct is not None:
        cn_t, ct_t = map(float, true_cnct)
        if cn_t <= 0 or ct_t <= 0:
            raise ValueError("true_cnct values must be > 0")
        cn = np.unique(np.sort(np.append(cn, cn_t)))
        ct = np.unique(np.sort(np.append(ct, ct_t)))
        true_pair = (cn_t, ct_t)
        i_true = int(np.argmin(np.abs(np.log10(cn) - np.log10(cn_t))))
        j_true = int(np.argmin(np.abs(np.log10(ct) - np.log10(ct_t))))

    return {
        "cn": cn,
        "ct": ct,
        "n_cn": int(cn.size),
        "n_ct": int(ct.size),
        "true_cnct": true_pair,
        "i_true": i_true,
        "j_true": j_true,
    }

def cnct_search_pack(
    Csp_all,
    s_hat,
    cn,
    ct,
    idx_array,
    a,
    axes=None,
    mode="R",
    domain="time",
    **extra,
):
    """
    Post-process a (cn, ct) cost cube and build the search result dict.

    Parameters
    ----------
    Csp_all : (n_tr, n_cn, n_ct)
    s_hat : (n_cn, n_ct)
    cn, ct : 1-D axes
    idx_array : geometry indices used in the search
    a : float
    axes : dict from cnct_search_axes (optional)
        If given, copies true_cnct / i_true / j_true and adds
        s_hat_true, Csp_sum_true at that node.
    mode, domain : str
    **extra : merged into the returned dict (e.g. s_hat_kind=...)

    Returns
    -------
    dict
        Same keys as cnct_search_Rtime / RtimeV2, plus true-node
        fields when axes has true_cnct.
    """
    Csp_all = np.asarray(Csp_all, float)
    s_hat = np.asarray(s_hat, float)
    cn = np.asarray(cn, float)
    ct = np.asarray(ct, float)
    idx_array = np.asarray(idx_array, int).ravel()

    n_tr = Csp_all.shape[0]
    n_cn, n_ct = cn.size, ct.size
    if Csp_all.shape != (n_tr, n_cn, n_ct):
        raise ValueError(
            f"Csp_all shape {Csp_all.shape} != ({n_tr}, {n_cn}, {n_ct})"
        )
    if s_hat.shape != (n_cn, n_ct):
        raise ValueError(f"s_hat shape {s_hat.shape} != ({n_cn}, {n_ct})")

    cn_best = np.empty(n_tr, float)
    ct_best = np.empty(n_tr, float)
    Csp_min = np.empty(n_tr, float)
    i_best = np.empty(n_tr, int)
    j_best = np.empty(n_tr, int)
    geom_idx = idx_array.copy()
    index_arr = np.arange(n_tr, dtype=int)
    n_top = min(100, n_cn * n_ct)
    sub100_all = np.empty((n_tr, n_top, 3), float)

    for k in range(n_tr):
        Csp = Csp_all[k]
        ii0, jj0 = np.unravel_index(np.argmin(Csp), Csp.shape)
        cn_best[k] = float(cn[ii0])
        ct_best[k] = float(ct[jj0])
        Csp_min[k] = float(Csp[ii0, jj0])
        i_best[k] = int(ii0)
        j_best[k] = int(jj0)

        flat = Csp.ravel()
        idx_flat = np.argpartition(flat, n_top - 1)[:n_top]
        idx_flat = idx_flat[np.argsort(flat[idx_flat])]
        ii, jj = np.unravel_index(idx_flat, Csp.shape)
        sub100_all[k] = np.column_stack([cn[ii], ct[jj], flat[idx_flat]])

    Csp_sum = np.sum(Csp_all, axis=0)
    Csp_mean = np.mean(Csp_all, axis=0)
    i_sum_best, j_sum_best = np.unravel_index(np.argmin(Csp_sum), Csp_sum.shape)
    cn_sum_best = float(cn[i_sum_best])
    ct_sum_best = float(ct[j_sum_best])
    Csp_sum_min = float(Csp_sum[i_sum_best, j_sum_best])
    s_hat_best = float(s_hat[i_sum_best, j_sum_best])

    out = {
        "mode": mode,
        "domain": domain,
        "a": float(a),
        "Csp": Csp_all,
        "Csp_sum": Csp_sum,
        "Csp_mean": Csp_mean,
        "s_hat": s_hat,
        "s_hat_best": s_hat_best,
        "cn": cn,
        "ct": ct,
        "cn_sum_best": cn_sum_best,
        "ct_sum_best": ct_sum_best,
        "Csp_sum_min": Csp_sum_min,
        "cn_best": cn_best,
        "ct_best": ct_best,
        "Csp_min": Csp_min,
        "i_best": i_best,
        "j_best": j_best,
        "i_sum_best": int(i_sum_best),
        "j_sum_best": int(j_sum_best),
        "sub100": sub100_all,
        "index": index_arr,
        "geom_idx": geom_idx,
        "idx": idx_array,
        "true_cnct": None,
        "i_true": None,
        "j_true": None,
        "s_hat_true": None,
        "Csp_sum_true": None,
    }
    out.update(extra)

    if axes is not None and axes.get("true_cnct") is not None:
        i_t = int(axes["i_true"])
        j_t = int(axes["j_true"])
        out["true_cnct"] = tuple(axes["true_cnct"])
        out["i_true"] = i_t
        out["j_true"] = j_t
        out["s_hat_true"] = float(s_hat[i_t, j_t])
        out["Csp_sum_true"] = float(Csp_sum[i_t, j_t])
    return out
def cnct_search_Rtime(
    r_batch,
    ref_data,
    inc_wave,
    a,
    vp,
    vs,
    rho,
    search_space=((1e-14, 1e-8), (1e-14, 1e-8)),
    nsearch=(30, 30),
    true_cnct=None,
    wave="SS",
):
    """
    Time-domain (cn, ct) search for reflection synthetics via R_wave.

    Like cnct_search_RTomegaV2, but:
      - observed / model in time: Er_time
      - synthetics: R_wave(..., a, cn, ct)  (same a for every trace)
      - for each m=(cn,ct):
            H_i = Er_time_i(m)
            one global s_hat(m) from all traces
            J_r^{(i)} = sum_k [d_i - s_hat * H_i]^2
              (no 1/2, no /||d||^2)
      - Csp[k]     = map of J_r^{(k)}          (n_tr, n_cn, n_ct)
      - Csp_sum    = sum_i J_r^{(i)}           (n_cn, n_ct)
      - Csp_mean   = mean_i J_r^{(i)}          (n_cn, n_ct)
      - s_hat      = s_hat(m) grid             (n_cn, n_ct)
      - s_hat_best = s_hat at argmin(Csp_sum)

    Parameters
    ----------
    r_batch : dict
        Output of R_wave_batch (needs Er_time, idx).
    ref_data, inc_wave :
        Same as for R_wave.
    a : float
        Incident-wave scalar passed to R_wave (same for all traces).
    true_cnct : (cn, ct) or None
        True compliance [m/Pa]. If given, both values are inserted
        into the axes (np.unique keeps order).
    """
    if "Er_time" not in r_batch:
        raise ValueError("r_batch missing 'Er_time'")

    idx_array = np.asarray(r_batch["idx"], int).ravel()
    n_tr = idx_array.size
    if n_tr == 0:
        raise ValueError("r_batch has no traces")

    D_all = np.asarray(r_batch["Er_time"], float)
    if D_all.ndim != 2 or D_all.shape[0] != n_tr:
        raise ValueError("r_batch['Er_time'] must be (n_tr, n_t)")

    axes = cnct_search_axes(search_space, nsearch, true_cnct=true_cnct)
    cn, ct = axes["cn"], axes["ct"]
    n_cn, n_ct = axes["n_cn"], axes["n_ct"] 
    a = float(a)

    Csp_all = np.empty((n_tr, n_cn, n_ct), float)
    s_hat = np.empty((n_cn, n_ct), float)

    nt = D_all.shape[1]
    H_all = np.empty((n_tr, nt), float)

    # grid outer loop: one global s_hat(m) for all traces
    for j, ct_j in enumerate(ct):
        for i, cn_i in enumerate(cn):
            for k, geom in enumerate(idx_array):
                syn = R_wave(
                    ref_data, inc_wave, int(geom), a,
                    vp, vs, rho, cn_i, ct_j, wave=wave,
                )
                h = np.asarray(syn["Er_time"], float).ravel()
                if h.size != nt:
                    raise ValueError(
                        f"Er_time length {h.size} != observed nt={nt}"
                    )
                H_all[k] = h

            s = scale_hat_time(D_all, H_all)
            s_hat[i, j] = s

            for k in range(n_tr):
                Csp_all[k, i, j] = C_time_unnorm(D_all[k], s * H_all[k])
    
    out=cnct_search_pack(
        Csp_all, s_hat, cn, ct, idx_array, a,
        axes=axes,
        mode="R",
        domain="time")
    return out

def scale_hat_omega(D, H):
    """
    Optimal real scalar for complex spectra (all traces):

        s_hat = sum_{i,f} Re(conj(H) D) / sum_{i,f} |H|^2

    D, H : (N, nf) complex
    """
    D = np.asarray(D)
    H = np.asarray(H)
    if D.shape != H.shape:
        raise ValueError(f"D and H shape mismatch: {D.shape} vs {H.shape}")
    if D.ndim != 2:
        raise ValueError("D and H must be 2D (n_tr, n_f)")

    num = float(np.sum(np.real(np.conj(H) * D)))
    den = float(np.sum(np.abs(H) ** 2))
    if den == 0.0:
        raise ValueError("sum |H|^2 is zero; cannot compute s_hat")
    return num / den


def cnct_search_RtimeV2(
    r_batch,
    ref_data,
    inc_wave,
    a,
    vp,
    vs,
    rho,
    search_space=((1e-14, 1e-8), (1e-14, 1e-8)),
    nsearch=(30, 30),
    true_cnct=None,
    wave="SS",
):
    """
    Same as cnct_search_Rtime, but

        s_hat(m) = sum d^2 / sum h(m)^2     (energy ratio, not LS)

    Cost still uses that one global s_hat for all traces:

        J_r^{(i)} = sum_k [d_i - s_hat * h_i]^2
    """
    if "Er_time" not in r_batch:
        raise ValueError("r_batch missing 'Er_time'")

    idx_array = np.asarray(r_batch["idx"], int).ravel()
    n_tr = idx_array.size
    if n_tr == 0:
        raise ValueError("r_batch has no traces")

    D_all = np.asarray(r_batch["Er_time"], float)
    if D_all.ndim != 2 or D_all.shape[0] != n_tr:
        raise ValueError("r_batch['Er_time'] must be (n_tr, n_t)")

    axes = cnct_search_axes(search_space, nsearch, true_cnct=true_cnct)
    cn, ct = axes["cn"], axes["ct"]
    n_cn, n_ct = axes["n_cn"], axes["n_ct"] 
    a = float(a)

    Csp_all = np.empty((n_tr, n_cn, n_ct), float)
    s_hat = np.empty((n_cn, n_ct), float)

    nt = D_all.shape[1]
    H_all = np.empty((n_tr, nt), float)
    d2 = float(np.sum(D_all * D_all))   # numerator, independent of m

    for j, ct_j in enumerate(ct):
        for i, cn_i in enumerate(cn):
            for k, geom in enumerate(idx_array):
                syn = R_wave(
                    ref_data, inc_wave, int(geom), a,
                    vp, vs, rho, cn_i, ct_j, wave=wave,
                )
                h = np.asarray(syn["Er_time"], float).ravel()
                if h.size != nt:
                    raise ValueError(
                        f"Er_time length {h.size} != observed nt={nt}"
                    )
                H_all[k] = h

            den = float(np.sum(H_all * H_all))
            if den == 0.0:
                raise ValueError("sum h^2 is zero; cannot compute s_hat_alt")
            s = float(np.sqrt(d2 / den))
            s_hat[i, j] = s

            for k in range(n_tr):
                Csp_all[k, i, j] = C_time_unnorm(D_all[k], s * H_all[k])

    out=cnct_search_pack(
        Csp_all, s_hat, cn, ct, idx_array, a,
        axes=axes,
        mode="R",
        domain="time",
        s_hat_kind="energy_ratio")
    return out


def s_hat_at_cnct(gsR, cnct):
    """
    Nearest s_hat on the (cn, ct) search grid.

    Parameters
    ----------
    gsR : dict
        Output of cnct_search_Rtime or cnct_search_RtimeV2
        (needs s_hat, cn, ct).
    cnct : (cn, ct)
        Target pair [m/Pa].

    Returns
    -------
    dict
        s_hat, cn, ct at the nearest grid node; i, j indices;
        log10 distances.
    """
    if "s_hat" not in gsR or "cn" not in gsR or "ct" not in gsR:
        raise KeyError("gsR needs 's_hat', 'cn', and 'ct'")

    cn_t, ct_t = map(float, cnct)
    if cn_t <= 0 or ct_t <= 0:
        raise ValueError("cn and ct must be > 0")

    cn = np.asarray(gsR["cn"], float)
    ct = np.asarray(gsR["ct"], float)
    S = np.asarray(gsR["s_hat"], float)

    i = int(np.argmin(np.abs(np.log10(cn) - np.log10(cn_t))))
    j = int(np.argmin(np.abs(np.log10(ct) - np.log10(ct_t))))

    return {
        "s_hat": float(S[i, j]),
        "cn": float(cn[i]),
        "ct": float(ct[j]),
        "cn_query": cn_t,
        "ct_query": ct_t,
        "i": i,
        "j": j,
       
    }

def s_hat_at_cnct_list(gsR, cnct_list):
    """
    Nearest s_hat for many (cn, ct) pairs.

    Parameters
    ----------
    gsR : dict
        Output of cnct_search_Rtime or cnct_search_RtimeV2.
    cnct_list : list of (cn, ct)
        Target pairs [m/Pa].

    Returns
    -------
    dict of 1D arrays (length = n pairs)
        s_hat, cn, ct, cn_query, ct_query, i, j
    """
    cnct_list = list(cnct_list)
    n = len(cnct_list)
    if n == 0:
        raise ValueError("cnct_list is empty")

    s_hat = np.empty(n, float)
    cn = np.empty(n, float)
    ct = np.empty(n, float)
    cn_query = np.empty(n, float)
    ct_query = np.empty(n, float)
    i_arr = np.empty(n, int)
    j_arr = np.empty(n, int)

    for k, pair in enumerate(cnct_list):
        out = s_hat_at_cnct(gsR, pair)
        s_hat[k] = out["s_hat"]
        cn[k] = out["cn"]
        ct[k] = out["ct"]
        cn_query[k] = out["cn_query"]
        ct_query[k] = out["ct_query"]
        i_arr[k] = out["i"]
        j_arr[k] = out["j"]

    return {
        "s_hat": s_hat,
        "cn": cn,
        "ct": ct,
        "cn_query": cn_query,
        "ct_query": ct_query,
        "i": i_arr,
        "j": j_arr,
    }


def s_hat_at_ratio_ct(gsR, ratio_ct):
    """
    Nearest (cn_ratio, ct, s_hat) on a ratio–ct search grid.

    Parameters
    ----------
    gsR : dict
        Output of cnct_search_Rtime_ratio (needs s_hat, cn_ratio, ct).
    ratio_ct : (cn_ratio, ct)
        Target ratio = cn/ct (dimensionless) and ct [m/Pa].

    Returns
    -------
    dict
        s_hat, cn_ratio, ct, cn (= ratio*ct) at nearest node;
        queries and (i, j).
    """
    if "s_hat" not in gsR or "cn_ratio" not in gsR or "ct" not in gsR:
        raise KeyError("gsR needs 's_hat', 'cn_ratio', and 'ct'")

    r_t, ct_t = map(float, ratio_ct)
    if r_t <= 0 or ct_t <= 0:
        raise ValueError("cn_ratio and ct must be > 0")

    cn_ratio = np.asarray(gsR["cn_ratio"], float)
    ct = np.asarray(gsR["ct"], float)
    S = np.asarray(gsR["s_hat"], float)

    i = int(np.argmin(np.abs(np.log10(cn_ratio) - np.log10(r_t))))
    j = int(np.argmin(np.abs(np.log10(ct) - np.log10(ct_t))))

    r = float(cn_ratio[i])
    c = float(ct[j])
    return {
        "s_hat": float(S[i, j]),
        "cn_ratio": r,
        "ct": c,
        "cn": r * c,
        "cn_ratio_query": r_t,
        "ct_query": ct_t,
        "i": i,
        "j": j,
    }


def s_hat_at_ratio_ct_list(gsR, ratio_ct_list):
    """
    Nearest (cn_ratio, ct, s_hat) for many (cn_ratio, ct) pairs.

    Parameters
    ----------
    gsR : dict
        Output of cnct_search_Rtime_ratio.
    ratio_ct_list : list of (cn_ratio, ct)
        Target pairs: ratio = cn/ct, ct [m/Pa].

    Returns
    -------
    dict of 1D arrays (length = n pairs)
        s_hat, cn_ratio, ct, cn, cn_ratio_query, ct_query, i, j
    """
    ratio_ct_list = list(ratio_ct_list)
    n = len(ratio_ct_list)
    if n == 0:
        raise ValueError("ratio_ct_list is empty")

    s_hat = np.empty(n, float)
    cn_ratio = np.empty(n, float)
    ct = np.empty(n, float)
    cn = np.empty(n, float)
    cn_ratio_query = np.empty(n, float)
    ct_query = np.empty(n, float)
    i_arr = np.empty(n, int)
    j_arr = np.empty(n, int)

    for k, pair in enumerate(ratio_ct_list):
        out = s_hat_at_ratio_ct(gsR, pair)
        s_hat[k] = out["s_hat"]
        cn_ratio[k] = out["cn_ratio"]
        ct[k] = out["ct"]
        cn[k] = out["cn"]
        cn_ratio_query[k] = out["cn_ratio_query"]
        ct_query[k] = out["ct_query"]
        i_arr[k] = out["i"]
        j_arr[k] = out["j"]

    return {
        "s_hat": s_hat,
        "cn_ratio": cn_ratio,
        "ct": ct,
        "cn": cn,
        "cn_ratio_query": cn_ratio_query,
        "ct_query": ct_query,
        "i": i_arr,
        "j": j_arr,
    }


def lag_at_list(gsR, i_arr, j_arr, idx):
    """
    Per-trace cc lag (samples) at already-known grid indices, for one
    specific trace geometry index `idx`.

    s_hat/cn/ct from a cnct_search_*_Gd_cc result (cnct_search_Rtime_Gd_cc
    or cnct_search_Rtime_ratio_Gd_cc) are GLOBAL: one value per (i, j)
    grid point, fit jointly across every trace used by the search (see
    scale_hat_time -- s_hat there is not per-trace). The per-grid-point cc
    realignment lag, in contrast, IS per-trace (lag_samples has shape
    (n_tr, n_axis1, n_axis2)). Bundling a per-trace `idx` into a lookup
    that otherwise returns global (s_hat, cn, ct) is misleading -- this
    function keeps that per-trace lookup separate: pass it the (i, j)
    grid indices already returned by s_hat_at_cnct_list (plain grid) or
    s_hat_at_ratio_ct_list (ratio grid), plus the trace's geometry index,
    and it returns just that trace's lag at each of those points.

    Parameters
    ----------
    gsR : dict
        Output of cnct_search_Rtime_Gd_cc or cnct_search_Rtime_ratio_Gd_cc
        (needs 'lag_samples' and 'idx', the geometry indices used by the
        search).
    i_arr, j_arr : array-like of int
        Grid indices, e.g. s_hat_at_cnct_list(...)['i']/['j'] or
        s_hat_at_ratio_ct_list(...)['i']/['j'].
    idx : int
        Geometry index (as in gsR['idx']/ref_data arrays) of the trace to
        retrieve the cc lag for.

    Returns
    -------
    (n,) int array
        lag (samples), one per (i, j) pair, for this trace.
    """
    if "lag_samples" not in gsR or "idx" not in gsR:
        raise KeyError(
            "gsR needs 'lag_samples' and 'idx' (a cnct_search_*_Gd_cc result)"
        )

    lag_samples = np.asarray(gsR["lag_samples"])
    idx_array = np.asarray(gsR["idx"], int).ravel()

    idx = int(idx)
    matches = np.where(idx_array == idx)[0]
    if matches.size == 0:
        raise ValueError(f"idx={idx} not found in gsR['idx']")
    k = int(matches[0])

    i_arr = np.asarray(i_arr, int).ravel()
    j_arr = np.asarray(j_arr, int).ravel()
    if i_arr.size != j_arr.size:
        raise ValueError("i_arr and j_arr must have the same length")

    return lag_samples[k, i_arr, j_arr].astype(int)


def ricker(time, t0, f0):
    """
    Ricker wavelet on a 1-D time axis, plus FFT.

        w(t) = [1 - 2 (π f0 (t - t0))²] exp[-(π f0 (t - t0))²]

    Parameters
    ----------
    time : array-like
        Time samples [s], 1-D, uniformly sampled.
    t0 : float
        Peak time [s].
    f0 : float
        Peak frequency [Hz].

    Returns
    -------
    dict
        trace : 1-D float wavelet
        time  : 1-D time axis [s]
        fft   : 1-D complex FFT (fftfreq order)
        freqs : 1-D frequency axis [Hz]
    """
    time = np.asarray(time, float).ravel()
    t0 = float(t0)
    f0 = float(f0)
    if f0 <= 0:
        raise ValueError("f0 must be > 0")
    if time.size < 2:
        raise ValueError("time must have at least 2 samples")

    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        raise ValueError("time axis must be increasing (dt > 0)")

    tau = np.pi * f0 * (time - t0)
    tau2 = tau * tau
    w = (1.0 - 2.0 * tau2) * np.exp(-tau2)

    n = w.size
    W = fft(w, n=n)
    freqs = fftfreq(n, d=dt)

    return {
        "trace": w,
        "time": time,
        "fft": W,
        "freqs": freqs,
    }

# ---------------------------------------------------------------------
# Gd-aware (R_wave_v2-based) counterparts of the cnct_search_* functions
# above. Those all call R_wave(...), which needs a separately-estimated,
# absolute-amplitude source spectrum S(omega) (e.g. via DAS_inversion.
# Estimate_stf). The functions below call R_wave_v2(...) instead, which
# takes the RAW (undeconvolved) direct-arrival spectrum directly and
# performs its own Gd-based deconvolution internally -- see the
# DASrefInv.ipynb / DASrefModel_source.ipynb discussion on not mixing an
# Estimate_stf-deconvolved S(omega) with R_wave_v2's own Gd division.
# Each one is otherwise identical to its R_wave counterpart (same cost
# function, same s_hat logic, same output keys) -- only the forward
# model call changes, so results are directly comparable.
# ---------------------------------------------------------------------

def cnct_search_Rtime_Gd(
    r_batch,
    ref_data,
    inc_wave,
    a,
    vp,
    vs,
    rho,
    Gd,
    eps=0.005,
    search_space=((1e-14, 1e-8), (1e-14, 1e-8)),
    nsearch=(30, 30),
    true_cnct=None,
    wave="SS",
):
    """
    Same as cnct_search_Rtime, but via R_wave_v2 (Gd-deconvolution implicit,
    inc_wave should be the RAW direct-arrival trace, not an Estimate_stf
    output). See module docstring above.
    """
    if "Er_time" not in r_batch:
        raise ValueError("r_batch missing 'Er_time'")

    idx_array = np.asarray(r_batch["idx"], int).ravel()
    n_tr = idx_array.size
    if n_tr == 0:
        raise ValueError("r_batch has no traces")

    D_all = np.asarray(r_batch["Er_time"], float)
    if D_all.ndim != 2 or D_all.shape[0] != n_tr:
        raise ValueError("r_batch['Er_time'] must be (n_tr, n_t)")

    axes = cnct_search_axes(search_space, nsearch, true_cnct=true_cnct)
    cn, ct = axes["cn"], axes["ct"]
    n_cn, n_ct = axes["n_cn"], axes["n_ct"]
    a = float(a)

    Csp_all = np.empty((n_tr, n_cn, n_ct), float)
    s_hat = np.empty((n_cn, n_ct), float)

    nt = D_all.shape[1]
    H_all = np.empty((n_tr, nt), float)

    for j, ct_j in enumerate(ct):
        for i, cn_i in enumerate(cn):
            for k, geom in enumerate(idx_array):
                syn = R_wave_v2(
                    ref_data, inc_wave, int(geom), a,
                    vp, vs, rho, cn_i, ct_j,
                    Gd=Gd, eps=eps, wave=wave,
                )
                h = np.asarray(syn["Er_time"], float).ravel()
                if h.size != nt:
                    raise ValueError(
                        f"Er_time length {h.size} != observed nt={nt}"
                    )
                H_all[k] = h

            s = scale_hat_time(D_all, H_all)
            s_hat[i, j] = s

            for k in range(n_tr):
                Csp_all[k, i, j] = C_time_unnorm(D_all[k], s * H_all[k])

    out = cnct_search_pack(
        Csp_all, s_hat, cn, ct, idx_array, a,
        axes=axes,
        mode="R",
        domain="time")
    return out


def cnct_search_RtimeV2_Gd(
    r_batch,
    ref_data,
    inc_wave,
    a,
    vp,
    vs,
    rho,
    Gd,
    eps=0.005,
    search_space=((1e-14, 1e-8), (1e-14, 1e-8)),
    nsearch=(30, 30),
    true_cnct=None,
    wave="SS",
):
    """
    Same as cnct_search_RtimeV2 (energy-ratio s_hat), but via R_wave_v2.
    See cnct_search_Rtime_Gd / module docstring above.
    """
    if "Er_time" not in r_batch:
        raise ValueError("r_batch missing 'Er_time'")

    idx_array = np.asarray(r_batch["idx"], int).ravel()
    n_tr = idx_array.size
    if n_tr == 0:
        raise ValueError("r_batch has no traces")

    D_all = np.asarray(r_batch["Er_time"], float)
    if D_all.ndim != 2 or D_all.shape[0] != n_tr:
        raise ValueError("r_batch['Er_time'] must be (n_tr, n_t)")

    axes = cnct_search_axes(search_space, nsearch, true_cnct=true_cnct)
    cn, ct = axes["cn"], axes["ct"]
    n_cn, n_ct = axes["n_cn"], axes["n_ct"]
    a = float(a)

    Csp_all = np.empty((n_tr, n_cn, n_ct), float)
    s_hat = np.empty((n_cn, n_ct), float)

    nt = D_all.shape[1]
    H_all = np.empty((n_tr, nt), float)
    d2 = float(np.sum(D_all * D_all))

    for j, ct_j in enumerate(ct):
        for i, cn_i in enumerate(cn):
            for k, geom in enumerate(idx_array):
                syn = R_wave_v2(
                    ref_data, inc_wave, int(geom), a,
                    vp, vs, rho, cn_i, ct_j,
                    Gd=Gd, eps=eps, wave=wave,
                )
                h = np.asarray(syn["Er_time"], float).ravel()
                if h.size != nt:
                    raise ValueError(
                        f"Er_time length {h.size} != observed nt={nt}"
                    )
                H_all[k] = h

            den = float(np.sum(H_all * H_all))
            if den == 0.0:
                raise ValueError("sum h^2 is zero; cannot compute s_hat_alt")
            s = float(np.sqrt(d2 / den))
            s_hat[i, j] = s

            for k in range(n_tr):
                Csp_all[k, i, j] = C_time_unnorm(D_all[k], s * H_all[k])

    out = cnct_search_pack(
        Csp_all, s_hat, cn, ct, idx_array, a,
        axes=axes,
        mode="R",
        domain="time",
        s_hat_kind="energy_ratio")
    return out


def cnct_search_Romega_Gd(
    r_batch,
    ref_data,
    inc_wave,
    a,
    vp,
    vs,
    rho,
    Gd,
    eps=0.005,
    search_space=((1e-14, 1e-8), (1e-14, 1e-8)),
    nsearch=(30, 30),
    wave="SS",
):
    """
    Same as cnct_search_Romega (frequency domain), but via R_wave_v2.
    See cnct_search_Rtime_Gd / module docstring above.
    """
    if "Er_omega" not in r_batch:
        raise ValueError("r_batch missing 'Er_omega'")

    idx_array = np.asarray(r_batch["idx"], int).ravel()
    n_tr = idx_array.size
    if n_tr == 0:
        raise ValueError("r_batch has no traces")

    D_all = np.asarray(r_batch["Er_omega"])
    if D_all.ndim != 2 or D_all.shape[0] != n_tr:
        raise ValueError("r_batch['Er_omega'] must be (n_tr, n_f)")

    try:
        n_cn, n_ct = int(nsearch[0]), int(nsearch[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("nsearch must be (n_cn, n_ct)") from exc
    if n_cn < 2 or n_ct < 2:
        raise ValueError("n_cn and n_ct must be >= 2")

    try:
        (cn_min, cn_max), (ct_min, ct_max) = search_space
        cn_min, cn_max = float(cn_min), float(cn_max)
        ct_min, ct_max = float(ct_min), float(ct_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "search_space must be [(cn_min, cn_max), (ct_min, ct_max)]"
        ) from exc
    if not (cn_min > 0 and cn_max > cn_min):
        raise ValueError("need 0 < cn_min < cn_max")
    if not (ct_min > 0 and ct_max > ct_min):
        raise ValueError("need 0 < ct_min < ct_max")

    cn = np.logspace(np.log10(cn_min), np.log10(cn_max), n_cn)
    ct = np.logspace(np.log10(ct_min), np.log10(ct_max), n_ct)
    a = float(a)

    Csp_all = np.empty((n_tr, n_cn, n_ct), float)
    s_hat = np.empty((n_cn, n_ct), float)

    nf = D_all.shape[1]
    H_all = np.empty((n_tr, nf), dtype=complex)

    for j, ct_j in enumerate(ct):
        for i, cn_i in enumerate(cn):
            for k, geom in enumerate(idx_array):
                syn = R_wave_v2(
                    ref_data, inc_wave, int(geom), a,
                    vp, vs, rho, cn_i, ct_j,
                    Gd=Gd, eps=eps, wave=wave,
                )
                h = np.asarray(syn["Er_omega"]).ravel()
                if h.size != nf:
                    raise ValueError(
                        f"Er_omega length {h.size} != observed nf={nf}"
                    )
                H_all[k] = h

            s = scale_hat_omega(D_all, H_all)
            s_hat[i, j] = s

            for k in range(n_tr):
                Csp_all[k, i, j] = C_spectrum_amplitude(D_all[k], s * H_all[k])

    ct_best = np.empty(n_tr, float)
    cn_best = np.empty(n_tr, float)
    Csp_min = np.empty(n_tr, float)
    i_best = np.empty(n_tr, int)
    j_best = np.empty(n_tr, int)
    geom_idx = np.asarray(idx_array, int).copy()
    index_arr = np.arange(n_tr, dtype=int)
    n_top = min(100, n_cn * n_ct)
    sub100_all = np.empty((n_tr, n_top, 3), float)

    for k in range(n_tr):
        Csp = Csp_all[k]
        ii0, jj0 = np.unravel_index(np.argmin(Csp), Csp.shape)
        cn_best[k] = float(cn[ii0])
        ct_best[k] = float(ct[jj0])
        Csp_min[k] = float(Csp[ii0, jj0])
        i_best[k] = int(ii0)
        j_best[k] = int(jj0)

        flat = Csp.ravel()
        idx_flat = np.argpartition(flat, n_top - 1)[:n_top]
        idx_flat = idx_flat[np.argsort(flat[idx_flat])]
        ii, jj = np.unravel_index(idx_flat, Csp.shape)
        sub100_all[k] = np.column_stack([cn[ii], ct[jj], flat[idx_flat]])

    Csp_sum = np.sum(Csp_all, axis=0)
    Csp_mean = np.mean(Csp_all, axis=0)
    i_sum_best, j_sum_best = np.unravel_index(np.argmin(Csp_sum), Csp_sum.shape)
    cn_sum_best = float(cn[i_sum_best])
    ct_sum_best = float(ct[j_sum_best])
    Csp_sum_min = float(Csp_sum[i_sum_best, j_sum_best])
    s_hat_best = float(s_hat[i_sum_best, j_sum_best])

    return {
        "mode": "R",
        "domain": "freq",
        "a": a,
        "Csp": Csp_all,
        "Csp_sum": Csp_sum,
        "Csp_mean": Csp_mean,
        "s_hat": s_hat,
        "s_hat_best": s_hat_best,
        "cn": cn,
        "ct": ct,
        "cn_sum_best": cn_sum_best,
        "ct_sum_best": ct_sum_best,
        "Csp_sum_min": Csp_sum_min,
        "cn_best": cn_best,
        "ct_best": ct_best,
        "Csp_min": Csp_min,
        "i_best": i_best,
        "j_best": j_best,
        "i_sum_best": int(i_sum_best),
        "j_sum_best": int(j_sum_best),
        "sub100": sub100_all,
        "index": index_arr,
        "geom_idx": geom_idx,
        "idx": idx_array,
    }


def cnct_search_Rtime_ratio_Gd(
    r_batch,
    ref_data,
    inc_wave,
    a,
    vp,
    vs,
    rho,
    Gd,
    eps=0.005,
    search_space=((0.01, 3.0), (1e-14, 1e-8)),
    nsearch=(30, 30),
    true_cnct=None,
    wave="SS",
):
    """
    Same as cnct_search_Rtime_ratio (searches (cn/ct ratio, ct)), but via
    R_wave_v2. See cnct_search_Rtime_Gd / module docstring above.
    """
    if "Er_time" not in r_batch:
        raise ValueError("r_batch missing 'Er_time'")

    idx_array = np.asarray(r_batch["idx"], int).ravel()
    n_tr = idx_array.size
    if n_tr == 0:
        raise ValueError("r_batch has no traces")

    D_all = np.asarray(r_batch["Er_time"], float)
    if D_all.ndim != 2 or D_all.shape[0] != n_tr:
        raise ValueError("r_batch['Er_time'] must be (n_tr, n_t)")

    try:
        n_ratio, n_ct = int(nsearch[0]), int(nsearch[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("nsearch must be (n_ratio, n_ct)") from exc
    if n_ratio < 2 or n_ct < 2:
        raise ValueError("n_ratio and n_ct must be >= 2")

    try:
        (r_min, r_max), (ct_min, ct_max) = search_space
        r_min, r_max = float(r_min), float(r_max)
        ct_min, ct_max = float(ct_min), float(ct_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "search_space must be [(ratio_min, ratio_max), (ct_min, ct_max)]"
        ) from exc
    if not (r_min > 0 and r_max > r_min):
        raise ValueError("need 0 < ratio_min < ratio_max")
    if not (ct_min > 0 and ct_max > ct_min):
        raise ValueError("need 0 < ct_min < ct_max")

    cn_ratio = np.logspace(np.log10(r_min), np.log10(r_max), n_ratio)
    ct = np.logspace(np.log10(ct_min), np.log10(ct_max), n_ct)

    i_true = j_true = None
    true_pair = None
    ratio_true = None
    if true_cnct is not None:
        cn_t, ct_t = map(float, true_cnct)
        if cn_t <= 0 or ct_t <= 0:
            raise ValueError("true_cnct values must be > 0")
        ratio_true = cn_t / ct_t
        cn_ratio = np.unique(np.sort(np.append(cn_ratio, ratio_true)))
        ct = np.unique(np.sort(np.append(ct, ct_t)))
        true_pair = (cn_t, ct_t)
        i_true = int(np.argmin(np.abs(np.log10(cn_ratio) - np.log10(ratio_true))))
        j_true = int(np.argmin(np.abs(np.log10(ct) - np.log10(ct_t))))

    n_ratio, n_ct = int(cn_ratio.size), int(ct.size)
    a = float(a)

    Csp_all = np.empty((n_tr, n_ratio, n_ct), float)
    s_hat = np.empty((n_ratio, n_ct), float)
    cn_grid = np.empty((n_ratio, n_ct), float)

    nt = D_all.shape[1]
    H_all = np.empty((n_tr, nt), float)

    for j, ct_j in enumerate(ct):
        for i, r in enumerate(cn_ratio):
            cn_i = r * ct_j
            cn_grid[i, j] = cn_i
            for k, geom in enumerate(idx_array):
                syn = R_wave_v2(
                    ref_data, inc_wave, int(geom), a,
                    vp, vs, rho, cn_i, ct_j,
                    Gd=Gd, eps=eps, wave=wave,
                )
                h = np.asarray(syn["Er_time"], float).ravel()
                if h.size != nt:
                    raise ValueError(
                        f"Er_time length {h.size} != observed nt={nt}"
                    )
                H_all[k] = h

            s = scale_hat_time(D_all, H_all)
            s_hat[i, j] = s
            for k in range(n_tr):
                Csp_all[k, i, j] = C_time_unnorm(D_all[k], s * H_all[k])

    ct_best = np.empty(n_tr, float)
    cn_best = np.empty(n_tr, float)
    ratio_best = np.empty(n_tr, float)
    Csp_min = np.empty(n_tr, float)
    i_best = np.empty(n_tr, int)
    j_best = np.empty(n_tr, int)
    geom_idx = np.asarray(idx_array, int).copy()
    index_arr = np.arange(n_tr, dtype=int)
    n_top = min(100, n_ratio * n_ct)
    sub100_all = np.empty((n_tr, n_top, 3), float)

    for k in range(n_tr):
        Csp = Csp_all[k]
        ii0, jj0 = np.unravel_index(np.argmin(Csp), Csp.shape)
        ratio_best[k] = float(cn_ratio[ii0])
        ct_best[k] = float(ct[jj0])
        cn_best[k] = float(cn_ratio[ii0] * ct[jj0])
        Csp_min[k] = float(Csp[ii0, jj0])
        i_best[k] = int(ii0)
        j_best[k] = int(jj0)

        flat = Csp.ravel()
        idx_flat = np.argpartition(flat, n_top - 1)[:n_top]
        idx_flat = idx_flat[np.argsort(flat[idx_flat])]
        ii, jj = np.unravel_index(idx_flat, Csp.shape)
        sub100_all[k] = np.column_stack([
            cn_ratio[ii] * ct[jj],
            ct[jj],
            flat[idx_flat],
        ])

    Csp_sum = np.sum(Csp_all, axis=0)
    Csp_mean = np.mean(Csp_all, axis=0)
    i_sum_best, j_sum_best = np.unravel_index(np.argmin(Csp_sum), Csp_sum.shape)
    ratio_sum_best = float(cn_ratio[i_sum_best])
    ct_sum_best = float(ct[j_sum_best])
    cn_sum_best = float(ratio_sum_best * ct_sum_best)
    Csp_sum_min = float(Csp_sum[i_sum_best, j_sum_best])
    s_hat_best = float(s_hat[i_sum_best, j_sum_best])

    out = {
        "mode": "R",
        "domain": "time",
        "grid": "ratio_ct",
        "a": a,
        "Csp": Csp_all,
        "Csp_sum": Csp_sum,
        "Csp_mean": Csp_mean,
        "s_hat": s_hat,
        "s_hat_best": s_hat_best,
        "cn_ratio": cn_ratio,
        "ct": ct,
        "cn_grid": cn_grid,
        "cn_sum_best": cn_sum_best,
        "ct_sum_best": ct_sum_best,
        "ratio_sum_best": ratio_sum_best,
        "Csp_sum_min": Csp_sum_min,
        "cn_best": cn_best,
        "ct_best": ct_best,
        "ratio_best": ratio_best,
        "Csp_min": Csp_min,
        "i_best": i_best,
        "j_best": j_best,
        "i_sum_best": int(i_sum_best),
        "j_sum_best": int(j_sum_best),
        "sub100": sub100_all,
        "index": index_arr,
        "geom_idx": geom_idx,
        "idx": idx_array,
        "true_cnct": true_pair,
        "ratio_true": ratio_true,
        "i_true": i_true,
        "j_true": j_true,
        "s_hat_true": None,
        "Csp_sum_true": None,
    }
    if true_pair is not None:
        out["s_hat_true"] = float(s_hat[i_true, j_true])
        out["Csp_sum_true"] = float(Csp_sum[i_true, j_true])
    return out


def cnct_search_Rtime_Gd_cc(
    r_batch,
    ref_data,
    inc_wave,
    a,
    vp,
    vs,
    rho,
    Gd,
    eps=0.005,
    search_space=((1e-14, 1e-8), (1e-14, 1e-8)),
    nsearch=(30, 30),
    true_cnct=None,
    wave="SS",
    max_lag=None,
    align_fn=None,
):
    """
    Same as cnct_search_Rtime_Gd, but at every (cn, ct) grid point each
    candidate model trace is first realigned to the observed trace (by
    default via full-trace cross-correlation, DAS_inversion.align_by_cc)
    before s_hat and the cost function are computed. This corrects for a
    timing offset between the forward model and the data (e.g. picking/
    windowing uncertainty) so the misfit reflects waveform SHAPE rather
    than an uncorrected time shift. See cnct_search_Rtime_Gd / module
    docstring.

    `align_fn` (default None -> DAS_inversion.align_by_cc) selects the
    realignment method; pass DAS_inversion.align_by_peak instead to align
    by the dominant-peak location rather than whole-trace correlation --
    see align_by_peak's docstring for why this is often the more robust
    choice (whole-trace correlation can be dominated by a compliance-
    dependent coda/secondary-lobe mismatch that is real, informative
    physics, not a timing error, and chase a large, misleading shift even
    when the dominant arrival is already well-timed). Any callable with
    the signature (trace, reference, max_lag=...) -> (aligned, lag) works.

    `max_lag` (samples, default None = unrestricted) caps the search
    range -- see align_by_cc's / align_by_peak's docstrings. Recommended:
    set this to a small value (a few ms) rather than leaving it
    unrestricted, especially for grid points far from the true compliance,
    where the candidate's shape/polarity can genuinely differ from the
    data (real physics, not a
    timing error) and an unrestricted search will chase a large,
    physically implausible shift instead of reporting an honest poor fit.
    """
    if "Er_time" not in r_batch:
        raise ValueError("r_batch missing 'Er_time'")

    idx_array = np.asarray(r_batch["idx"], int).ravel()
    n_tr = idx_array.size
    if n_tr == 0:
        raise ValueError("r_batch has no traces")

    D_all = np.asarray(r_batch["Er_time"], float)
    if D_all.ndim != 2 or D_all.shape[0] != n_tr:
        raise ValueError("r_batch['Er_time'] must be (n_tr, n_t)")

    align_fn = align_fn or align_by_cc

    axes = cnct_search_axes(search_space, nsearch, true_cnct=true_cnct)
    cn, ct = axes["cn"], axes["ct"]
    n_cn, n_ct = axes["n_cn"], axes["n_ct"]
    a = float(a)

    Csp_all = np.empty((n_tr, n_cn, n_ct), float)
    s_hat = np.empty((n_cn, n_ct), float)
    lag_samples = np.empty((n_tr, n_cn, n_ct), int)

    nt = D_all.shape[1]
    H_all = np.empty((n_tr, nt), float)

    for j, ct_j in enumerate(ct):
        for i, cn_i in enumerate(cn):
            for k, geom in enumerate(idx_array):
                syn = R_wave_v2(
                    ref_data, inc_wave, int(geom), a,
                    vp, vs, rho, cn_i, ct_j,
                    Gd=Gd, eps=eps, wave=wave,
                )
                h = np.asarray(syn["Er_time"], float).ravel()
                if h.size != nt:
                    raise ValueError(
                        f"Er_time length {h.size} != observed nt={nt}"
                    )
                h_cc, lag = align_fn(h, D_all[k], max_lag=max_lag)
                H_all[k] = h_cc
                lag_samples[k, i, j] = lag

            s = scale_hat_time(D_all, H_all)
            s_hat[i, j] = s

            for k in range(n_tr):
                Csp_all[k, i, j] = C_time_unnorm(D_all[k], s * H_all[k])

    out = cnct_search_pack(
        Csp_all, s_hat, cn, ct, idx_array, a,
        axes=axes,
        mode="R",
        domain="time",
        lag_samples=lag_samples,
    )
    return out


def cnct_search_Rtime_ratio_Gd_cc(
    r_batch,
    ref_data,
    inc_wave,
    a,
    vp,
    vs,
    rho,
    Gd,
    eps=0.005,
    search_space=((0.01, 3.0), (1e-14, 1e-8)),
    nsearch=(30, 30),
    true_cnct=None,
    wave="SS",
    max_lag=None,
    align_fn=None,
):
    """
    Same as cnct_search_Rtime_ratio_Gd (searches (cn/ct ratio, ct)), but at
    every grid point each candidate model trace is first realigned to the
    observed trace (by default via full-trace cross-correlation,
    DAS_inversion.align_by_cc) before s_hat and the cost function are
    computed. See cnct_search_Rtime_Gd_cc / cnct_search_Rtime_ratio_Gd.

    `align_fn` / `max_lag`: see cnct_search_Rtime_Gd_cc's docstring --
    pass DAS_inversion.align_by_peak as `align_fn` for the more robust,
    peak-location-based realignment discussed there.
    """
    if "Er_time" not in r_batch:
        raise ValueError("r_batch missing 'Er_time'")

    idx_array = np.asarray(r_batch["idx"], int).ravel()
    n_tr = idx_array.size
    if n_tr == 0:
        raise ValueError("r_batch has no traces")

    align_fn = align_fn or align_by_cc

    D_all = np.asarray(r_batch["Er_time"], float)
    if D_all.ndim != 2 or D_all.shape[0] != n_tr:
        raise ValueError("r_batch['Er_time'] must be (n_tr, n_t)")

    try:
        n_ratio, n_ct = int(nsearch[0]), int(nsearch[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("nsearch must be (n_ratio, n_ct)") from exc
    if n_ratio < 2 or n_ct < 2:
        raise ValueError("n_ratio and n_ct must be >= 2")

    try:
        (r_min, r_max), (ct_min, ct_max) = search_space
        r_min, r_max = float(r_min), float(r_max)
        ct_min, ct_max = float(ct_min), float(ct_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "search_space must be [(ratio_min, ratio_max), (ct_min, ct_max)]"
        ) from exc
    if not (r_min > 0 and r_max > r_min):
        raise ValueError("need 0 < ratio_min < ratio_max")
    if not (ct_min > 0 and ct_max > ct_min):
        raise ValueError("need 0 < ct_min < ct_max")

    cn_ratio = np.logspace(np.log10(r_min), np.log10(r_max), n_ratio)
    ct = np.logspace(np.log10(ct_min), np.log10(ct_max), n_ct)

    i_true = j_true = None
    true_pair = None
    ratio_true = None
    if true_cnct is not None:
        cn_t, ct_t = map(float, true_cnct)
        if cn_t <= 0 or ct_t <= 0:
            raise ValueError("true_cnct values must be > 0")
        ratio_true = cn_t / ct_t
        cn_ratio = np.unique(np.sort(np.append(cn_ratio, ratio_true)))
        ct = np.unique(np.sort(np.append(ct, ct_t)))
        true_pair = (cn_t, ct_t)
        i_true = int(np.argmin(np.abs(np.log10(cn_ratio) - np.log10(ratio_true))))
        j_true = int(np.argmin(np.abs(np.log10(ct) - np.log10(ct_t))))

    n_ratio, n_ct = int(cn_ratio.size), int(ct.size)
    a = float(a)

    Csp_all = np.empty((n_tr, n_ratio, n_ct), float)
    s_hat = np.empty((n_ratio, n_ct), float)
    cn_grid = np.empty((n_ratio, n_ct), float)
    lag_samples = np.empty((n_tr, n_ratio, n_ct), int)

    nt = D_all.shape[1]
    H_all = np.empty((n_tr, nt), float)

    for j, ct_j in enumerate(ct):
        for i, r in enumerate(cn_ratio):
            cn_i = r * ct_j
            cn_grid[i, j] = cn_i
            for k, geom in enumerate(idx_array):
                syn = R_wave_v2(
                    ref_data, inc_wave, int(geom), a,
                    vp, vs, rho, cn_i, ct_j,
                    Gd=Gd, eps=eps, wave=wave,
                )
                h = np.asarray(syn["Er_time"], float).ravel()
                if h.size != nt:
                    raise ValueError(
                        f"Er_time length {h.size} != observed nt={nt}"
                    )
                h_cc, lag = align_fn(h, D_all[k], max_lag=max_lag)
                H_all[k] = h_cc
                lag_samples[k, i, j] = lag

            s = scale_hat_time(D_all, H_all)
            s_hat[i, j] = s
            for k in range(n_tr):
                Csp_all[k, i, j] = C_time_unnorm(D_all[k], s * H_all[k])

    ct_best = np.empty(n_tr, float)
    cn_best = np.empty(n_tr, float)
    ratio_best = np.empty(n_tr, float)
    Csp_min = np.empty(n_tr, float)
    i_best = np.empty(n_tr, int)
    j_best = np.empty(n_tr, int)
    geom_idx = np.asarray(idx_array, int).copy()
    index_arr = np.arange(n_tr, dtype=int)
    n_top = min(100, n_ratio * n_ct)
    sub100_all = np.empty((n_tr, n_top, 3), float)

    for k in range(n_tr):
        Csp = Csp_all[k]
        ii0, jj0 = np.unravel_index(np.argmin(Csp), Csp.shape)
        ratio_best[k] = float(cn_ratio[ii0])
        ct_best[k] = float(ct[jj0])
        cn_best[k] = float(cn_ratio[ii0] * ct[jj0])
        Csp_min[k] = float(Csp[ii0, jj0])
        i_best[k] = int(ii0)
        j_best[k] = int(jj0)

        flat = Csp.ravel()
        idx_flat = np.argpartition(flat, n_top - 1)[:n_top]
        idx_flat = idx_flat[np.argsort(flat[idx_flat])]
        ii, jj = np.unravel_index(idx_flat, Csp.shape)
        sub100_all[k] = np.column_stack([
            cn_ratio[ii] * ct[jj],
            ct[jj],
            flat[idx_flat],
        ])

    Csp_sum = np.sum(Csp_all, axis=0)
    Csp_mean = np.mean(Csp_all, axis=0)
    i_sum_best, j_sum_best = np.unravel_index(np.argmin(Csp_sum), Csp_sum.shape)
    ratio_sum_best = float(cn_ratio[i_sum_best])
    ct_sum_best = float(ct[j_sum_best])
    cn_sum_best = float(ratio_sum_best * ct_sum_best)
    Csp_sum_min = float(Csp_sum[i_sum_best, j_sum_best])
    s_hat_best = float(s_hat[i_sum_best, j_sum_best])

    out = {
        "mode": "R",
        "domain": "time",
        "grid": "ratio_ct",
        "a": a,
        "Csp": Csp_all,
        "Csp_sum": Csp_sum,
        "Csp_mean": Csp_mean,
        "s_hat": s_hat,
        "s_hat_best": s_hat_best,
        "cn_ratio": cn_ratio,
        "ct": ct,
        "cn_grid": cn_grid,
        "cn_sum_best": cn_sum_best,
        "ct_sum_best": ct_sum_best,
        "ratio_sum_best": ratio_sum_best,
        "Csp_sum_min": Csp_sum_min,
        "cn_best": cn_best,
        "ct_best": ct_best,
        "ratio_best": ratio_best,
        "Csp_min": Csp_min,
        "i_best": i_best,
        "j_best": j_best,
        "i_sum_best": int(i_sum_best),
        "j_sum_best": int(j_sum_best),
        "sub100": sub100_all,
        "index": index_arr,
        "geom_idx": geom_idx,
        "idx": idx_array,
        "true_cnct": true_pair,
        "ratio_true": ratio_true,
        "i_true": i_true,
        "j_true": j_true,
        "s_hat_true": None,
        "Csp_sum_true": None,
        "lag_samples": lag_samples,
    }
    if true_pair is not None:
        out["s_hat_true"] = float(s_hat[i_true, j_true])
        out["Csp_sum_true"] = float(Csp_sum[i_true, j_true])
    return out


def cnct_search_Rtime_Gd_cc_obs(
    ref_data,
    inc_data,
    idx_array,
    medium,
    search_param,
    true_cnct=None,
):
    """
    Time-domain (cn, ct) grid search fitting REAL observed reflection data
    (ref_data['reflection_traces'].data), not a synthetic r_batch. Mirrors
    cnct_search_Rtime_Gd_cc, with the differences needed for real-data use:

      - `cc` (the last/4th element of search_param) selects the
        alignment's lag-picking rule, passed straight through to
        DAS_inversion.align_by_window_cc: "signed" (default) =
        argmax(rho), same-polarity match only; "abs" = argmax(|rho|), a
        strong negative correlation is also accepted (see
        align_by_window_cc's docstring for the trade-off);

      - the observed data is read directly from
        ref_data['reflection_traces'] -- there is no synthetic "truth" to
        precompute, so no r_batch argument is needed;
      - the forward-model scalar is fixed at a=1.0 (the free amplitude fit
        per grid point is s_hat, not a second free scalar -- see
        cnct_search_Rtime_Gd_cc's docstring on why a and s_hat would
        otherwise be redundant);
      - inc_data[0] (inc_wave) must be the RAW direct-arrival trace (e.g.
        dir_data['stf_trace']), NOT an Estimate_stf output -- R_wave_v2
        does its own Gd-deconvolution internally, so an already-deconvolved
        STF would be double-deconvolved;
      - each candidate model trace is realigned to the observed trace via
        DAS_inversion.align_by_window_cc, restricted to the data's own
        fixed extraction window (ref_data['time_window'], a (tbegin,tend)
        tuple set in DASrefData.ipynb's windowed_das call) -- this is the
        default and only alignment method here (not swappable via a
        parameter): whole-trace correlation (align_by_cc) chases the
        compliance-dependent coda instead of the dominant arrival, and a
        single-sample peak pick (align_by_peak) is fragile against this
        dataset's genuine ~4-5 ms two-comparable-lobe ambiguity (both
        confirmed empirically -- see
        tests/cc_alignment_for_inversion_notes.md). align_by_window_cc
        aggregates the whole windowed shape (both lobes) and explicitly
        considers both polarities, which fixed both failure modes;
      - optional trace selection / weighting via trace_opts, exactly as in
        cnct_ratiosearch_Rtime_obs (same _apply_trace_opts; weights
        normalized to mean 1 over the kept traces; all ones if unweighted);
      - search_space/nsearch (inside search_param) are in actual (cn, ct)
        values, not a ratio -- this is the FULL (cn, ct) grid equivalent of
        cnct_ratiosearch_Rtime_obs: same forward model, alignment, weights,
        s_hat, cost and extra outputs, but no s_hat constraint and no
        second (constrained) pack.

    Cost function, exactly (i = trace, k = time sample, m = (cn,ct),
    w_i = normalized trace weight, 1 if unweighted):

        J_r(m) = 1/2 * sum_i w_i sum_k [d_i^obs(t_k) - s_hat(m) * h_tilde_i(t_k; m)]^2
        s_hat(m) = sum_i w_i sum_k [d_i^obs(t_k) * h_tilde_i(t_k; m)]
                   / sum_i w_i sum_k [h_tilde_i(t_k; m)]^2

    where h_tilde_i is the window-cc-shifted candidate model trace for
    trace i (h_cc = align_by_window_cc(h, d_i^obs, taxis, time_window)).
    Csp_sum == J_r(m) (includes the 1/2); s_hat ==
    scale_hat_time(D_all, H_all, w) applied to the shifted H_all. Note this
    differs from the sibling _cc functions (cnct_search_Rtime_Gd_cc,
    cnct_search_Rtime_ratio_Gd_cc), whose Csp_sum has no 1/2 factor -- that
    doesn't change where the minimum falls (a positive constant factor
    doesn't move the argmin), it only matters if you compare raw cost
    values against this exact J_r(m) definition.

    Parameters
    ----------
    ref_data : dict
        Needs 'reflection_traces' (.data, .taxis) and 'time_window'.
    inc_data : (inc_wave, Gd)
        inc_wave : dict, RAW direct-arrival trace (see above).
        Gd : array-like, direct geometric factor G_d(omega).
    idx_array : array-like of int
        Trace/geometry indices to fit.
    medium : (vp, vs, rho, wave)
        Medium properties plus the wave-type string passed to R_wave_v2
        (e.g. "SS"), grouped into one tuple.
    search_param : (eps, search_space, nsearch, cc[, trace_opts[, max_lag]])
        eps : float, stabilization fraction passed to R_wave_v2.
        search_space : ((cn_min, cn_max), (ct_min, ct_max)) -- actual
            compliance values, not a ratio.
        nsearch : (n_cn, n_ct), number of log-spaced grid samples.
        cc : {"signed", "abs"}, alignment lag-picking rule (see above).
        trace_opts : dict or None, optional 5th element -- same as
            cnct_ratiosearch_Rtime_obs: {"select": (key, min_value),
            "weight": key}, e.g. {"weight": "w_noise",
            "select": ("snr_noise", 7.0)}.
        max_lag : int or None, optional 6th (last) element, passed to the
            align_by_window_cc realignment.
        Legacy form (eps, search_space, nsearch, cc, max_lag) with an int
        5th element is still accepted (read as max_lag, no trace_opts).
    true_cnct : (cn, ct) or None
        If given, the true (cn, ct) node is spliced into the axes and
        annotated in the output (true_cnct/i_true/j_true/s_hat_true/
        Csp_sum_true).

    Output
    ------
    dict from cnct_search_pack, including 'lag_samples' (int) and
    'cc_max' (float), both (n_tr, n_cn, n_ct): per trace and grid node,
    the alignment lag and the signed normalized correlation rho in
    [-1, 1] at that lag (align_by_window_cc's 3rd output); 'data_energy'
    (float, 0.5*sum_i w_i sum_k d_i^2) and 'cost_norm' (n_cn, n_ct) =
    Csp_sum / data_energy (NRMS^2); and the trace bookkeeping
    'idx' (traces actually fitted), 'idx_input', 'idx_dropped',
    'trace_w', 'trace_opts' -- same fields as cnct_ratiosearch_Rtime_obs's
    out_full.
    """
    inc_wave, Gd = inc_data
    vp, vs, rho, wave = medium

    # (eps, search_space, nsearch, cc[, trace_opts[, max_lag]]); legacy
    # 5-element form with an int 5th element = max_lag (no trace_opts)
    if len(search_param) not in (4, 5, 6):
        raise ValueError(
            "search_param must be (eps, search_space, nsearch, cc"
            "[, trace_opts[, max_lag]])"
        )
    eps, search_space, nsearch, cc = search_param[:4]
    trace_opts, max_lag = None, None
    if len(search_param) >= 5:
        p5 = search_param[4]
        if isinstance(p5, (int, np.integer)) and len(search_param) == 5:
            max_lag = int(p5)                      # legacy form
        else:
            trace_opts = p5
    if len(search_param) == 6:
        max_lag = search_param[5]

    idx_input = np.asarray(idx_array, int).ravel()
    if idx_input.size == 0:
        raise ValueError("idx_array is empty")
    # trace selection (and the weight array), before any forward modeling
    idx_array, idx_dropped, w_keep = _apply_trace_opts(ref_data, idx_input, trace_opts)
    n_tr = idx_array.size
    # normalized weights (mean 1 over kept traces); ones when unweighted
    trace_w = np.ones(n_tr) if w_keep is None else w_keep / w_keep.mean()

    if "time_window" not in ref_data:
        raise KeyError(
            "ref_data needs 'time_window' (tbegin, tend) -- set in "
            "DASrefData.ipynb's windowed_das call"
        )

    D_all = np.asarray(ref_data["reflection_traces"].data, float)[idx_array]
    taxis = np.asarray(ref_data["reflection_traces"].taxis, float)
    align_fn = functools.partial(
        align_by_window_cc, taxis=taxis, time_window=ref_data["time_window"], cc=cc
    )

    axes = cnct_search_axes(search_space, nsearch, true_cnct=true_cnct)
    cn, ct = axes["cn"], axes["ct"]
    n_cn, n_ct = axes["n_cn"], axes["n_ct"]
    a = 1.0

    Csp_all = np.empty((n_tr, n_cn, n_ct), float)
    s_hat = np.empty((n_cn, n_ct), float)
    lag_samples = np.empty((n_tr, n_cn, n_ct), int)
    cc_max = np.empty((n_tr, n_cn, n_ct), float)   # signed rho at the chosen lag

    nt = D_all.shape[1]
    H_all = np.empty((n_tr, nt), float)

    for j, ct_j in enumerate(ct):
        for i, cn_i in enumerate(cn):
            for k, geom in enumerate(idx_array):
                syn = R_wave_v2(
                    ref_data, inc_wave, int(geom), a,
                    vp, vs, rho, cn_i, ct_j,
                    Gd=Gd, eps=eps, wave=wave,
                )
                h = np.asarray(syn["Er_time"], float).ravel()
                if h.size != nt:
                    raise ValueError(
                        f"Er_time length {h.size} != observed nt={nt}"
                    )
                h_cc, lag, rho_k = align_fn(h, D_all[k], max_lag=max_lag)
                H_all[k] = h_cc
                lag_samples[k, i, j] = lag
                cc_max[k, i, j] = rho_k

            s = scale_hat_time(D_all, H_all, w=trace_w)
            s_hat[i, j] = s
            # per-trace weighted L2 misfit, all traces at once
            Csp_all[:, i, j] = 0.5 * trace_w * np.sum((D_all - s * H_all) ** 2, axis=1)

    out = cnct_search_pack(
        Csp_all, s_hat, cn, ct, idx_array, a,
        axes=axes,
        mode="R",
        domain="time",
        lag_samples=lag_samples,
        cc_max=cc_max,
    )
    # normalized cost + trace bookkeeping (same as cnct_ratiosearch_Rtime_obs)
    data_energy = float(0.5 * np.sum(trace_w * np.sum(D_all ** 2, axis=1)))
    out.update({
        "data_energy": data_energy,
        "cost_norm": out["Csp_sum"] / data_energy,
        "idx_input": idx_input,
        "idx_dropped": idx_dropped,
        "trace_w": trace_w,
        "trace_opts": trace_opts,
    })
    return out


def _ratio_grid_pack(Csp_all, s_hat, cn_ratio, ct, cn_grid, idx_array, a,
                      true_pair=None, ratio_true=None, i_true=None, j_true=None,
                      **extra):
    """
    Post-process a (cn/ct ratio, ct) cost cube into a search-result dict.

    Same job as cnct_search_pack, but for a ratio-parameterized grid:
    fields use cn_ratio/ratio_best/ratio_sum_best/cn_grid naming (cn_best/
    cn_sum_best hold the ACTUAL compliance cn_ratio*ct at the best node,
    not the ratio itself) -- matches cnct_search_Rtime_ratio_Gd_cc's own
    field-naming convention, so ratio values are never mislabeled "cn".
    Used by cnct_ratiosearch_Rtime_obs for both its full-grid and
    constraint-filtered packs.

    Purpose: build per-trace and grid-summed best-fit/top-100 fields from
    an already-computed cost cube.
    Input: Csp_all (n_tr, n_ratio, n_ct) cost cube; s_hat (n_ratio, n_ct);
        cn_ratio, ct (1-D axes); cn_grid (n_ratio, n_ct) actual compliance;
        idx_array (geometry indices); a (fixed forward-model scalar);
        true_pair/ratio_true/i_true/j_true (optional true-node annotation);
        **extra merged into the output (e.g. lag_samples).
    Output: dict, see cnct_ratiosearch_Rtime_obs's docstring for keys.
    """
    n_tr = Csp_all.shape[0]
    n_ratio, n_ct = cn_ratio.size, ct.size

    ratio_best = np.empty(n_tr, float)
    ct_best = np.empty(n_tr, float)
    cn_best = np.empty(n_tr, float)
    Csp_min = np.empty(n_tr, float)
    i_best = np.empty(n_tr, int)
    j_best = np.empty(n_tr, int)
    geom_idx = np.asarray(idx_array, int).copy()
    index_arr = np.arange(n_tr, dtype=int)
    n_top = min(100, n_ratio * n_ct)
    sub100_all = np.empty((n_tr, n_top, 3), float)

    for k in range(n_tr):
        Csp = Csp_all[k]
        ii0, jj0 = np.unravel_index(np.argmin(Csp), Csp.shape)
        ratio_best[k] = float(cn_ratio[ii0])
        ct_best[k] = float(ct[jj0])
        cn_best[k] = float(cn_ratio[ii0] * ct[jj0])
        Csp_min[k] = float(Csp[ii0, jj0])
        i_best[k] = int(ii0)
        j_best[k] = int(jj0)

        flat = Csp.ravel()
        idx_flat = np.argpartition(flat, n_top - 1)[:n_top]
        idx_flat = idx_flat[np.argsort(flat[idx_flat])]
        ii, jj = np.unravel_index(idx_flat, Csp.shape)
        sub100_all[k] = np.column_stack([
            cn_ratio[ii] * ct[jj], ct[jj], flat[idx_flat],
        ])

    Csp_sum = np.sum(Csp_all, axis=0)
    Csp_mean = np.mean(Csp_all, axis=0)
    i_sum_best, j_sum_best = np.unravel_index(np.argmin(Csp_sum), Csp_sum.shape)
    ratio_sum_best = float(cn_ratio[i_sum_best])
    ct_sum_best = float(ct[j_sum_best])
    cn_sum_best = float(ratio_sum_best * ct_sum_best)
    Csp_sum_min = float(Csp_sum[i_sum_best, j_sum_best])
    s_hat_best = float(s_hat[i_sum_best, j_sum_best])

    out = {
        "mode": "R",
        "domain": "time",
        "grid": "ratio_ct",
        "a": float(a),
        "Csp": Csp_all,
        "Csp_sum": Csp_sum,
        "Csp_mean": Csp_mean,
        "s_hat": s_hat,
        "s_hat_best": s_hat_best,
        "cn_ratio": cn_ratio,
        "ct": ct,
        "cn_grid": cn_grid,
        "cn_sum_best": cn_sum_best,
        "ct_sum_best": ct_sum_best,
        "ratio_sum_best": ratio_sum_best,
        "Csp_sum_min": Csp_sum_min,
        "cn_best": cn_best,
        "ct_best": ct_best,
        "ratio_best": ratio_best,
        "Csp_min": Csp_min,
        "i_best": i_best,
        "j_best": j_best,
        "i_sum_best": int(i_sum_best),
        "j_sum_best": int(j_sum_best),
        "sub100": sub100_all,
        "index": index_arr,
        "geom_idx": geom_idx,
        "idx": np.asarray(idx_array, int),
        "true_cnct": true_pair,
        "ratio_true": ratio_true,
        "i_true": i_true,
        "j_true": j_true,
        "s_hat_true": None,
        "Csp_sum_true": None,
    }
    out.update(extra)
    if true_pair is not None:
        out["s_hat_true"] = float(s_hat[i_true, j_true])
        out["Csp_sum_true"] = float(Csp_sum[i_true, j_true])
    return out


def _ratio_cn_pack(out):
    """
    Purpose
    -------
    Relabel a ratio-grid pack built on a (cn/ct, cn) grid ("ratio_cn"). The
    searches build it with the same code as the (cn/ct, ct) grid
    (_ratio_grid_pack / constrain_ratiosearch), passing the cn axis in the
    slot named 'ct'; this function then gives every field its real meaning:
        'cn'      : 1-D cn axis (was 'ct')            [m/Pa]
        'ct_grid' : (n_ratio, n_cn) ct = cn / ratio at each node
        'cn_grid' : (n_ratio, n_cn) cn at each node
        'cn_best', 'ct_best' (per trace), 'cn_sum_best', 'ct_sum_best'
                  : actual cn and ct = cn / ratio at the best nodes
        'sub100'  : columns (cn, ct, cost)
        'grid'    : "ratio_cn"
    'cn_ratio', costs, s_hat and indices are unchanged.

    Input
    -----
    out : dict, pack from _ratio_grid_pack / constrain_ratiosearch whose
        'ct' axis holds cn values (empty constrained packs allowed).

    Output
    ------
    out : the same dict, relabelled in place (also returned).
    """
    ratio = np.asarray(out["cn_ratio"], float)
    cn_ax = np.asarray(out.pop("ct"), float)                # the axis passed in the 'ct' slot is cn
    out["cn"] = cn_ax
    out["cn_grid"] = np.broadcast_to(cn_ax[None, :], (ratio.size, cn_ax.size)).copy()
    out["ct_grid"] = cn_ax[None, :] / ratio[:, None]        # ct = cn / (cn/ct) at every node

    if out.get("ct_best") is not None:                      # per trace: 'ct_best' slot held cn
        cn_b = np.asarray(out["ct_best"], float)
        out["cn_best"] = cn_b
        out["ct_best"] = cn_b / np.asarray(out["ratio_best"], float)
    if out.get("ct_sum_best") is not None:                  # grid-summed best node
        cn_sb = float(out["ct_sum_best"])
        out["cn_sum_best"] = cn_sb
        out["ct_sum_best"] = cn_sb / float(out["ratio_sum_best"])
    if out.get("sub100") is not None:                       # (ratio*cn, cn, cost) -> (cn, ct, cost)
        sub = np.asarray(out["sub100"], float)
        cn_col = sub[..., 1].copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            ct_col = cn_col ** 2 / sub[..., 0]              # cn / ratio = cn^2 / (ratio*cn)
        sub[..., 0], sub[..., 1] = cn_col, ct_col
        out["sub100"] = sub
    out["grid"] = "ratio_cn"
    return out


def _trace_array(ref_data, key):
    """
    Purpose: fetch a per-trace array (one value per ref_data trace) by key,
        for trace weighting/selection.
    Input: ref_data (dict); key (str) -- any ref_data key holding an (n_tr,)
        array (e.g. "snr_lat" from lateral_snr, "weights"
        from DAS_inversion.calculate_trace_weight). "cc" is special-cased:
        if ref_data has no "cc" entry it is built as
        min(max_cc_positive, max_cc_negative), the same cc
        calculate_trace_weight uses.
    Output: (n_tr,) float array.
    """
    if key == "cc" and "cc" not in ref_data:
        return np.minimum(np.asarray(ref_data["max_cc_positive"], float),
                          np.asarray(ref_data["max_cc_negative"], float))
    if key not in ref_data:
        raise KeyError(f"ref_data has no per-trace array '{key}'")
    return np.asarray(ref_data[key], float).ravel()


def _apply_trace_opts(ref_data, idx_array, trace_opts):
    """
    Purpose: apply cnct_ratiosearch_Rtime_obs's optional trace selection
        (drop traces below a threshold) and return the kept indices plus the
        SNR array to weight them with.
    Input: ref_data (dict); idx_array (int array, requested traces);
        trace_opts (dict or None) with optional keys
          "select": (key, min_value) -- keep traces with
                    ref_data[key][idx] >= min_value, e.g. ("snr_lat", 3.0)
                    or ("cc", 0.85);
          "weight": key of a PRECOMPUTED per-trace weight array, e.g.
                    "w_lat" (1/sigma^2, add_lateral_weights).
        Traces whose select or weight value is NaN (or weight <= 0) are
        dropped too (e.g. traces without lateral neighbours).
    Output: (idx_keep, idx_dropped, w_keep) -- w_keep is None when no
        "weight" key is given.
    """
    idx_array = np.asarray(idx_array, int).ravel()
    trace_opts = trace_opts or {}
    unknown = set(trace_opts) - {"select", "weight"}
    if unknown:
        raise ValueError(f"unknown trace_opts keys {sorted(unknown)} (use 'select', 'weight')")

    keep = np.ones(idx_array.size, bool)
    sel = trace_opts.get("select")
    if sel is not None:
        sel_key, sel_min = sel
        v = _trace_array(ref_data, sel_key)[idx_array]
        keep &= np.isfinite(v) & (v >= float(sel_min))   # NaN compares False -> dropped

    wkey = trace_opts.get("weight")
    w = None
    if wkey is not None:
        w = _trace_array(ref_data, wkey)[idx_array]
        keep &= np.isfinite(w) & (w > 0)

    if not keep.any():
        raise ValueError(f"trace_opts {trace_opts} drops every trace in idx_array")
    return idx_array[keep], idx_array[~keep], (None if w is None else w[keep])


def _empty_ratio_grid_pack(cn_ratio, ct, cn_grid, mask):
    """
    Purpose: build an all-NaN/None ratio-grid pack when a constraint
    excludes every grid node (avoids feeding an all-inf cost cube into
    _ratio_grid_pack's argmin, which would be meaningless).
    Input: cn_ratio, ct (1-D axes); cn_grid (n_ratio, n_ct) actual
        compliance; mask (n_ratio, n_ct) bool, all False.
    Output: dict with the same keys as _ratio_grid_pack's output, maps
        filled with NaN, best-fit fields None, n_compliant=0.
    """
    nan_grid = np.full((cn_ratio.size, ct.size), np.nan)
    return {
        "mode": "R", "domain": "time", "grid": "ratio_ct", "a": 1.0,
        "Csp": None, "Csp_sum": nan_grid.copy(), "Csp_mean": nan_grid.copy(),
        "s_hat": nan_grid.copy(), "s_hat_best": None,
        "cn_ratio": cn_ratio, "ct": ct, "cn_grid": cn_grid,
        "cn_sum_best": None, "ct_sum_best": None, "ratio_sum_best": None,
        "Csp_sum_min": None, "cn_best": None, "ct_best": None, "ratio_best": None,
        "Csp_min": None, "i_best": None, "j_best": None,
        "i_sum_best": None, "j_sum_best": None, "sub100": None,
        "index": None, "geom_idx": None, "idx": None,
        "true_cnct": None, "ratio_true": None, "i_true": None, "j_true": None,
        "s_hat_true": None, "Csp_sum_true": None,
        "mask": mask, "n_compliant": 0,
    }


def lowpass_tukey_zerophase_stf(dir_data, f_end, alpha=0.5):
    """
    Purpose
    -------
    Source-wavelet (STF) version of DAS_processing.lowpass_tukey_zerophase_scat:
    zero-phase low-pass of dir_data['stf_trace'] with the same right-sided
    Tukey frequency window, so STF and data share the same filter:

        X(f)   = FFT(trace)
        W(f)   = tukey_right_freq_window(freqs, f_end, alpha)
                 (1 for |f| <= f_end (1 - alpha), half-cosine to 0 at f_end,
                  0 above; real and even in f -> zero phase)
        'fft'  = X(f) W(f)
        'trace'= real(IFFT(X(f) W(f)))

    'trace' and 'fft' are OVERWRITTEN (deep-copy dir_data beforehand to keep
    the unfiltered STF).

    Input
    -----
    dir_data : dict (ref_stf_data['stf_data'] from DASrefData.ipynb) with
        dir_data['stf_trace'] = {'trace' (nt,), 'time' (nt,) [s],
                                 'fft' (nt,), 'freqs' (nt,) [Hz]}.
        The stf_trace dict itself is also accepted.
    f_end : float
        End of the taper [Hz]; W = 0 for |f| >= f_end (e.g. 320).
    alpha : float
        Taper fraction in [0, 1]; flat passband up to f_end (1 - alpha).

    Output
    ------
    dir_data : the same dict, updated in place (also returned), with
        stf_trace['fft']   : complex (nt,), filtered spectrum
        stf_trace['trace'] : float (nt,), filtered STF
    """
    import DAS_processing as DASproc                       # local import: window helper lives there
    stf = dir_data["stf_trace"] if "stf_trace" in dir_data else dir_data
    x = np.asarray(stf["trace"], float)
    t = np.asarray(stf["time"], float)

    freqs = fftfreq(x.size, d=t[1] - t[0])                 # frequency axis of the STF [Hz]
    if "freqs" in stf and not np.allclose(stf["freqs"], freqs):
        raise ValueError("stf_trace['freqs'] does not match fftfreq(nt, dt) of the STF")

    W = DASproc.tukey_right_freq_window(freqs, f_end=f_end, alpha=alpha)  # same window as the data
    X = fft(x) * W                                         # filtered spectrum
    stf["fft"] = X
    stf["trace"] = ifft(X).real                            # zero-phase filtered STF
    return dir_data


def lateral_snr(ref_data, gauge_length, max_dist=8.0, min_nb=2):
    """
    Per-trace SNR from LATERAL COHERENCE inside the fit window.

    Purpose
    -------
    Estimate a noise level per reflected trace without using the samples
    outside the fit window: those hold scattering from other reflectors
    (coherent energy the model doesn't predict), so their RMS is not a
    noise estimate. Inside the window the reflection changes little between
    nearby channels while random noise does not repeat, so for trace k:

        stack_k = mean of the traces at gauge_length <= |dz| <= max_dist
                  (k itself excluded, so its own noise is not in the stack)
        resid_k = d_k - stack_k
        sigma_k^2 = var(resid_k) * N_k / (N_k + 1)   (removes the stack's
                                                       own noise share)
        SNR_k = (mean(d_k^2) - sigma_k^2) / sigma_k^2

    Neighbours closer than one gauge length share fibre and hence share
    noise, which would be counted as signal (SNRs inflated to 20-70 in
    tests/lateral_snr_weights_notes.md), so gauge_length is the minimum
    neighbour distance. Coherent scattering inside the window still counts
    as signal, and real along-fibre changes of the reflection (e.g. near an
    |R| null) count as noise -- see the notes for both caveats.

    Input
    -----
    ref_data : dict
        Needs 'reflection_traces' (Data2D: .data (n_tr, nt), .taxis,
        .daxis [m]) and 'time_window' (tbegin, tend).
    gauge_length : float
        DAS gauge length [m]; nearest neighbour distance used.
    max_dist : float
        Farthest neighbour distance [m] (the reflection must still look
        alike over it).
    min_nb : int
        Minimum number of neighbours for an estimate; fewer -> NaN.

    Output
    ------
    dict with (n_tr,) arrays, one entry per trace of ref_data:
        'snr'    : lateral SNR (NaN where < min_nb neighbours, e.g. at
                   depth gaps in the trace selection).
        'sigma2' : noise variance per sample (NaN where snr is NaN).
        'n_nb'   : number of neighbours used.
    Store e.g. ref_data['snr_lat'] = out['snr'] to use it as a search
    weight/selection key (see cnct_ratiosearch_Rtime_obs's trace_opts).
    """
    rr = ref_data["reflection_traces"]
    t = np.asarray(rr.taxis, float)
    depth = np.asarray(rr.daxis, float)
    tb, te = ref_data["time_window"]
    win = (t >= tb) & (t <= te)
    Dw = np.asarray(rr.data, float)[:, win]            # fit window only

    # neighbour mask from along-fibre distance (self excluded by dist >= gauge_length)
    dist = np.abs(depth[:, None] - depth[None, :])
    nb = (dist >= gauge_length) & (dist <= max_dist)
    n_nb = nb.sum(1)

    stack = (nb @ Dw) / np.maximum(n_nb, 1)[:, None]   # leave-one-out neighbour mean
    sigma2 = (Dw - stack).var(1) * n_nb / (n_nb + 1.0)
    P_s = np.mean(Dw**2, 1) - sigma2

    ok = (n_nb >= min_nb) & (sigma2 > 0)
    snr = np.full(Dw.shape[0], np.nan)
    snr[ok] = np.clip(P_s[ok], 0, None) / sigma2[ok]   # negative signal power -> SNR 0
    return {"snr": snr, "sigma2": np.where(ok, sigma2, np.nan), "n_nb": n_nb}


def add_lateral_weights(ref_data, gauge_length, max_dist=8.0, min_nb=2):
    """
    Compute the lateral-coherence SNR and noise weight for ALL traces once
    and store them in ref_data, so the inversion reads them instead of
    recomputing them on every run.

    Purpose
    -------
    Wraps lateral_snr (see its docstring for the method) and turns its noise
    variance into the per-trace weight w_k = 1/sigma_k^2: each trace's misfit
    is then measured relative to its own noise, so a trace counts by how
    clean it is rather than by how large its amplitude is.

    Input
    -----
    ref_data : dict
        The reflection dict as built in DASrefInv.ipynb (needs
        'reflection_traces' and 'time_window'; e.g. after the
        ref_data.update({"Gr": ..., "weights": ...}) cell).
    gauge_length : float
        DAS gauge length [m] (notebook's `gl`); nearest neighbour distance.
    max_dist : float
        Farthest neighbour distance [m].
    min_nb : int
        Minimum neighbours for an estimate; fewer -> NaN.

    Output
    ------
    ref_data : dict
        The SAME dict, updated in place (also returned) with (n_tr,) arrays:
          'snr_lat'    : lateral SNR
          'sigma2_lat' : noise variance per sample
          'w_lat'      : weight 1/sigma2_lat (use as trace_opts
                         {"weight": "w_lat"} in cnct_ratiosearch_Rtime_obs)
        All NaN for traces without enough neighbours (the search drops them).
    """
    lat = lateral_snr(ref_data, gauge_length, max_dist=max_dist, min_nb=min_nb)
    ref_data.update({
        "snr_lat": lat["snr"],
        "sigma2_lat": lat["sigma2"],
        "w_lat": 1.0 / lat["sigma2"],      # NaN stays NaN
    })
    return ref_data


def add_noise_weights(ref_data, taper_alpha=None, cap=False):
    """
    SNR and noise weight of each reflected trace from its stored pre-event
    noise, added to ref_data (same role as add_lateral_weights).

    Purpose
    -------
    ref_data['noise'] holds each trace's pre-event noise power sigma2
    (preevent_noise, measured on the unshifted panel in DASrefData). Here it
    is compared with the trace's power in the fit window. The fit window was
    tapered (windowed_das, Tukey), which also lowers the noise inside it, so
    the noise power is scaled by the taper's mean square:

        noise_win_k = sigma2_k * mean(w^2)
        SNR_k       = (mean(d_k^2) - noise_win_k) / noise_win_k
        w_k         = 1 / sigma2_k   (maximum-likelihood weight)

    With cap=True (same as the "noise_cap" option of
    DAS_modeling.cnct_search_RfromT) sigma2 is first limited to its median
    over all traces, because the pre-event window of some channels contains
    transient signal:
        sigma2_k -> min(sigma2_k, median(sigma2))
    and the capped noise is used for BOTH w_noise and snr_noise.

    Input
    -----
    ref_data : dict
        Needs 'reflection_traces' (windowed traces), 'time_window' and
        'noise' ((n_tr,) sigma2, same trace order as reflection_traces).
    taper_alpha : float or None
        Tukey alpha windowed_das used on the fit window; None -> use
        ref_data['taper_alpha'] if present, else 0.4 (DASrefData.ipynb).
    cap : bool
        True -> median-capped noise; False (default) -> noise as stored.

    Output
    ------
    ref_data : dict
        The SAME dict, updated in place (also returned) with (n_tr,) arrays:
          'snr_noise' : SNR against the pre-event noise
          'w_noise'   : weight 1/sigma2 (use as trace_opts
                        {"weight": "w_noise", "select": ("snr_noise", ...)}
                        in cnct_ratiosearch_Rtime_obs)
        and 'noise_cap' : bool, the cap option used.
    """
    if "noise" not in ref_data:
        raise KeyError("ref_data needs 'noise' (pre-event sigma2 per trace, from preevent_noise)")
    if taper_alpha is None:
        taper_alpha = float(ref_data.get("taper_alpha", 0.4))

    rr = ref_data["reflection_traces"]
    t = np.asarray(rr.taxis, float)
    tb, te = ref_data["time_window"]
    win = (t >= tb) & (t <= te)
    D = np.asarray(rr.data, float)[:, win]
    sigma2 = np.asarray(ref_data["noise"], float).ravel()
    if sigma2.size != D.shape[0]:
        raise ValueError(f"ref_data['noise'] has {sigma2.size} values for {D.shape[0]} traces")
    if cap:                                             # pre-event transients: noise capped at its median
        sigma2 = np.minimum(sigma2, np.median(sigma2))

    w = tukey(int(win.sum()), alpha=taper_alpha, sym=True)
    noise_win = sigma2 * np.mean(w**2)                  # noise power after the same taper
    snr = np.clip(np.mean(D**2, axis=1) - noise_win, 0, None) / noise_win

    ref_data.update({"snr_noise": snr, "w_noise": 1.0 / sigma2, "noise_cap": bool(cap)})
    return ref_data


def trace_grouping(ref_data, snr_min=3.0, s_min=0.85, min_size=10, max_lag_s=0.002,
                   user_idx_drop=None, medium=None, crit_tol=2.0):
    """
    Group the reflected traces into contiguous angle ranges of similar
    waveform (shape AND amplitude), for running the search per group.

    Purpose
    -------
    Data-driven replacement for fixed angle bins (investigation:
    tests/waveform_similarity_groups_notes.md). Steps:
      1. keep traces with ref_data['snr_noise'] >= snr_min (pre-event SNR,
         from add_noise_weights; NaN dropped), minus any user_idx_drop;
      2. multiply each trace by its path length d_r = source->reflection
         point + reflection point->fibre (removes geometric spreading);
      3. similarity of every pair over the whole fit window
         (ref_data['time_window']), at the lag (|lag| <= max_lag_s) of the
         signed cross-correlation peak:
             S_ij = 2 * sum_t x_i(t) x_j(t+lag) / (E_i + E_j)
         (1 = same shape and amplitude, -1 = opposite polarity);
      4. traces ordered by angle, each its own group; repeatedly merge the
         ADJACENT pair of groups with the highest average S until it drops
         below s_min (groups are always contiguous in angle);
      5. groups with fewer than min_size traces are merged into their more
         similar neighbour (the noisy low-angle traces otherwise stay as
         many small groups);
      6. if `medium` is given, steps 4-5 run separately on the pre- and
         post-critical traces, split at theta_c - crit_tol with
         theta_c = asin(vs/vp) (S-to-P critical angle): no group, and no
         min_size merge, ever crosses the split. Traces within crit_tol
         below theta_c go with the post-critical side, since they already
         behave like it.

    Input
    -----
    ref_data : dict
        Needs 'reflection_traces', 'time_window', 'theta_i_deg',
        'dist_source_pinc', 'dist_pinc_rr', and 'snr_noise' (from
        add_noise_weights).
    snr_min : float
        Minimum pre-event SNR (snr_noise) for a trace to be grouped.
    s_min : float
        Average similarity needed to merge two neighbouring groups.
    min_size : int
        Minimum number of traces per group.
    max_lag_s : float
        Maximum lag [s] allowed when comparing two traces.
    user_idx_drop : array-like of int or None
        Trace indices (ref_data rows) to leave out, e.g. traces judged bad
        by eye, on top of the snr_min cut.
    medium : (vp, vs, rho, wave) or None
        Medium tuple as used by the search. None = no critical-angle split.
        Only wave 'SS' is supported (S-to-P critical angle).
    crit_tol : float
        Degrees below theta_c where the split is placed: angle <=
        theta_c - crit_tol is pre-critical, above it post-critical.

    Output
    ------
    groups : tuple of int arrays
        Trace indices (ref_data rows) of each group, sorted ascending; the
        groups are ordered by index, so the first group holds the lowest
        idx (highest angles). Traces failing snr_min or listed in
        user_idx_drop are in no group.
    """
    if "snr_noise" not in ref_data:
        raise KeyError("ref_data needs 'snr_noise' -- run add_noise_weights first")
    snr = np.asarray(ref_data["snr_noise"], float)
    theta = np.asarray(ref_data["theta_i_deg"], float)
    keep = np.isfinite(snr) & (snr >= snr_min)
    if user_idx_drop is not None:
        drop = np.asarray(user_idx_drop, int).ravel()
        if np.any(drop < 0) or np.any(drop >= snr.size):
            raise ValueError(f"user_idx_drop has indices outside 0..{snr.size - 1}")
        keep[drop] = False                                   # user-rejected traces
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise ValueError(f"no trace left with snr_noise >= {snr_min} after user_idx_drop")
    idx = idx[np.argsort(theta[idx])]                        # angle order (contiguity)

    rr = ref_data["reflection_traces"]
    t = np.asarray(rr.taxis, float)
    tb, te = ref_data["time_window"]
    win = (t >= tb) & (t <= te)
    d_r = (np.asarray(ref_data["dist_source_pinc"], float)
           + np.asarray(ref_data["dist_pinc_rr"], float))[idx]
    X = np.asarray(rr.data, float)[idx][:, win] * d_r[:, None]   # spreading removed

    # similarity of all pairs, max over small lags (window edges are tapered, roll is safe)
    E = np.sum(X**2, 1)
    den = E[:, None] + E[None, :]
    max_lag = int(round(max_lag_s / (t[1] - t[0])))
    S = np.full((idx.size,) * 2, -np.inf)
    for lag in range(-max_lag, max_lag + 1):
        S = np.maximum(S, 2 * (X @ np.roll(X, lag, axis=1).T) / den)
    S = np.maximum(S, S.T)

    def link(a, b):
        return S[np.ix_(a, b)].mean()                        # average S between two groups

    def merge(pos):
        """Contiguous grouping of the angle-ordered positions `pos` (steps 4-5)."""
        groups = [np.array([k]) for k in pos]
        # contiguous average-linkage merging down to s_min
        while len(groups) > 1:
            links = [link(a, b) for a, b in zip(groups[:-1], groups[1:])]
            k = int(np.argmax(links))
            if links[k] < s_min:
                break
            groups[k:k + 2] = [np.r_[groups[k], groups[k + 1]]]
        # absorb groups smaller than min_size into their more similar neighbour
        while len(groups) > 1:
            small = [k for k, g in enumerate(groups) if g.size < min_size]
            if not small:
                break
            k = min(small, key=lambda q: groups[q].size)     # smallest first
            if k == 0:
                j = 1
            elif k == len(groups) - 1:
                j = k - 1
            else:
                j = k - 1 if link(groups[k], groups[k - 1]) >= link(groups[k], groups[k + 1]) else k + 1
            lo, hi = min(k, j), max(k, j)
            groups[lo:hi + 1] = [np.r_[groups[lo], groups[hi]]]
        return groups

    pos = np.arange(idx.size)
    if medium is None:
        groups = merge(pos)
    else:
        vp, vs, _rho, wave = medium
        if wave != "SS":
            raise ValueError(f"critical-angle split only implemented for wave='SS' (got {wave!r})")
        theta_split = np.degrees(np.arcsin(vs / vp)) - crit_tol
        pre = theta[idx] <= theta_split                      # idx is in angle order
        groups = [g for side in (pos[pre], pos[~pre]) if side.size for g in merge(side)]

    out = [np.sort(idx[g]) for g in groups]
    return tuple(sorted(out, key=lambda g: g[0]))

# per-trace bookkeeping copied from a full pack into any constrained pack
_TRACE_INFO_KEYS = ("idx_input", "idx_dropped", "trace_w", "trace_opts")


def constrain_ratiosearch(out_full, s_hat_range, save=False):
    """
    Sample an s_hat-constrained result from an UNCONSTRAINED (cn/ct ratio,
    ct) search, without re-running the forward model.

    Purpose
    -------
    The full pack of cnct_ratiosearch_Rtime_obs already stores everything
    a constraint needs: the per-trace cost cube Csp, the global s_hat grid
    and the per-trace lag_samples. This function masks the nodes whose
    s_hat falls outside `s_hat_range` and recomputes the best-fit fields on
    the compliant nodes only, so several s_hat ranges can be sampled from
    one saved run (old runs included). It is the same logic the search
    uses for its own constrained pack (the search calls this function).
    Excluded nodes are +inf for the internal argmin/top-100 (never picked
    as a best fit) and NaN in the returned maps (so a density plot skips
    them cleanly).

    Input
    -----
    out_full : dict, or str/Path
        The unconstrained pack (needs Csp, s_hat, cn_ratio, ct, cn_grid,
        idx, a, lag_samples), or the path of a run pickle holding
        (out_full, out_constraint) -- only out_full is used.
    s_hat_range : (s_hat_min, s_hat_max)
        Inclusive bounds on s_hat.
    save : bool
        Only with a path input: also write (out_full, out_constraint) to
        <run_dir>/shat_samples/<run>_shat<min>-<max>.pkl, a two-pack pickle
        that plot_ratiosearch / pick_ratiosearch read like the original
        (subfolder, so plot_waveform_picks' single main *.pkl in the run
        folder is left untouched).

    Output
    ------
    out_constraint : dict
        Same fields as the full pack, restricted to the compliant nodes,
        plus 'mask' (bool grid), 'n_compliant' (int) and 's_hat_range'.
        If no node complies, best-fit fields are None and maps all-NaN.
        With save=True, returns (out_constraint, saved_path).
    """
    pkl_path = None
    if not isinstance(out_full, dict):
        pkl_path = Path(out_full)
        with open(pkl_path, "rb") as f:
            out_full = pickle.load(f)[0]
    elif save:
        raise ValueError("save=True needs a run-pickle path as out_full")

    Csp_all = np.asarray(out_full["Csp"], float)
    s_hat = np.asarray(out_full["s_hat"], float)
    cn_ratio = np.asarray(out_full["cn_ratio"], float)
    ct = np.asarray(out_full["ct"], float)
    cn_grid = np.asarray(out_full["cn_grid"], float)

    # compliant nodes, vectorized (single boolean comparison over the grid)
    s_min, s_max = map(float, s_hat_range)
    mask = (s_hat >= s_min) & (s_hat <= s_max)
    n_compliant = int(np.count_nonzero(mask))

    if n_compliant == 0:
        out_constraint = _empty_ratio_grid_pack(cn_ratio, ct, cn_grid, mask)
    else:
        # +inf only for the internal argmin/top-100 search
        Csp_all_forfit = np.where(mask[None, :, :], Csp_all, np.inf)
        out_constraint = _ratio_grid_pack(
            Csp_all_forfit, s_hat, cn_ratio, ct, cn_grid, out_full["idx"], out_full["a"],
            true_pair=out_full.get("true_cnct"), ratio_true=out_full.get("ratio_true"),
            i_true=out_full.get("i_true"), j_true=out_full.get("j_true"),
            lag_samples=out_full["lag_samples"],
        )
        # NaN (not inf) in the RETURNED maps
        out_constraint["Csp_sum"] = np.where(mask, out_constraint["Csp_sum"], np.nan)
        out_constraint["Csp_mean"] = np.where(mask, out_constraint["Csp_mean"], np.nan)
        out_constraint["s_hat"] = np.where(mask, s_hat, np.nan)
        out_constraint["Csp"] = np.where(mask[None, :, :], Csp_all, np.nan)
        top_cost = out_constraint["sub100"][..., 2]
        out_constraint["sub100"][np.isinf(top_cost)] = np.nan
        out_constraint["mask"] = mask
        out_constraint["n_compliant"] = n_compliant

    out_constraint["s_hat_range"] = (s_min, s_max)
    # trace bookkeeping (fitted/dropped traces, weights), when the run has it
    out_constraint.update({k: out_full[k] for k in _TRACE_INFO_KEYS if k in out_full})
    # per-trace cc value at the chosen lag (runs made after it was added)
    if "cc_max" in out_full:
        out_constraint["cc_max"] = out_full["cc_max"]
    # normalized cost (runs made after it was added); Csp_sum is already
    # NaN outside the compliant region, so cost_norm is too
    if "data_energy" in out_full:
        E = float(out_full["data_energy"])
        out_constraint["data_energy"] = E
        out_constraint["cost_norm"] = np.asarray(out_constraint["Csp_sum"], float) / E

    if not save:
        return out_constraint
    out_dir = pkl_path.parent / "shat_samples"
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / f"{pkl_path.stem}_shat{s_min:g}-{s_max:g}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump((out_full, out_constraint), f)
    print(f"saved {save_path}", flush=True)
    return out_constraint, save_path


def cnct_ratiosearch_Rtime_obs(
    ref_data,
    inc_data,
    idx_array,
    medium,
    search_param,
    true_cnct=None,
):
    """
    (cn/ct ratio, ct)-parameterized sibling of cnct_search_Rtime_Gd_cc_obs:
    fits REAL observed reflection data (ref_data['reflection_traces'].data)
    over a grid of (cn/ct ratio, ct) instead of (cn, ct). At every grid
    node (ratio_i, ct_j), cn = ratio_i * ct_j is passed to R_wave_v2,
    exactly as in the synthetic-data sibling cnct_search_Rtime_ratio_Gd_cc,
    but using the same real-data machinery as cnct_search_Rtime_Gd_cc_obs
    (align_by_window_cc realignment, a fixed at 1.0, s_hat as the only
    free scalar, no r_batch) -- see that function's docstring for why.

    Also returns a second, "constrained" pack that excludes grid nodes
    whose fitted s_hat falls outside the requested range -- built by
    constrain_ratiosearch, which masks the ALREADY-COMPUTED cost cube (no
    new R_wave_v2 calls; call it later on out_full, or on a saved run
    pickle, to sample other s_hat ranges from the same run):
    non-compliant nodes are set to NaN in the constrained pack's
    Csp_sum/Csp_mean/s_hat/Csp maps (so they render as gaps in a density
    plot, e.g. pcolormesh, instead of breaking a log-scaled colorbar the
    way +inf would), and the constrained pack's own best-fit fields
    (ratio_sum_best, ct_sum_best, cn_sum_best, Csp_sum_min, s_hat_best,
    i_sum_best, j_sum_best, per-trace best/sub100) are recomputed from the
    argmin restricted to the compliant nodes only.

    Purpose
    -------
    Grid-search (cn/ct ratio, ct) against real reflection data and, from
    that single search, also report the sub-grid that satisfies a
    physical s_hat constraint -- without re-running the forward model.

    Input
    -----
    ref_data : dict
        Needs 'reflection_traces' (.data, .taxis) and 'time_window'
        (tbegin, tend), as set by DASrefData.ipynb's windowed_das call.
    inc_data : (inc_wave, Gd)
        inc_wave : dict, RAW direct-arrival trace (e.g.
            dir_data['stf_trace']), NOT an Estimate_stf output -- R_wave_v2
            does its own Gd-deconvolution internally, so an already-
            deconvolved STF would be double-deconvolved.
        Gd : array-like, direct geometric factor G_d(omega) (e.g.
            geometric_factor_dir(...)['Gd']).
    idx_array : array-like of int
        Geometry/trace indices to fit.
    medium : (vp, vs, rho, wave)
        Medium properties plus the wave-type string passed to R_wave_v2
        (e.g. "SS"), grouped into one tuple.
    search_param : (eps, search_space, nsearch, cc, constraint[, trace_opts[, max_lag]])
        eps : float, stabilization fraction passed to R_wave_v2.
        search_space : ((ratio_min, ratio_max), (ct_min, ct_max)) -- first
            tuple is the cn/ct RATIO range (not cn itself); second is ct
            [m/Pa].
        nsearch : (n_ratio, n_ct), number of log-spaced grid samples
            (before true_cnct, if given, is spliced in).
        cc : {"signed", "abs"}, alignment lag-picking rule passed to
            align_by_window_cc -- see cnct_search_Rtime_Gd_cc_obs's
            docstring for the trade-off.
        constraint : (s_hat_min, s_hat_max), inclusive bounds on s_hat
            defining the "constrained" pack.
        trace_opts : dict or None, optional 6th element (defaults to None = every
            trace in idx_array used, unweighted -- identical to runs made
            before this option existed). Keys (both optional):
              "select": (key, min_value) -- drop traces with
                  ref_data[key][idx] < min_value, e.g. ("snr_lat", 3.0) or
                  ("cc", 0.85) ("cc" = min(max_cc_positive,
                  max_cc_negative) unless ref_data has its own "cc").
              "weight": key of a PRECOMPUTED per-trace weight array in
                  ref_data, e.g. "w_lat" = 1/sigma_k^2 from
                  add_lateral_weights (computed once for
                  all traces, not per run). Normalized here to mean 1 over
                  the kept traces; s_hat and the cost then use
                  s_hat = sum_k w_k d_k.h_k / sum_k w_k h_k.h_k and
                  Csp_k = 0.5 * w_k * sum_t (d_k - s_hat h_k)^2.
            Traces with a NaN select/weight value are dropped as well.
            Being part of search_param, it is saved in the run's
            *_data.pkl by cnct_search.
        max_lag : int or None, optional LAST (7th) element, defaults to
            None when not passed; passed to the align_by_window_cc
            realignment. To set it, pass trace_opts too (None if unused).
    true_cnct : (cn, ct) or None
        If given, the true (ratio, ct) node is spliced into the axes and
        annotated in both output packs (true_cnct/i_true/j_true/
        s_hat_true/Csp_sum_true).

    Output
    ------
    (out_full, out_constraint) : (dict, dict)
        out_full : the unconstrained (ratio, ct) grid-search pack.
            Includes 'lag_samples' (int) and 'cc_max' (float), both
            (n_tr, n_ratio, n_ct): per trace and grid node, the alignment
            lag and the signed normalized correlation rho in [-1, 1] at
            that lag (align_by_window_cc's 3rd output). Both are also
            carried into out_constraint (unmasked).
            Also 'data_energy' (float, 0.5*sum_k w_k*sum_t d_k^2) and
            'cost_norm' (n_ratio, n_ct) = Csp_sum / data_energy, the
            normalized misfit (NRMS^2: 0 = perfect fit, 1 = the model
            explains none of the data); in out_constraint it is NaN
            outside the compliant region, like Csp_sum.
        out_constraint : same fields, restricted to nodes complying with
            the constraint; Csp_sum/Csp_mean/s_hat/Csp are NaN outside the
            compliant region; additionally has 'mask' (bool grid),
            'n_compliant' (int), and 's_hat_range' (the constraint tuple
            echoed back). If no grid node complies, all best-fit fields
            are None and the maps are all-NaN.
        Both dicts also carry the trace bookkeeping: 'idx'/'geom_idx'
        are the traces actually fitted (after trace_opts' selection),
        'idx_input' the requested idx_array, 'idx_dropped' the traces
        removed, 'trace_w' the per-trace weights (all 1 if unweighted),
        and 'trace_opts' echoed back.
        Both dicts use the ratio-grid field names cn_ratio, ratio_best,
        ratio_sum_best, ct, ct_best, ct_sum_best, cn_grid (= ratio*ct,
        the actual compliance at each node), cn_best/cn_sum_best (actual
        compliance at the best-fit node) -- see _ratio_grid_pack.
    """
    inc_wave, Gd = inc_data
    vp, vs, rho, wave = medium

    # optional trailing elements: trace_opts (6th), max_lag (7th, last);
    # each defaults to None when not passed
    if len(search_param) not in (5, 6, 7):
        raise ValueError(
            "search_param must be (eps, search_space, nsearch, cc, "
            "constraint[, trace_opts[, max_lag]])"
        )
    eps, search_space, nsearch, cc, constraint = search_param[:5]
    trace_opts = search_param[5] if len(search_param) >= 6 else None
    max_lag = search_param[6] if len(search_param) == 7 else None

    idx_input = np.asarray(idx_array, int).ravel()
    if idx_input.size == 0:
        raise ValueError("idx_array is empty")
    # trace selection (and the SNR to weight by), before any forward modeling
    idx_array, idx_dropped, w_keep = _apply_trace_opts(ref_data, idx_input, trace_opts)
    n_tr = idx_array.size

    if "time_window" not in ref_data:
        raise KeyError(
            "ref_data needs 'time_window' (tbegin, tend) -- set in "
            "DASrefData.ipynb's windowed_das call"
        )

    D_all = np.asarray(ref_data["reflection_traces"].data, float)[idx_array]
    taxis = np.asarray(ref_data["reflection_traces"].taxis, float)
    align_fn = functools.partial(
        align_by_window_cc, taxis=taxis, time_window=ref_data["time_window"], cc=cc
    )

    # per-trace weights, precomputed in ref_data (e.g. w_lat = 1/sigma^2);
    # only normalized here (mean 1 over the kept traces, so the cost
    # magnitude stays comparable to unweighted); all ones when unweighted
    trace_w = np.ones(n_tr) if w_keep is None else w_keep / w_keep.mean()

    try:
        n_ratio, n_ct = int(nsearch[0]), int(nsearch[1])
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("nsearch must be (n_ratio, n_ct)") from exc
    if n_ratio < 2 or n_ct < 2:
        raise ValueError("n_ratio and n_ct must be >= 2")

    try:
        (r_min, r_max), (ct_min, ct_max) = search_space
        r_min, r_max = float(r_min), float(r_max)
        ct_min, ct_max = float(ct_min), float(ct_max)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "search_space must be [(ratio_min, ratio_max), (ct_min, ct_max)]"
        ) from exc
    if not (r_min > 0 and r_max > r_min):
        raise ValueError("need 0 < ratio_min < ratio_max")
    if not (ct_min > 0 and ct_max > ct_min):
        raise ValueError("need 0 < ct_min < ct_max")

    cn_ratio = np.logspace(np.log10(r_min), np.log10(r_max), n_ratio)
    ct = np.logspace(np.log10(ct_min), np.log10(ct_max), n_ct)

    i_true = j_true = None
    true_pair = None
    ratio_true = None
    if true_cnct is not None:
        cn_t, ct_t = map(float, true_cnct)
        if cn_t <= 0 or ct_t <= 0:
            raise ValueError("true_cnct values must be > 0")
        ratio_true = cn_t / ct_t
        cn_ratio = np.unique(np.sort(np.append(cn_ratio, ratio_true)))
        ct = np.unique(np.sort(np.append(ct, ct_t)))
        true_pair = (cn_t, ct_t)
        i_true = int(np.argmin(np.abs(np.log10(cn_ratio) - np.log10(ratio_true))))
        j_true = int(np.argmin(np.abs(np.log10(ct) - np.log10(ct_t))))

    n_ratio, n_ct = int(cn_ratio.size), int(ct.size)
    a = 1.0

    Csp_all = np.empty((n_tr, n_ratio, n_ct), float)
    s_hat = np.empty((n_ratio, n_ct), float)
    lag_samples = np.empty((n_tr, n_ratio, n_ct), int)
    cc_max = np.empty((n_tr, n_ratio, n_ct), float)   # signed rho at the chosen lag

    nt = D_all.shape[1]
    H_all = np.empty((n_tr, nt), float)

    # forward model -- the only part that needs recomputation; the
    # constraint below reuses these Csp_all/s_hat without calling
    # R_wave_v2 again
    for j, ct_j in enumerate(ct):
        for i, r in enumerate(cn_ratio):
            cn_i = r * ct_j
            for k, geom in enumerate(idx_array):
                syn = R_wave_v2(
                    ref_data, inc_wave, int(geom), a,
                    vp, vs, rho, cn_i, ct_j,
                    Gd=Gd, eps=eps, wave=wave,
                )
                h = np.asarray(syn["Er_time"], float).ravel()
                if h.size != nt:
                    raise ValueError(
                        f"Er_time length {h.size} != observed nt={nt}"
                    )
                h_cc, lag, rho_k = align_fn(h, D_all[k], max_lag=max_lag)
                H_all[k] = h_cc
                lag_samples[k, i, j] = lag
                cc_max[k, i, j] = rho_k

            s = scale_hat_time(D_all, H_all, w=trace_w)
            s_hat[i, j] = s
            # per-trace weighted L2 misfit, all traces at once
            # (trace_w = 1 reproduces 0.5 * C_time_unnorm per trace)
            Csp_all[:, i, j] = 0.5 * trace_w * np.sum((D_all - s * H_all) ** 2, axis=1)

    cn_grid = cn_ratio[:, None] * ct[None, :]

    # data energy with the same 0.5 * w_k factor as Csp -> cost_norm =
    # Csp_sum / data_energy = sum_k w_k|d_k - s h_k|^2 / sum_k w_k|d_k|^2
    # (NRMS^2: 0 = perfect fit, 1 = model explains none of the data);
    # comparable across groups/runs, same minimum as Csp_sum
    data_energy = float(0.5 * np.sum(trace_w * np.sum(D_all ** 2, axis=1)))

    out_full = _ratio_grid_pack(
        Csp_all, s_hat, cn_ratio, ct, cn_grid, idx_array, a,
        true_pair=true_pair, ratio_true=ratio_true, i_true=i_true, j_true=j_true,
        lag_samples=lag_samples,
    )

    # trace bookkeeping ('idx' = traces actually fitted)
    out_full.update({
        "idx_input": idx_input,
        "idx_dropped": idx_dropped,
        "trace_w": trace_w,
        "trace_opts": trace_opts,
        "cc_max": cc_max,
        "data_energy": data_energy,
        "cost_norm": out_full["Csp_sum"] / data_energy,
    })

    # constrained pack sampled from the full one (same masking any later
    # constrain_ratiosearch call does -- no new R_wave_v2 calls)
    out_constraint = constrain_ratiosearch(out_full, constraint)

    return out_full, out_constraint


def _empty_cnct_pack(cn, ct, mask):
    """
    Purpose: all-NaN/None (cn, ct) pack when an s_hat constraint excludes
        every grid node (cn_ct counterpart of _empty_ratio_grid_pack).
    Input: cn, ct (1-D axes); mask (n_cn, n_ct) bool, all False.
    Output: dict with cnct_search_pack's keys, maps NaN, best-fit fields
        None, n_compliant=0.
    """
    nan_grid = np.full((cn.size, ct.size), np.nan)
    return {
        "mode": "R", "domain": "time", "grid": "cn_ct", "a": 1.0,
        "Csp": None, "Csp_sum": nan_grid.copy(), "Csp_mean": nan_grid.copy(),
        "s_hat": nan_grid.copy(), "s_hat_best": None, "cn": cn, "ct": ct,
        "cn_sum_best": None, "ct_sum_best": None, "Csp_sum_min": None,
        "cn_best": None, "ct_best": None, "Csp_min": None,
        "i_best": None, "j_best": None, "i_sum_best": None, "j_sum_best": None,
        "sub100": None, "index": None, "geom_idx": None, "idx": None,
        "true_cnct": None, "i_true": None, "j_true": None,
        "s_hat_true": None, "Csp_sum_true": None,
        "mask": mask, "n_compliant": 0,
    }


def constrain_searchR(out_full, s_hat_range):
    """
    Purpose
    -------
    s_hat-constrained pack from an UNCONSTRAINED cnct_searchR_obs result,
    for either grid, without re-running the forward model:
      - ratio grid ('cn_ratio' in out_full) -> constrain_ratiosearch
        (unchanged, same result as before);
      - (cn, ct) grid -> same logic here on the cnct_search_pack fields:
        nodes with s_hat outside s_hat_range are +inf for the internal
        argmin/top-100 and NaN in the returned maps.

    Input
    -----
    out_full : dict, full pack of cnct_searchR_obs (either grid). For the
        ratio grid a run-pickle path also works (see constrain_ratiosearch).
    s_hat_range : (s_hat_min, s_hat_max), inclusive bounds on s_hat.

    Output
    ------
    out_constraint : dict, same fields as out_full restricted to the
        compliant nodes, plus 'mask', 'n_compliant', 's_hat_range' (and
        cc_max / data_energy / cost_norm / trace bookkeeping copied over).
    """
    # ratio grid (or a pickle path): the existing function does it all
    if not isinstance(out_full, dict) or "cn_ratio" in out_full:
        return constrain_ratiosearch(out_full, s_hat_range)

    Csp_all = np.asarray(out_full["Csp"], float)
    s_hat = np.asarray(out_full["s_hat"], float)
    cn = np.asarray(out_full["cn"], float)
    ct = np.asarray(out_full["ct"], float)

    # compliant nodes (one vectorized comparison over the grid)
    s_min, s_max = map(float, s_hat_range)
    mask = (s_hat >= s_min) & (s_hat <= s_max)
    n_compliant = int(np.count_nonzero(mask))

    if n_compliant == 0:
        out_constraint = _empty_cnct_pack(cn, ct, mask)
    else:
        # true-node annotation carried through cnct_search_pack's axes arg
        axes = {k: out_full.get(k) for k in ("true_cnct", "i_true", "j_true")}
        # +inf only for the internal argmin/top-100 search
        Csp_all_forfit = np.where(mask[None, :, :], Csp_all, np.inf)
        out_constraint = cnct_search_pack(
            Csp_all_forfit, s_hat, cn, ct, out_full["idx"], out_full["a"],
            axes=axes, mode="R", domain="time",
            lag_samples=out_full["lag_samples"], grid="cn_ct",
        )
        # NaN (not inf) in the RETURNED maps, so plots show gaps
        out_constraint["Csp_sum"] = np.where(mask, out_constraint["Csp_sum"], np.nan)
        out_constraint["Csp_mean"] = np.where(mask, out_constraint["Csp_mean"], np.nan)
        out_constraint["s_hat"] = np.where(mask, s_hat, np.nan)
        out_constraint["Csp"] = np.where(mask[None, :, :], Csp_all, np.nan)
        top_cost = out_constraint["sub100"][..., 2]
        out_constraint["sub100"][np.isinf(top_cost)] = np.nan
        out_constraint["mask"] = mask
        out_constraint["n_compliant"] = n_compliant

    out_constraint["s_hat_range"] = (s_min, s_max)
    # trace bookkeeping, cc and normalized cost (NaN outside the mask)
    out_constraint.update({k: out_full[k] for k in _TRACE_INFO_KEYS if k in out_full})
    for k in ("cc_max", "search_mode"):
        if k in out_full:
            out_constraint[k] = out_full[k]
    if "data_energy" in out_full:
        E = float(out_full["data_energy"])
        out_constraint["data_energy"] = E
        out_constraint["cost_norm"] = np.asarray(out_constraint["Csp_sum"], float) / E
    return out_constraint


def cnct_searchR_obs(
    ref_data,
    inc_data,
    idx_array,
    medium,
    search_param,
    true_cnct=None,
):
    """
    Purpose
    -------
    Single time-domain compliance grid search on REAL observed reflection
    data that combines cnct_ratiosearch_Rtime_obs and
    cnct_search_Rtime_Gd_cc_obs (both kept unchanged). Two switches at the
    start of search_param choose:
      search_mode : "full" -> unconstrained grid only;
                    "constrained" -> also the s_hat-constrained pack.
      grid        : "ratio_ct" -> (cn/ct ratio, ct) grid
                    (cn = ratio*ct at each node; formerly "cn_ctratio");
                    "ratio_cn" -> (cn/ct ratio, cn) grid
                    (ct = cn/ratio at each node);
                    "non_ratio" -> (cn, ct) grid.
    Everything else is identical for all four combinations: R_wave_v2
    forward model with a = 1, align_by_window_cc realignment within
    ref_data['time_window'] (optionally |lag| <= max_lag, edge rule),
    trace selection/weights (trace_opts, weights normalized to mean 1),
    s_hat = sum_i w_i d_i.h_i / sum_i w_i h_i.h_i and
    Csp_i = 0.5 * w_i * sum_t (d_i - s_hat h_i)^2.
    The output of each combination equals the matching old function.

    Input
    -----
    ref_data : dict, needs 'reflection_traces' (.data, .taxis) and
        'time_window' (plus the trace_opts keys, e.g. from
        add_noise_weights).
    inc_data : (inc_wave, Gd) -- RAW direct-arrival trace dict and the
        direct geometric factor G_d(omega).
    idx_array : array-like of int, trace/geometry indices to fit.
    medium : (vp, vs, rho, wave).
    search_param : (search_mode, grid, eps, search_space, nsearch, cc,
                    constraint[, trace_opts[, max_lag]])
        search_mode : {"full", "constrained"}.
        grid : {"ratio_ct", "ratio_cn", "non_ratio"}.
        eps : float, stabilization fraction for R_wave_v2.
        search_space : ((p_min, p_max), (q_min, q_max)): p = cn/ct ratio
            for "ratio_ct" / "ratio_cn", p = cn [m/Pa] for "non_ratio";
            q = ct [m/Pa] for "ratio_ct" / "non_ratio", q = cn [m/Pa] for
            "ratio_cn".
        nsearch : (n_p, n_ct), log-spaced samples per axis.
        cc : {"signed", "abs"}, align_by_window_cc lag-picking rule.
        constraint : (s_hat_min, s_hat_max) for "constrained"; ignored
            (use None) for "full".
        trace_opts : dict or None, e.g. {"weight": "w_noise",
            "select": ("snr_noise", 7.0)} (see cnct_ratiosearch_Rtime_obs).
        max_lag : int or None, 9th element.
        Q_opts : None (default: no attenuation, 1/Q = 0; search exactly as
            before) or dict {"Q": list of Q values (np.inf allowed = no Q),
            "rd": direct (STF) path [m] = dir_data['mean_dist_source']},
            10th element. With Q_opts the forward model is R_batch_InvQ
            (a = 1, beta = 1/Q, attenuation exp(-|w|(rr-rd)/(2 Q vs))) and
            every (p, ct) node is fitted at every Q (align, s_hat, cost as
            above); the returned maps are taken at the Q with the lowest
            summed cost at each node.
    true_cnct : (cn, ct) or None, true node spliced into the axes.

    Output
    ------
    "full"        -> out_full (dict)
    "constrained" -> (out_full, out_constraint)
        out_full : "ratio_ct" -> _ratio_grid_pack fields (cn_ratio, ct,
            cn_grid, ratio_sum_best, ...); "ratio_cn" -> the same fields
            relabelled by _ratio_cn_pack (axes cn_ratio, cn; 2-D ct_grid;
            cn/ct_sum_best = actual values); (cn, ct) grid ->
            cnct_search_pack fields (cn, ct, ...). Both add 'lag_samples',
            'cc_max' (n_tr, n_p, n_ct), 'data_energy', 'cost_norm'
            (= Csp_sum / data_energy), trace bookkeeping ('idx',
            'idx_input', 'idx_dropped', 'trace_w', 'trace_opts'), 'grid'
            ("ratio_ct" / "cn_ct") and 'search_mode'.
        out_constraint : constrain_searchR(out_full, constraint).
        With Q_opts, both packs also carry 'Q_list', 'rd', 'Q_hat' and
        'beta_hat' (n_p, n_ct) (Q of the chosen fit at each node), and
        out_full the cubes 'Csp_sum_Q', 's_hat_Q' (n_Q, n_p, n_ct). The
        constrained pack picks, at each node, the lowest-cost Q among those
        with s_hat inside the constraint (then constrain_searchR).
    """
    inc_wave, Gd = inc_data
    vp, vs, rho, wave = medium

    # ---- search_param: (search_mode, grid, eps, space, nsearch, cc, constraint[, trace_opts[, max_lag]])
    if len(search_param) not in (7, 8, 9, 10):
        raise ValueError(
            "search_param must be (search_mode, grid, eps, search_space, nsearch, "
            "cc, constraint[, trace_opts[, max_lag[, Q_opts]]])"
        )
    search_mode, grid, eps, search_space, nsearch, cc, constraint = search_param[:7]
    trace_opts = search_param[7] if len(search_param) >= 8 else None
    max_lag = search_param[8] if len(search_param) >= 9 else None
    Q_opts = search_param[9] if len(search_param) == 10 else None
    if Q_opts is not None:                              # Q grid: list of Q + direct path rd
        if not isinstance(Q_opts, dict) or not {"Q", "rd"} <= set(Q_opts):
            raise ValueError('Q_opts must be None or {"Q": [Q values], "rd": direct path [m]}')
        Q_list = np.asarray(Q_opts["Q"], float).ravel()
        if Q_list.size == 0 or np.any(~(Q_list > 0)):
            raise ValueError("Q_opts['Q'] must be a non-empty list of Q > 0 (np.inf = no Q)")
        rd = float(Q_opts["rd"])
        beta_list = np.where(np.isinf(Q_list), 0.0, 1.0 / Q_list)
    if search_mode not in ("full", "constrained"):
        raise ValueError("search_mode must be 'full' or 'constrained'")
    if grid == "cn_ctratio":
        raise ValueError("grid 'cn_ctratio' was renamed 'ratio_ct'")
    if grid not in ("ratio_ct", "ratio_cn", "non_ratio"):
        raise ValueError("grid must be 'ratio_ct', 'ratio_cn' or 'non_ratio'")
    if search_mode == "constrained" and constraint is None:
        raise ValueError("search_mode='constrained' needs constraint=(s_hat_min, s_hat_max)")
    is_ratio = grid in ("ratio_ct", "ratio_cn")          # first axis is cn/ct
    is_rcn = grid == "ratio_cn"                         # second axis is cn (ct = cn / ratio)

    def node_cnct(p_i, q_j):
        """actual (cn, ct) of grid node (p_i, q_j) for the chosen grid"""
        if is_rcn:
            return q_j, q_j / p_i
        if is_ratio:
            return p_i * q_j, q_j
        return p_i, q_j

    # ---- trace selection / weights, before any forward modeling
    idx_input = np.asarray(idx_array, int).ravel()
    if idx_input.size == 0:
        raise ValueError("idx_array is empty")
    idx_array, idx_dropped, w_keep = _apply_trace_opts(ref_data, idx_input, trace_opts)
    n_tr = idx_array.size
    trace_w = np.ones(n_tr) if w_keep is None else w_keep / w_keep.mean()   # mean 1

    if "time_window" not in ref_data:
        raise KeyError(
            "ref_data needs 'time_window' (tbegin, tend) -- set in "
            "DASrefData.ipynb's windowed_das call"
        )
    D_all = np.asarray(ref_data["reflection_traces"].data, float)[idx_array]
    taxis = np.asarray(ref_data["reflection_traces"].taxis, float)
    align_fn = functools.partial(
        align_by_window_cc, taxis=taxis, time_window=ref_data["time_window"], cc=cc
    )

    # ---- axes: first axis p is the ratio or cn; same log-spacing and
    # true-node splicing for both (true node given as (ratio_true, ct) on
    # the ratio grid)
    true_p = None
    if true_cnct is not None:
        cn_t, ct_t = map(float, true_cnct)
        true_p = ((cn_t / ct_t, cn_t) if is_rcn else
                  (cn_t / ct_t, ct_t) if is_ratio else (cn_t, ct_t))
    axes = cnct_search_axes(search_space, nsearch, true_cnct=true_p)
    p_ax, ct = axes["cn"], axes["ct"]
    n_p, n_ct = axes["n_cn"], axes["n_ct"]
    a = 1.0

    Csp_all = np.empty((n_tr, n_p, n_ct), float)
    s_hat = np.empty((n_p, n_ct), float)
    lag_samples = np.empty((n_tr, n_p, n_ct), int)
    cc_max = np.empty((n_tr, n_p, n_ct), float)    # signed rho at the chosen lag
    nt = D_all.shape[1]
    H_all = np.empty((n_tr, nt), float)

    # ---- Q grid: forward model R_batch_InvQ at every (node, Q), then the
    # per-node best Q gives the returned maps (Csp_all, s_hat, lags, cc)
    if Q_opts is not None:
        nQ = Q_list.size
        CspQ = np.empty((nQ, n_tr, n_p, n_ct), float)
        sQ = np.empty((nQ, n_p, n_ct), float)
        lagQ = np.empty((nQ, n_tr, n_p, n_ct), int)
        ccQ = np.empty((nQ, n_tr, n_p, n_ct), float)
        for j, q_j in enumerate(ct):
            for i, p_i in enumerate(p_ax):
                cn_i, ct_j = node_cnct(p_i, q_j)        # actual (cn, ct) at this node
                for m, b in enumerate(beta_list):
                    Hq = R_batch_InvQ(ref_data, inc_wave, idx_array, a, vp, vs, rho, cn_i, ct_j,
                                      Gd=Gd, eps=eps, beta=b, rd=rd, wave=wave)["Er_time"]
                    for k in range(n_tr):
                        H_all[k], lagQ[m, k, i, j], ccQ[m, k, i, j] = align_fn(
                            Hq[k], D_all[k], max_lag=max_lag)
                    sq = scale_hat_time(D_all, H_all, w=trace_w)
                    sQ[m, i, j] = sq
                    CspQ[m, :, i, j] = 0.5 * trace_w * np.sum((D_all - sq * H_all) ** 2, axis=1)
        CsumQ = CspQ.sum(axis=1)                        # (nQ, n_p, n_ct)
        mbest = CsumQ.argmin(axis=0)                    # best Q index per node
        take = lambda A, mi: np.take_along_axis(A, mi[None, None] if A.ndim == 4 else mi[None],
                                                axis=0)[0]
        Csp_all, s_hat = take(CspQ, mbest), take(sQ, mbest)
        lag_samples, cc_max = take(lagQ, mbest), take(ccQ, mbest)

    # ---- forward model + alignment + s_hat + cost at every node (no Q)
    for j, q_j in (enumerate(ct) if Q_opts is None else ()):
        for i, p_i in enumerate(p_ax):
            cn_i, ct_j = node_cnct(p_i, q_j)            # actual (cn, ct) at this node
            for k, geom in enumerate(idx_array):
                syn = R_wave_v2(
                    ref_data, inc_wave, int(geom), a,
                    vp, vs, rho, cn_i, ct_j,
                    Gd=Gd, eps=eps, wave=wave,
                )
                h = np.asarray(syn["Er_time"], float).ravel()
                if h.size != nt:
                    raise ValueError(f"Er_time length {h.size} != observed nt={nt}")
                h_cc, lag, rho_k = align_fn(h, D_all[k], max_lag=max_lag)
                H_all[k] = h_cc
                lag_samples[k, i, j] = lag
                cc_max[k, i, j] = rho_k

            s = scale_hat_time(D_all, H_all, w=trace_w)
            s_hat[i, j] = s
            # per-trace weighted L2 misfit, all traces at once
            Csp_all[:, i, j] = 0.5 * trace_w * np.sum((D_all - s * H_all) ** 2, axis=1)

    # ---- pack (field names follow the grid type, as in the old functions)
    if is_ratio:
        out_full = _ratio_grid_pack(
            Csp_all, s_hat, p_ax, ct,
            (np.broadcast_to(ct[None, :], (p_ax.size, ct.size)) if is_rcn
             else p_ax[:, None] * ct[None, :]), idx_array, a,
            true_pair=None if true_cnct is None else (cn_t, ct_t),
            ratio_true=None if true_p is None else true_p[0],
            i_true=axes["i_true"], j_true=axes["j_true"],
            lag_samples=lag_samples,
        )
    else:
        out_full = cnct_search_pack(
            Csp_all, s_hat, p_ax, ct, idx_array, a,
            axes=axes, mode="R", domain="time",
            lag_samples=lag_samples, grid="cn_ct",
        )

    # normalized cost (NRMS^2, same argmin as Csp_sum) + trace bookkeeping
    data_energy = float(0.5 * np.sum(trace_w * np.sum(D_all ** 2, axis=1)))
    out_full.update({
        "idx_input": idx_input,
        "idx_dropped": idx_dropped,
        "trace_w": trace_w,
        "trace_opts": trace_opts,
        "cc_max": cc_max,
        "data_energy": data_energy,
        "cost_norm": out_full["Csp_sum"] / data_energy,
        "search_mode": search_mode,
    })

    if Q_opts is not None:                              # Q fields of the full pack
        out_full.update({"Q_list": Q_list, "rd": rd, "Q_opts": Q_opts,
                         "Q_hat": Q_list[mbest], "beta_hat": beta_list[mbest],
                         "Csp_sum_Q": CsumQ, "s_hat_Q": sQ})

    # output matches the call: constrained -> both packs, full -> full only
    if search_mode == "constrained":
        if Q_opts is None:
            out_c = constrain_searchR(out_full, constraint)
            return (_ratio_cn_pack(out_full), _ratio_cn_pack(out_c)) if is_rcn else (out_full, out_c)
        # per node: lowest-cost Q among those with s_hat inside the constraint
        s_lo, s_hi = map(float, constraint)
        mc = np.where((sQ >= s_lo) & (sQ <= s_hi), CsumQ, np.inf).argmin(axis=0)
        tmp = dict(out_full)
        tmp.update({"Csp": take(CspQ, mc), "s_hat": take(sQ, mc),
                    "lag_samples": take(lagQ, mc), "cc_max": take(ccQ, mc)})
        out_c = constrain_searchR(tmp, constraint)
        mask = out_c["mask"]
        out_c.update({"Q_list": Q_list, "rd": rd, "Q_opts": Q_opts,
                      "Q_hat": np.where(mask, Q_list[mc], np.nan),
                      "beta_hat": np.where(mask, beta_list[mc], np.nan)})
        return (_ratio_cn_pack(out_full), _ratio_cn_pack(out_c)) if is_rcn else (out_full, out_c)
    return _ratio_cn_pack(out_full) if is_rcn else out_full


def R_synth_Gd(
    s_hat_list,
    ref_data,
    inc_wave,
    idx,
    vp,
    vs,
    rho,
    Gd,
    eps=0.005,
    a=1.0,
    wave="SS",
):
    """
    Same as R_synth, but via R_wave_v2 (Gd-deconvolution implicit, inc_wave
    should be the RAW direct-arrival trace, not an Estimate_stf output) --
    see cnct_search_Rtime_Gd / module docstring above. Use this, not
    R_synth, to reconstruct a fitted waveform for comparison against
    R_wave_batch_v2-generated "true" synthetic data or _Gd search results
    -- mixing R_synth (plain R_wave) with R_wave_v2-based truth/search
    would compare two different, inconsistent forward models.

    IMPORTANT: `a` must match the fixed incident-wave scalar that was
    passed to whichever cnct_search_*_Gd call produced s_hat_list. The
    search fits s_hat against a template already built with that `a`
    baked in (h = R_wave_v2(..., a=a_search, ...), then data ~ s_hat*h),
    so reconstructing the fit needs the SAME a, not s_hat alone:

        Er = s_hat * R_wave_v2(..., a=a_search, cn, ct, Gd, eps)
           = R_wave_v2(..., a=s_hat*a_search, cn, ct, Gd, eps)   (R_wave_v2 is linear in a)

    a=1.0 (the default) reproduces the previous behavior exactly for the
    common case where the search itself used a=1.
    """
    req = ("s_hat", "cn", "ct")
    missing = [k for k in req if k not in s_hat_list]
    if missing:
        raise ValueError(f"s_hat_list missing keys: {missing}")

    s_arr = np.asarray(s_hat_list["s_hat"], float).ravel()
    cn_arr = np.asarray(s_hat_list["cn"], float).ravel()
    ct_arr = np.asarray(s_hat_list["ct"], float).ravel()
    if not (s_arr.size == cn_arr.size == ct_arr.size):
        raise ValueError("s_hat, cn, ct must have the same length")

    n_m = s_arr.size
    if n_m == 0:
        raise ValueError("s_hat_list is empty")

    idx = int(idx)
    a = float(a)
    Er_time = Er_omega = None

    for m in range(n_m):
        syn = R_wave_v2(
            ref_data, inc_wave, idx, float(s_arr[m]) * a,
            vp, vs, rho, float(cn_arr[m]), float(ct_arr[m]),
            Gd=Gd, eps=eps, wave=wave,
        )
        if Er_time is None:
            nt = syn["Er_time"].size
            nf = syn["Er_omega"].size
            Er_time = np.empty((n_m, nt), float)
            Er_omega = np.empty((n_m, nf), complex)
            freqs = np.asarray(syn["freqs"])
            taxis = np.asarray(syn["time"])
            angle = float(syn["angle"])
        Er_time[m] = np.asarray(syn["Er_time"], float)
        Er_omega[m] = np.asarray(syn["Er_omega"])

    return {
        "Er_time": Er_time,
        "Er_omega": Er_omega,
        "time": taxis,
        "freqs": freqs,
        "cn": cn_arr,
        "ct": ct_arr,
        "s_hat": s_arr,
        "idx": idx,
        "angle": angle,
    }


def R_synth_Gd_cc(
    s_hat_list,
    lag_arr,
    ref_data,
    inc_wave,
    idx,
    vp,
    vs,
    rho,
    Gd,
    eps=0.005,
    a=1.0,
    wave="SS",
):
    """
    Same as R_synth_Gd, but also replays the cross-correlation shift that
    cnct_search_Rtime_Gd_cc / cnct_search_Rtime_ratio_Gd_cc / (real-data)
    cnct_search_Rtime_Gd_cc_obs applied to each candidate model trace
    before fitting s_hat. Use this, not R_synth_Gd, to reconstruct a
    fitted waveform from a _cc search's results.

    s_hat_list (GLOBAL: same for every trace in the search, keyed only by
    grid point) comes from the existing s_hat_at_cnct_list (plain grid) or
    s_hat_at_ratio_ct_list (ratio grid), unchanged. lag_arr (PER-TRACE) is
    fetched separately for this `idx`, at the same grid points, via
    lag_at_list(gsR, s_hat_list['i'], s_hat_list['j'], idx) -- kept as a
    distinct argument rather than folded into s_hat_list, since s_hat/cn/ct
    and lag genuinely have different scopes (see lag_at_list's docstring).

    The search computed, per grid point:
        h_cc, lag = align_by_cc(h, D_all[k]);  H_all[k] = h_cc
    i.e. h_cc = roll(h, -lag). Reproducing that fit here means: build the
    raw R_wave_v2 trace, then roll it by -lag before returning -- so the
    reconstructed synthetic lines up with the observed/synthetic-truth
    trace exactly as it did when s_hat/cost were computed. `Er_omega` is
    returned UNshifted (the raw model spectrum); the lag is a time-domain
    realignment only, not folded into the returned spectrum.

    See R_synth_Gd's docstring re: `a` must match the fixed incident-wave
    scalar used by the search that produced s_hat_list.
    """
    req = ("s_hat", "cn", "ct")
    missing = [key for key in req if key not in s_hat_list]
    if missing:
        raise ValueError(f"s_hat_list missing keys: {missing}")

    s_arr = np.asarray(s_hat_list["s_hat"], float).ravel()
    cn_arr = np.asarray(s_hat_list["cn"], float).ravel()
    ct_arr = np.asarray(s_hat_list["ct"], float).ravel()
    lag_arr = np.asarray(lag_arr, int).ravel()
    if not (s_arr.size == cn_arr.size == ct_arr.size == lag_arr.size):
        raise ValueError("s_hat, cn, ct, lag_arr must have the same length")

    n_m = s_arr.size
    if n_m == 0:
        raise ValueError("s_hat_list is empty")

    idx = int(idx)
    a = float(a)
    Er_time = Er_omega = None

    for m in range(n_m):
        syn = R_wave_v2(
            ref_data, inc_wave, idx, float(s_arr[m]) * a,
            vp, vs, rho, float(cn_arr[m]), float(ct_arr[m]),
            Gd=Gd, eps=eps, wave=wave,
        )
        er_time_aligned = np.roll(
            np.asarray(syn["Er_time"], float), -int(lag_arr[m])
        )

        if Er_time is None:
            nt = er_time_aligned.size
            nf = syn["Er_omega"].size
            Er_time = np.empty((n_m, nt), float)
            Er_omega = np.empty((n_m, nf), complex)
            freqs = np.asarray(syn["freqs"])
            taxis = np.asarray(syn["time"])
            angle = float(syn["angle"])
        Er_time[m] = er_time_aligned
        Er_omega[m] = np.asarray(syn["Er_omega"])

    return {
        "Er_time": Er_time,
        "Er_omega": Er_omega,
        "time": taxis,
        "freqs": freqs,
        "cn": cn_arr,
        "ct": ct_arr,
        "s_hat": s_arr,
        "lag": lag_arr,
        "idx": idx,
        "angle": angle,
    }


# ============================================================================
# Frequency-domain (amplitude-spectrum) compliance search with attenuation
# Procedure: tests/spectrum_inversion/frequency_inversion_procedure.md
# ============================================================================

_RSPEC_DEFAULTS = {
    "band": (10.0, 300.0),   # fit band [Hz], positive frequencies only
    "misfit": "log",         # "log" (beta fixed), "log_Q" (beta = 1/Q free if beta None), "linear"
    "W": "ones",             # frequency weight: "ones" = 1 (flat), "D2" = |D|^2
    "beta": 0.0,             # 0 = no Q (default); float fixes beta = 1/Q; None + "log_Q" = estimated
    "rd": None,              # direct (STF) path length [m], dir_data['mean_dist_source']
    "a": 1.0,                # scalar on the STF in H (synthetic tests; real data a = 1)
}


def _rspec_opts(freq_opts):
    """
    Purpose: merge the user's freq_opts with _RSPEC_DEFAULTS and validate.
    Input: freq_opts (dict or None).
    Output: dict with every key of _RSPEC_DEFAULTS; 'beta_mode' added:
        "free" (closed-form beta) or "fixed" (beta = fo['beta']).
    """
    fo = dict(_RSPEC_DEFAULTS)
    fo.update(freq_opts or {})
    unknown = set(fo) - set(_RSPEC_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown freq_opts keys {sorted(unknown)}; allowed {sorted(_RSPEC_DEFAULTS)}")
    if fo["misfit"] not in ("log_Q", "log", "linear"):
        raise ValueError("freq_opts['misfit'] must be 'log_Q', 'log' or 'linear'")
    if fo["W"] not in ("ones", "D2"):
        raise ValueError("freq_opts['W'] must be 'ones' or 'D2'")
    fmin, fmax = map(float, fo["band"])
    if not (0 <= fmin < fmax):
        raise ValueError("freq_opts['band'] must be (fmin, fmax) with 0 <= fmin < fmax")
    fo["band"] = (fmin, fmax)
    fo["a"] = float(fo["a"])
    if fo["a"] <= 0:
        raise ValueError("freq_opts['a'] must be > 0")

    # beta: free only for "log_Q" with beta=None; otherwise fixed (default 0 = no Q)
    if fo["beta"] is None and fo["misfit"] == "log_Q":
        fo["beta_mode"] = "free"
    else:
        fo["beta"] = 0.0 if fo["beta"] is None else float(fo["beta"])
        if fo["beta"] < 0:
            raise ValueError("freq_opts['beta'] = 1/Q must be >= 0")
        fo["beta_mode"] = "fixed"
    if fo["rd"] is None and (fo["beta_mode"] == "free" or fo["beta"] > 0):
        raise ValueError("freq_opts['rd'] (dir_data['mean_dist_source']) is needed to model attenuation")
    return fo


def _rspec_setup(ref_data, inc_wave, Gd, idx_array, trace_w, vs, eps, fo):
    """
    Purpose: everything of the spectrum search that does not depend on
        (cn, ct), computed once before the grid (section 7 of the md).
    Input: ref_data (needs reflection_traces (its .fft from fft_traces is
        used if present, else the FFT of .data), freqs, omega, Gr, theta_i_deg,
        time_window; dist_source_rr if attenuation is modeled; 'noise' for
        the data water level); inc_wave (RAW STF dict); Gd (G_d(omega));
        idx_array (kept traces); trace_w (n_tr,) trace weights; vs; eps
        (G_d stabilization fraction, as R_wave_v2); fo (from _rspec_opts).
    Output: dict with (n_tr = traces, nf = frequencies in band)
        band (bool mask on ref_data['freqs']), f (nf,), omega (nf,),
        K (n_tr, nf) complex: a*Gr*S*conj(Gd)/(|Gd|^2+eps_val^2) = the
            R_wave_v2 spectrum without R (Gr, Gd already hold 1/rr, 1/rd);
        absD, lnD (n_tr, nf): |D| and ln sqrt(|D|^2 + delta_d^2);
        u (n_tr, nf): w_i * W_i(omega); x (n_tr, nf): |omega|(rr-rd)/(2vs);
        ps (n_tr,): ray parameter sin(theta_i)/vs; delta_d (n_tr,).
    """
    f_all = np.asarray(ref_data["freqs"], float)
    fmin, fmax = fo["band"]
    band = (f_all >= fmin) & (f_all <= fmax)            # one-sided band (f >= 0)
    if band.sum() < 3:
        raise ValueError(f"band {fo['band']} holds < 3 frequency samples")
    f = f_all[band]
    omega = np.asarray(ref_data["omega"], float)[band]

    # ---- data spectra: FFT of the full (already windowed) traces, no re-windowing.
    # Uses reflection_traces.fft (fft_traces) when present and the same shape
    # as .data; otherwise computes it here.
    tr = ref_data["reflection_traces"]
    F = getattr(tr, "fft", None)
    if F is not None and np.shape(F) == np.shape(tr.data):
        absD = np.abs(np.asarray(F)[idx_array][:, band])
    else:
        D_t = np.asarray(tr.data, float)[idx_array]
        absD = np.abs(np.fft.fft(D_t, axis=1)[:, band])
    # water level = noise amplitude in the tapered window, sqrt(sigma2 * sum w^2)
    if "noise" in ref_data:
        t = np.asarray(ref_data["reflection_traces"].taxis, float)
        tb, te = ref_data["time_window"]
        nwin = int(np.count_nonzero((t >= tb) & (t <= te)))
        w_tap = tukey(nwin, alpha=float(ref_data.get("taper_alpha", 0.4)), sym=True)
        sigma2 = np.asarray(ref_data["noise"], float).ravel()[idx_array]
        delta_d = np.sqrt(sigma2 * np.sum(w_tap ** 2))
    else:                                               # fallback: 1e-3 of each trace's peak
        delta_d = 1e-3 * absD.max(axis=1)
    lnD = 0.5 * np.log(absD ** 2 + delta_d[:, None] ** 2)

    # ---- weights u = w_i * W_i(omega)
    W = absD ** 2 if fo["W"] == "D2" else np.ones_like(absD)
    u = trace_w[:, None] * W

    # ---- fixed model part K (same stabilized G_d division as R_wave_v2)
    S = fo["a"] * np.asarray(_as_inc_wave(inc_wave)["S_omega"])[band]
    Gd = np.asarray(Gd, complex).ravel()
    eps_val = float(np.max(np.abs(Gd))) * float(eps)
    gd_term = np.conj(Gd[band]) / (np.abs(Gd[band]) ** 2 + eps_val ** 2)
    Gr = np.asarray(ref_data["Gr"], float)[idx_array][:, band]
    K = Gr * (S * gd_term)[None, :]

    # ---- attenuation abscissa x = |omega| (rr - rd) / (2 vs); zeros if not modeled
    if fo["rd"] is not None:
        rr = np.asarray(ref_data["dist_source_rr"], float)[idx_array]
        x = np.abs(omega)[None, :] * (rr - float(fo["rd"]))[:, None] / (2.0 * float(vs))
    else:
        x = np.zeros_like(absD)

    theta = np.deg2rad(np.asarray(ref_data["theta_i_deg"], float)[idx_array])
    return {"band": band, "f": f, "omega": omega, "K": K, "absD": absD, "lnD": lnD,
            "u": u, "x": x, "ps": np.sin(theta) / float(vs), "delta_d": delta_d}


def _rspec_fit(absH, st, fo):
    """
    Purpose: closed-form s (and beta) plus per-trace cost at a batch of grid
        nodes (sections 4-6 of the md), vectorized over the batch.
    Input: absH (n_b, n_tr, nf) |H| at n_b nodes; st (from _rspec_setup);
        fo (from _rspec_opts).
    Output: (s_hat (n_b,), beta (n_b,), Csp (n_b, n_tr)) with
        Csp[b, i] = 0.5 * sum_omega u_i(omega) r_i(omega)^2, r the log
        misfit y - ln s + beta x (or |D| - s|H|e^{-beta x} for "linear").
    """
    u, x = st["u"][None], st["x"][None]                 # (1, n_tr, nf)
    ax = (1, 2)                                         # sum over traces and frequencies
    su = u.sum(axis=ax)

    if fo["misfit"] == "linear":                        # amplitude misfit, beta fixed
        beta = np.full(absH.shape[0], fo["beta"])
        m = absH * np.exp(-fo["beta"] * x)
        s = (u * st["absD"][None] * m).sum(axis=ax) / (u * m ** 2).sum(axis=ax)
        r = st["absD"][None] - s[:, None, None] * m
        return s, beta, 0.5 * (u * r ** 2).sum(axis=2)

    # log misfit: y = ln|D| - ln|H|, water level delta_m = 1e-3 max|H_i|
    delta_m = 1e-3 * absH.max(axis=2, keepdims=True)
    y = st["lnD"][None] - 0.5 * np.log(absH ** 2 + delta_m ** 2)
    if fo["beta_mode"] == "fixed":
        beta = np.full(absH.shape[0], fo["beta"])
        lns = (u * (y + fo["beta"] * x)).sum(axis=ax) / su
    else:                                               # weighted line fit in x
        xbar = (u * x).sum(axis=ax) / su
        ybar = (u * y).sum(axis=ax) / su
        dx = x - xbar[:, None, None]
        sxx = (u * dx ** 2).sum(axis=ax)
        if np.any(sxx <= 0):
            raise ValueError("no spread in x = |omega|(rr-rd)/(2vs): beta cannot be estimated")
        beta = -(u * dx * (y - ybar[:, None, None])).sum(axis=ax) / sxx
        neg = beta < 0                                  # Q < 0 not physical -> beta = 0
        beta = np.where(neg, 0.0, beta)
        lns = np.where(neg, ybar, ybar + beta * xbar)
    r = y - lns[:, None, None] + beta[:, None, None] * x
    return np.exp(lns), beta, 0.5 * (u * r ** 2).sum(axis=2)


def cnct_search_Rspectrum(
    ref_data,
    inc_data,
    idx_array,
    medium,
    search_param,
    true_cnct=None,
):
    """
    Purpose
    -------
    Frequency-domain (amplitude-spectrum) compliance grid search on REAL
    observed reflection data, with a global scale s and constant-Q
    attenuation beta = 1/Q (tests/spectrum_inversion/
    frequency_inversion_procedure.md). Counterpart of cnct_searchR_obs
    (time domain); no alignment (amplitude only, so cc / max_lag do not
    exist here).

    Model per trace i and frequency omega in the band:
        |E_i| = s * |H_i(omega; cn, ct)| * exp(-beta x_i),
        x_i   = |omega| (rr_i - rd) / (2 vs)
        H_i   = a * Gr_i * S * R(omega; theta_i, cn, ct) * conj(Gd)
                / (|Gd|^2 + eps_val^2),   eps_val = eps * max|Gd|
    (Gr holds 1/rr and Gd 1/rd, so rd/rr is inside Gr/Gd.) a = 1 for real
    data; with synthetic data built at a = 1 the search should return
    s = 1/a.
    Cost (misfit "log_Q"/"log"):
        C(cn, ct) = 1/2 sum_i sum_omega u_i(omega) r_i(omega)^2
        r_i = ln|D_i| - ln|H_i| - ln s + beta x_i,  u_i = w_i W_i(omega)
    s and beta in closed form at each node (weighted straight-line fit in
    x; beta < 0 -> beta = 0). w_i = 1 for every trace unless trace_opts
    gives a weight (then normalized to mean 1). W = 1 (default) or |D|^2.
    ln 0 guarded by water levels: data delta = noise amplitude
    (ref_data['noise']), model delta = 1e-3 max|H_i|.

    Input
    -----
    ref_data : dict, needs reflection_traces, freqs, omega, Gr,
        theta_i_deg, time_window; dist_source_rr when attenuation is
        modeled; 'noise' (optional) for the data water level.
    inc_data : (inc_wave, Gd) -- RAW direct-arrival trace dict
        (dir_data['stf_trace']) and G_d(omega).
    idx_array : array-like of int, trace indices to fit.
    medium : (vp, vs, rho, wave).
    search_param : (search_mode, grid, eps, search_space, nsearch,
                    constraint[, trace_opts[, freq_opts]])
        search_mode : "full" or "constrained".
        grid : "cn_ctratio" ((cn/ct, ct) grid) or "non_ratio" ((cn, ct)).
        eps : float, G_d stabilization fraction.
        search_space : ((p_min, p_max), (ct_min, ct_max)), p = cn/ct or cn.
        nsearch : (n_p, n_ct), log-spaced samples.
        constraint : (s_hat_min, s_hat_max) for "constrained", else None.
        trace_opts : dict or None (as cnct_searchR_obs), e.g.
            {"select": ("snr_noise", 7.0), "weight": "w_noise"}.
        freq_opts : dict or None, keys (defaults in _RSPEC_DEFAULTS):
            band (10, 300) Hz; misfit "log" (default) | "log_Q" | "linear";
            W "ones" (default, flat) | "D2"; beta 0.0 (default, no Q),
            fixed float, or None with misfit "log_Q" (Q estimated);
            rd direct path [m] (needed when attenuation is modeled);
            a 1.0.
    true_cnct : (cn, ct) or None, true node spliced into the axes.

    Output
    ------
    "full" -> out_full; "constrained" -> (out_full, out_constraint).
        out_full : cnct_search_pack fields ((cn, ct) grid) or
            _ratio_grid_pack fields (ratio grid), with domain="frequency",
            plus 'beta_hat', 'Q_hat' (= 1/beta, inf if beta = 0)
            (n_p, n_ct), 'data_energy' (1/2 sum u lnD^2, or 1/2 sum u |D|^2
            for "linear"), 'cost_norm' = Csp_sum / data_energy,
            'freq_opts', 'band', 'n_freq', 'delta_d', trace bookkeeping
            ('idx_input', 'idx_dropped', 'trace_w', 'trace_opts'),
            'search_mode', 'lag_samples' = None, 'cc_max' = None.
        out_constraint : constrain_search_Rspectrum(out_full, constraint).
    """
    inc_wave, Gd = inc_data
    vp, vs, rho, wave = medium

    # ---- search_param
    if len(search_param) not in (6, 7, 8):
        raise ValueError("search_param must be (search_mode, grid, eps, search_space, "
                         "nsearch, constraint[, trace_opts[, freq_opts]])")
    search_mode, grid, eps, search_space, nsearch, constraint = search_param[:6]
    trace_opts = search_param[6] if len(search_param) >= 7 else None
    fo = _rspec_opts(search_param[7] if len(search_param) == 8 else None)
    if search_mode not in ("full", "constrained"):
        raise ValueError("search_mode must be 'full' or 'constrained'")
    if grid not in ("cn_ctratio", "non_ratio"):
        raise ValueError("grid must be 'cn_ctratio' or 'non_ratio'")
    if search_mode == "constrained" and constraint is None:
        raise ValueError("search_mode='constrained' needs constraint=(s_hat_min, s_hat_max)")
    is_ratio = grid == "cn_ctratio"

    # ---- trace selection / weights (w_i = 1 unless trace_opts gives a weight)
    idx_input = np.asarray(idx_array, int).ravel()
    if idx_input.size == 0:
        raise ValueError("idx_array is empty")
    idx_array, idx_dropped, w_keep = _apply_trace_opts(ref_data, idx_input, trace_opts)
    n_tr = idx_array.size
    trace_w = np.ones(n_tr) if w_keep is None else w_keep / w_keep.mean()

    # ---- precompute data / geometry / fixed model part
    st = _rspec_setup(ref_data, inc_wave, Gd, idx_array, trace_w, vs, eps, fo)

    # ---- axes (true node as (ratio, ct) on the ratio grid)
    true_p = None
    if true_cnct is not None:
        cn_t, ct_t = map(float, true_cnct)
        true_p = (cn_t / ct_t, ct_t) if is_ratio else (cn_t, ct_t)
    axes = cnct_search_axes(search_space, nsearch, true_cnct=true_p)
    p_ax, ct = axes["cn"], axes["ct"]
    n_p, n_ct = axes["n_cn"], axes["n_ct"]

    Csp_all = np.empty((n_tr, n_p, n_ct), float)
    s_hat = np.empty((n_p, n_ct), float)
    beta_hat = np.empty((n_p, n_ct), float)
    ps = st["ps"][None, :, None]                        # (1, n_tr, 1)
    om = st["omega"][None, None, :]                     # (1, 1, nf)

    # ---- loop over ct, vectorized over the p axis, traces and frequencies
    for j, ct_j in enumerate(ct):
        cn_vec = (p_ax * ct_j if is_ratio else p_ax)[:, None, None]   # (n_p, 1, 1)
        with np.errstate(invalid="ignore"):             # sqrt(<0) branch discarded by np.where
            R, _ = RPS_coef(vp, vs, ps, rho, om, cn_vec, ct_j, wave)
        absH = np.abs(st["K"][None] * R)                # (n_p, n_tr, nf)
        s, b, C = _rspec_fit(absH, st, fo)
        s_hat[:, j], beta_hat[:, j] = s, b
        Csp_all[:, :, j] = C.T

    # ---- pack with the existing helpers, then mark the domain
    a = fo["a"]
    if is_ratio:
        out_full = _ratio_grid_pack(
            Csp_all, s_hat, p_ax, ct, p_ax[:, None] * ct[None, :], idx_array, a,
            true_pair=None if true_cnct is None else (cn_t, ct_t),
            ratio_true=None if true_p is None else true_p[0],
            i_true=axes["i_true"], j_true=axes["j_true"],
            lag_samples=None,
        )
    else:
        out_full = cnct_search_pack(
            Csp_all, s_hat, p_ax, ct, idx_array, a,
            axes=axes, mode="R", domain="frequency",
            lag_samples=None, grid="cn_ct",
        )

    y0 = st["absD"] if fo["misfit"] == "linear" else st["lnD"]
    data_energy = float(0.5 * np.sum(st["u"] * y0 ** 2))
    with np.errstate(divide="ignore"):
        Q_hat = 1.0 / beta_hat                          # inf where beta = 0
    out_full.update({
        "domain": "frequency",
        "beta_hat": beta_hat,
        "Q_hat": Q_hat,
        "data_energy": data_energy,
        "cost_norm": out_full["Csp_sum"] / data_energy,
        "freq_opts": fo,
        "band": fo["band"],
        "n_freq": int(st["f"].size),
        "delta_d": st["delta_d"],
        "idx_input": idx_input,
        "idx_dropped": idx_dropped,
        "trace_w": trace_w,
        "trace_opts": trace_opts,
        "search_mode": search_mode,
        "cc_max": None,
    })
    if out_full.get("i_true") is not None:
        out_full["beta_true_node"] = float(beta_hat[out_full["i_true"], out_full["j_true"]])

    if search_mode == "constrained":
        return out_full, constrain_search_Rspectrum(out_full, constraint)
    return out_full


def constrain_search_Rspectrum(out_full, s_hat_range):
    """
    Purpose: s_hat-constrained pack of a cnct_search_Rspectrum result,
        without re-running the forward model. Calls the existing
        constrain_searchR (either grid) and then restores the
        frequency-domain fields it does not know about.
    Input: out_full (dict, full pack of cnct_search_Rspectrum, or the path
        of a run pickle (out_full, ...)); s_hat_range (s_min, s_max).
    Output: out_constraint (dict): constrain_searchR's fields, with
        domain="frequency", freq_opts/band/n_freq copied, and
        beta_hat / Q_hat NaN outside 'mask'.
    """
    if not isinstance(out_full, dict):
        with open(Path(out_full), "rb") as f:
            out_full = pickle.load(f)[0]
    out_c = constrain_searchR(out_full, s_hat_range)
    out_c["domain"] = out_full.get("domain", "frequency")      # time packs with Q stay "time"
    for k in ("freq_opts", "band", "n_freq", "search_mode", "Q_list", "rd", "Q_opts"):
        if k in out_full:
            out_c[k] = out_full[k]
    mask = out_c["mask"]
    for k in ("beta_hat", "Q_hat"):
        out_c[k] = np.where(mask, np.asarray(out_full[k], float), np.nan)
    return out_c


def R_batch_InvQ(
    ref_data,
    inc_wave,
    idx_array,
    a,
    vp,
    vs,
    rho,
    cn,
    ct,
    Gd,
    eps,
    beta,
    rd,
    wave="SS",
):
    """
    Purpose
    -------
    Batch version of R_waveInvQ: R_wave_batch_v2 forward reflected DAS
    spectra plus constant-Q attenuation of each reflected path relative to
    the direct (STF) path:

        ER_Q,i(ω) = ER_i(ω) * exp(-|ω| * beta / (2 vs) * (rr_i - rd))

    ER_i is R_wave_batch_v2's Er_omega row i; rr_i = ref_data
    ['dist_source_rr'][idx_i]. Amplitude-only factor with |ω| (Hermitian,
    Er_time real). beta = 0 reproduces R_wave_batch_v2 exactly.

    Input
    -----
    ref_data, inc_wave, idx_array, a, vp, vs, rho, cn, ct, Gd, eps, wave :
        as in R_wave_batch_v2 (ref_data also needs 'dist_source_rr').
    beta : float, inverse quality factor 1/Q (>= 0; 0 = no attenuation).
    rd : float, direct-wave path length [m] (dir_data['mean_dist_source']).

    Output
    ------
    dict : R_wave_batch_v2's keys (Er_omega (n, nf), Er_time (n, nt), freqs,
        time, idx, angle, a, Gd, eps, eps_val) with Er_omega / Er_time
        attenuated, plus 'beta', 'rr' (n,), 'rd', 'att' (n, nf).
    """
    beta = float(beta)
    if beta < 0:
        raise ValueError("beta = 1/Q must be >= 0")
    if "ref_data" in ref_data and "Gr" not in ref_data:
        ref_data = ref_data["ref_data"]
    if "dist_source_rr" not in ref_data:
        raise ValueError("ref_data missing key: 'dist_source_rr'")

    # unattenuated batch (same R, same stabilized Gd division)
    out = R_wave_batch_v2(ref_data, inc_wave, idx_array, a, vp, vs, rho, cn, ct,
                          Gd=Gd, eps=eps, wave=wave)

    rr = np.asarray(ref_data["dist_source_rr"], float)[out["idx"]]      # (n,)
    omega = np.abs(np.asarray(ref_data["omega"], float))                # |ω|: Hermitian
    # (n, nf) attenuation, all traces at once
    att = np.exp(-(beta / (2.0 * float(vs))) * np.outer(rr - float(rd), omega))

    Er_omega = out["Er_omega"] * att
    out.update({
        "Er_omega": Er_omega,
        "Er_time": np.stack([real_wave(e) for e in Er_omega]),
        "beta": beta,
        "rr": rr,
        "rd": float(rd),
        "att": att,
    })
    return out


def Add_constrain_cntct_output(out_full, s_range=None, Q_range=None):
    """
    Purpose
    -------
    Post-processing of a FULL-mode cnct_search_Rspectrum /
    cnct_search_Rspectrum_syntest result: adds the constrained pack for a
    user range on s_hat, on Q_hat, or on both, without re-running the
    search. Nodes are filtered as they are (free-fit s_hat and Q_hat of
    each node, no refit): a node is kept when
        s_range[0] <= s_hat <= s_range[1]   (if s_range is given) and
        Q_range[0] <= Q_hat <= Q_range[1]   (if Q_range is given),
    and the best (cn, ct) is the lowest-cost kept node. Q_hat = inf
    (beta = 0) is kept only if Q_range[1] is inf (plain comparison).
    Uses constrain_search_Rspectrum (unchanged): nodes failing Q_range get
    s_hat = NaN in a copy, so its s_hat test drops them too.

    Input
    -----
    out_full : dict, full pack (or an (out_full, ...) tuple; out_full used).
    s_range : (s_min, s_max) with s_min < s_max, or None (no s constraint).
    Q_range : (Q_min, Q_max) with Q_min < Q_max, or None (no Q constraint).
        At least one of s_range / Q_range must be given.

    Output
    ------
    (out_full, out_constraint) -- same form as cnct_search_Rspectrum in
        "constrained" mode. out_constraint: constrain_search_Rspectrum
        fields (Csp_sum, s_hat, beta_hat, Q_hat, cost_norm NaN outside
        'mask'; best node on the kept nodes), with 's_hat_range' and
        'Q_range' set to the user input (None if not given), and
        'constraint_type' = "s", "Q" or "s_Q" (which ranges were applied).
    """
    if isinstance(out_full, tuple):
        out_full = out_full[0]
    if s_range is None and Q_range is None:
        raise ValueError("give s_range, Q_range or both")

    def _check(r, name):                                  # (lo, hi) with lo < hi
        lo, hi = map(float, r)
        if not lo < hi:
            raise ValueError(f"{name} must be (min, max) with min < max, got {r}")
        return lo, hi

    s_hat = np.asarray(out_full["s_hat"], float)
    keep_Q = np.ones(s_hat.shape, bool)
    if Q_range is not None:
        Q_lo, Q_hi = _check(Q_range, "Q_range")
        Q = np.asarray(out_full["Q_hat"], float)
        keep_Q = (Q >= Q_lo) & (Q <= Q_hi)                # inclusive bounds
    s_lim = _check(s_range, "s_range") if s_range is not None else (-np.inf, np.inf)

    # nodes failing Q -> s_hat NaN (NaN fails every comparison, so dropped)
    tmp = dict(out_full)
    tmp["s_hat"] = np.where(keep_Q, s_hat, np.nan)
    out_c = constrain_search_Rspectrum(tmp, s_lim)
    out_c["s_hat"] = np.where(out_c["mask"], s_hat, np.nan)   # real s_hat on kept nodes
    out_c["s_hat_range"] = None if s_range is None else s_lim
    out_c["Q_range"] = None if Q_range is None else (Q_lo, Q_hi)
    out_c["constraint_type"] = ("s_Q" if s_range is not None and Q_range is not None
                                else "s" if s_range is not None else "Q")
    return out_full, out_c


def Add_constrain_cntct_file(pkl_path, s_range=None, Q_range=None):
    """
    Purpose
    -------
    File version of Add_constrain_cntct_output: loads a saved
    cnct_search_Rspectrum run (<run>.pkl written by cnct_search), adds the
    constrained pack for s_range / Q_range / both, and writes the result
    back into the SAME <run>.pkl as (out_full, out_constraint). The full
    pack is unchanged; a constrained pack already in the file is replaced.
    The requested constraint is also recorded in the run's metadata,
    <run>_data.pkl and <run>_data.json (key 'constraint_added'), when they
    exist.

    Input
    -----
    pkl_path : str or Path, <run>.pkl (dict = full-mode run, or
        (out_full, out_constraint) tuple).
    s_range, Q_range : as Add_constrain_cntct_output ((min, max) inclusive,
        min < max, or None; at least one given).

    Output
    ------
    (out_full, out_constraint) -- also written to pkl_path.
    """
    import cnct_search as _cs                             # local: write_data_json

    pkl_path = Path(pkl_path)
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)
    out_full, out_c = Add_constrain_cntct_output(res, s_range=s_range, Q_range=Q_range)
    with open(pkl_path, "wb") as f:
        pickle.dump((out_full, out_c), f)

    # metadata: record the constraint next to the run's input parameters
    data_path = pkl_path.with_name(f"{pkl_path.stem}_data.pkl")
    if data_path.exists():
        with open(data_path, "rb") as f:
            input_data = pickle.load(f)
        input_data["constraint_added"] = {"constraint_type": out_c["constraint_type"],
                                          "s_range": out_c["s_hat_range"],
                                          "Q_range": out_c["Q_range"]}
        with open(data_path, "wb") as f:
            pickle.dump(input_data, f)
        _cs.write_data_json(input_data, (out_full, out_c),
                            pkl_path.with_name(f"{pkl_path.stem}_data.json"))
    return out_full, out_c


def fft_traces(ref_data):
    """
    Purpose
    -------
    FFT of every reflected trace, stored on the traces object:
        ref_data['reflection_traces'].fft = np.fft.fft(data, axis=1)
    Full two-sided spectrum of the stored (already windowed) traces, in
    np.fft order, so it matches ref_data['freqs'] (= np.fft.fftfreq(nt, dt),
    see tests/spectrum_inversion/test_fft_traces.py). Same transform
    _rspec_setup uses for |D| in cnct_search_Rspectrum. (The object's own
    'fftdata' attribute is left untouched.)

    Input
    -----
    ref_data : dict with 'reflection_traces' (Data2D_XT, .data (n_tr, nt)),
        or the full ref_stf_data pickle (key 'ref_data').

    Output
    ------
    ref_data : the same dict (updated in place), with
        ref_data['reflection_traces'].fft : complex (n_tr, nt).
    """
    if "ref_data" in ref_data and "reflection_traces" not in ref_data:
        ref_data = ref_data["ref_data"]
    tr = ref_data["reflection_traces"]
    tr.fft = np.fft.fft(np.asarray(tr.data, float), axis=1)   # all traces at once
    return ref_data


def select_DAS_traces(ref_data, cc_thresholds):
    """
    Purpose
    -------
    Select (extract) the reflected traces whose max cc of the positive and of the
    negative lobe are above the thresholds, and subset the whole dictionary to
    them: the DAS object (data, daxis and the .fft of fft_traces, if present) and
    every per-trace array (trace_ids, max cc, angles, rays, distances, noise, ...).
    Arrays that are not per trace (normal, freqs, omega) and scalars are kept.

    Input
    -----
    ref_data : dict, reflected data (ref_stf_data['ref_data'], e.g. after fft_traces)
    cc_thresholds : (pos_thr, neg_thr)
        keep cc_pos > pos_thr and cc_neg > neg_thr; None skips that lobe.

    Output
    ------
    ref_sel : new dict (ref_data is not modified) with the selected traces, plus
        'idx_select' : (m,) positions of the selected traces in the input ref_data
        'cc_thresholds' : (pos_thr, neg_thr) used
    """
    pos_thr, neg_thr = cc_thresholds
    das = ref_data["reflection_traces"]
    n_tr = das.data.shape[0]

    # selection mask (NaN cc never passes)
    keep = np.ones(n_tr, bool)
    if pos_thr is not None:
        cc_pos = np.asarray(ref_data["max_cc_positive"], float)
        keep &= np.isfinite(cc_pos) & (cc_pos > pos_thr)
    if neg_thr is not None:
        cc_neg = np.asarray(ref_data["max_cc_negative"], float)
        keep &= np.isfinite(cc_neg) & (cc_neg > neg_thr)
    idx = np.flatnonzero(keep)

    # DAS object: copy, then keep the selected rows of data, daxis and fft
    das_sel = das.copy()
    das_sel.data = np.asarray(das.data)[idx].copy()
    das_sel.daxis = np.asarray(das.daxis)[idx].copy()
    if getattr(das, "fft", None) is not None:
        das_sel.fft = np.asarray(das.fft)[idx].copy()

    # per-trace arrays: first axis = n_tr (keys that are not per trace are excluded)
    not_per_trace = {"reflection_traces", "normal", "freqs", "omega"}
    ref_sel = {}
    for k, v in ref_data.items():
        if (k not in not_per_trace and isinstance(v, np.ndarray)
                and v.ndim >= 1 and v.shape[0] == n_tr):
            ref_sel[k] = v[idx].copy()
        else:
            ref_sel[k] = copy.deepcopy(v) if k != "reflection_traces" else das_sel
    ref_sel["idx_select"] = idx
    ref_sel["cc_thresholds"] = (pos_thr, neg_thr)
    return ref_sel
