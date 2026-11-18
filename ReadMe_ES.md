# ScatteringDAS: notebooks, modules and inversion functions

## Notebooks

| Notebook | Purpose |
|---|---|
| `DASrefData.ipynb` | Reflected and STF data processing |
| `DASscatData.ipynb` | Paired reflected and transmitted data |
| `DASrefInv.ipynb` | Inversion: STF / reflection data |
| `DASscatInv.ipynb` | Inversion: reflection predicted from transmission |
| `DASrefModel_source.ipynb` | Forward-inverse test, reflected data only |
| `DASrefModel_ricker.ipynb` | Forward-inverse test, paired reflected and transmitted data |

## Modules (`py_files/`)

- `DASref_modeling.py`: STF / reflection forward model and searches (`cnct_searchR_obs`)
- `DAS_modeling.py`: reflection / transmission forward model and searches (`RfromT`,
  `cnct_search_RfromT`), and the waveform fitness of the best fits per group (`waveform_fitness`)
- `DAS_processing.py`: data processing (windows, filters, noise)
- `DAS_plots.py`: plots
- `cnct_search.py`: save wrappers for the searches (`cnct_search`, `cnct_search_scat`) and
  background jobs (`submit_detached`, section 3)

## Inversion outputs

- STF / reflection data: `./inv_results/cons7`, using `cnct_searchR_obs()`
- Reflection / transmission data: `./inv_results/RfromT/<run>/` (e.g. `weighted_cap`,
  `weighted_ratio_cn`, `weighted_ratio_cn_snr10`), using `cnct_search_RfromT()`

## Inversion functions

1. `cnct_searchR_obs()`: STF / reflection data (`DASref_modeling.py`)
2. `cnct_search_RfromT()`: reflection predicted from the observed transmission (`DAS_modeling.py`)

Both are time-domain compliance grid searches (waveform fit). Each has a save wrapper in
`py_files/cnct_search.py` that runs the search and saves it in `inv_results/<name>/`:

| Search function | Save wrapper |
|---|---|
| `DASrefmod.cnct_searchR_obs` | `cnct_search(ref_data, inc_data, idx, medium, search_param, search_fn, "<name>", true_cnct=None, out_root=None)` |
| `DASmod.cnct_search_RfromT` | `cnct_search_scat(scat_data, idx, medium, search_param, "<name>", search_fn=None, out_root=None)` |

Both wrappers write the same three files and return the search result:

- `<name>.pkl`: the search result (load it with `pickle`, or plot it with `DASplt.plot_cnct_search`)
- `<name>_data.pkl`: the run's own inputs (search function name, idx, medium, search_param,
  runtime, timestamp); the large data dict is not saved
- `<name>_data.json`: the same inputs as readable text, plus `idx_used` / `idx_dropped`

`out_root` changes the parent folder (default `./inv_results`).

To run either wrapper **in the background** with the notebook's own variables, see section 3
(`submit_detached`).

### Grid options (both searches)

| `grid` | Axes of the map (y, x) | Node values |
|---|---|---|
| `"ratio_ct"` | cn/ct, ct | cn = ratio·ct |
| `"ratio_cn"` | cn/ct, cn | ct = cn/ratio |
| `"non_ratio"` | cn, ct | – |

`"ratio_cn"` shows directly how well cn is constrained: cn is on the x-axis of the maps. Its result
has the axes `cn_ratio` and `cn`, the 2-D `ct_grid` (= cn/ratio at each node), and the actual
`cn_sum_best` / `ct_sum_best` (`grid` = `"ratio_cn"`). The old name `"cn_ctratio"` now raises an
error (use `"ratio_ct"`); saved runs that used it still load and plot. `DASplt.plot_cnct_search`
reads the grid type from the result and labels the axes accordingly.

---

## 1) `cnct_searchR_obs`: TIME domain, reflection modeled from the STF

```python
out = DASrefmod.cnct_searchR_obs(ref_data, inc_data, idx_array, medium, search_param, true_cnct=None)
```

### Inputs

| Input | Description |
|---|---|
| `ref_data` | dict from `data/ref_stf_data_V2.pkl['ref_data']`, updated with `Gr`, `d_factor`, `freqs`, `omega` (`geometric_factor_ref`) and the noise weights (`add_noise_weights`) |
| `inc_data` | `(dir_data['stf_trace'], gd['Gd'])`: recorded direct-S trace dict (E_D) and `Gd = geometric_factor_dir(dir_data, gl, vs, alpha)['Gd']` |
| `idx_array` | trace indices to fit, e.g. `groups[k]` from `trace_grouping(ref_data, snr_min, medium, crit_tol, user_idx_drop)` |
| `medium` | `(vp, vs, rho, wave)`, e.g. `(4291.584, 2407.92, 2600, "SS")` |
| `true_cnct` | `(cn, ct)` or `None`: synthetic tests only (true node added to the axes) |

### `search_param`

```python
search_param = ("constrained",                 # search_mode
                "ratio_ct",                    # grid
                0.06,                          # eps
                ((0.01, 3.5), (1e-14, 1e-8)),  # search_space
                (50, 50),                      # nsearch
                "signed",                      # cc
                (0.2, 3.0),                    # constraint: (s_min, s_max) or None
                None,                          # trace_opts: weights/selection or None
                35,                            # max_lag
                None)                          # Q_opts: None = no Q (1/Q = 0)
```

The first seven entries are required; the last three can be dropped (→ weights 1, no lag
limit, no Q). To give a later entry, write the earlier ones before it (`None` if unused).

| Parameter | Options |
|---|---|
| `search_mode` | `"full"` (global only) or `"constrained"` (global + s_hat-constrained) |
| `grid` | `"ratio_ct"` → grid over (cn/ct, ct), cn = ratio·ct (formerly `"cn_ctratio"`); `"ratio_cn"` → grid over (cn/ct, cn), ct = cn/ratio; `"non_ratio"` → grid over (cn, ct) |
| `eps` | Gd stabilization fraction, eps_val = eps·max\|Gd\| (e.g. 0.06) |
| `search_space` | `((p_min, p_max), (q_min, q_max))`: p = cn/ct (`"ratio_ct"`, `"ratio_cn"`) or cn (`"non_ratio"`); q = ct (`"ratio_ct"`, `"non_ratio"`) or cn (`"ratio_cn"`) [m/Pa], e.g. `((0.01, 3.5), (1e-14, 1e-8))` |
| `nsearch` | `(n_p, n_q)` log-spaced samples of the two axes, e.g. `(50, 50)` |
| `cc` | `"signed"` or `"abs"`: lag-picking rule of the cross-correlation alignment (`align_by_window_cc`, inside `ref_data['time_window']`) |
| `constraint` | `(s_min, s_max)` for `"constrained"`, e.g. `(0.2, 3.0)`; `None` for `"full"` |
| `trace_opts` | `None` (all weights 1) or dict: `"weight"`: precomputed weight key, e.g. `"w_noise"`; `"select"`: `(key, min)`, e.g. `("snr_noise", 7.0)` drops traces below min. Weights are normalized to mean 1 |
| `max_lag` | int or `None`: max \|lag\| [samples] of the alignment, e.g. 35 |
| `Q_opts` | `None` (no Q), or `{"Q": [20, 30, ..., 120, np.inf], "rd": dir_data["mean_dist_source"]}` |

With `Q_opts`, the forward model is `R_batch_InvQ` (a = 1, beta = 1/Q), and every (cn/ct, ct)
node is fitted at every Q. The maps are taken at the lowest-cost Q per node, and the output adds
`Q_hat` / `beta_hat` maps and the cubes `Csp_sum_Q`, `s_hat_Q` (n_Q, n_p, n_ct). In constrained
mode, the lowest-cost Q with s_hat inside the range is used. Runtime is about n_Q × the no-Q run.

### Model and cost

- Forward model `R_wave_v2`, per trace i (S = true source spectrum, E_D = recorded direct S):
  - Direct and reflected arrivals: `E_D = (ω²/vs) · S · Gd`, `E_R,i = (ω²/vs) · S · Gr_i · R(ω; θ_i, cn, ct)`
  - Source removed with the direct wave, `(ω²/vs) · S = E_D / Gd` (the ω²/vs factor cancels: both legs are S):
    `E_R,i = Gr_i · R(ω; θ_i, cn, ct) · E_D / Gd`
  - Stabilized: `1/Gd → conj(Gd) / (|Gd|² + eps_val²)`, `eps_val = eps · max|Gd|`, so
    `H_i = Gr_i · E_D · R(ω; θ_i, cn, ct) · conj(Gd) / (|Gd|² + eps_val²)`
  - E_D is the observed direct-S trace (`stf_trace`, the `S` / `S_omega` of the code); only `Gr/Gd`
    (fiber projection, gauge-length sinc, spreading) and R are modeled
- `Csp_i = 0.5 · w_i · Σ_t (d_i − s_hat·h_i)²`
- `s_hat = Σ_i w_i d_i·h_i / Σ_i w_i h_i·h_i` (one scalar per node)
- Node cost `Csp_sum = Σ_i Csp_i`; the best node minimizes `Csp_sum`

### Output

- `search_mode "full"` → `out_full` (dict)
- `search_mode "constrained"` → `(out_full, out_constraint)`; `out_constraint = constrain_searchR(out_full, constraint)`
- Maps (n_p, n_ct): `Csp_sum` (cost), `cost_norm` (= Csp_sum / data_energy), `s_hat`
- Best node: `ratio_sum_best` / `ct_sum_best` (ratio grid) or `cn_sum_best` / `ct_sum_best`
- Also: `Csp` (n_tr, n_p, n_ct), `lag_samples`, `cc_max` (n_tr, n_p, n_ct), `data_energy`,
  `idx`, `idx_input`, `idx_dropped`, `trace_w`, `trace_opts`, `grid`, `search_mode`

### Save wrapper: `cnct_search`

```python
from cnct_search import cnct_search
out = cnct_search(ref_data, inc_data, idx, medium, search_param, DASrefmod.cnct_searchR_obs, "<name>")
```

Same inputs as `cnct_searchR_obs`, plus `search_fn` (the search function) and the run name.
Returns what `cnct_searchR_obs` returns (`out_full`, or `(out_full, out_constraint)` in
constrained mode) and saves it in `inv_results/<name>/`.

### Example (cons7)

```python
search_param = ("constrained", "ratio_ct", 0.06, ((0.01, 3.5), (1e-14, 1e-8)),
                (50, 50), "signed", (0.2, 3.0),
                {"weight": "w_noise", "select": ("snr_noise", 7.0)}, 35)
out_full, out_cons = DASrefmod.cnct_searchR_obs(ref_data, inc_data, idx, medium, search_param)
```

---

## 2) `cnct_search_RfromT`: TIME domain, reflection predicted from the transmission

```python
out = DASmod.cnct_search_RfromT(scat_data, idx, medium, search_param)
```

At every (cn, ct) node, the reflected trace of each pair is predicted from its **observed
transmitted trace** (forward model `RfromT`) and compared with the observed reflected trace.
There is no source wavelet and no amplitude scalar: the source and the source-to-fracture leg
are common to both legs, so the reflected amplitude is predicted in absolute terms.

### Inputs

| Input | Description |
|---|---|
| `scat_data` | dict from `data/scattering_data.pkl` (`DASscatData.ipynb`), prepared as below |
| `idx` | pair indices to fit (rows of the per-pair arrays) |
| `medium` | `(vp, vs, rho, wave)`, e.g. `(4291.584, 2407.92, 2600, "SS")` |

Preparation (`DASscatInv.ipynb`):

```python
scat_data = DASmod.fft_traces_scat(scat_data)                    # .fft of both trace sets + 'freqs'
g = DASmod.geometric_factor(scat_data, gl, vs, alpha=0.5)        # Gr, Gt (distances dist_source_rr / _tr)
scat_data.update({"Gr": g["Gr"], "Gt": g["Gt"], "omega": g["omega"]})
DASproc.lowpass_tukey_zerophase_scat(scat_data, "transmitted_traces", f_end=320, alpha=0.6)
```

The low-pass on the transmitted traces is needed: without it, `RfromT` boosts the 350–565 Hz
band (G_T sinc notch, R ∝ f) and the modeled reflection rings (`tests/RfromT/ringing_notes.md`).
Choice of `f_end` / `alpha` (flat up to `f_end·(1−alpha)`): a compromise between keeping the
waveform amplitude (the cost uses the absolute amplitude) and removing pair-specific
high-frequency bumps of the transmitted spectra (e.g. pair 25, ~200 Hz). 320 / 0.6 keeps ~95% of
the peak amplitude; 220 / 0.3 removes the bumps but costs ~10%
(`tests/RfromT/test_two_step_lowpass.py`, `test_plot_transmission_lowpass.py <f_end> <alpha>`).
The filter is not stored in the run files: put it in the run name (e.g. `..._lp320-0.6`).

### `search_param`

```python
search_param = ("full",                        # search_mode
                "ratio_ct",                    # grid
                0.06,                          # eps
                ((0.01, 3.5), (1e-12, 1e-8)),  # search_space
                (50, 50),                      # nsearch
                "signed",                      # cc
                None,                          # constraint: must be None
                None,                          # trace_opts: weights/selection or None
                35)                            # max_lag
```

The first six entries are required; the last three can be dropped (→ no constraint, weights 1,
no lag limit). To give a later entry, write the earlier ones before it (`None` if unused).

| Parameter | Options |
|---|---|
| `search_mode` | `"full"` only (unconstrained grid). `"constrained"` raises `NotImplementedError` |
| `grid` | `"ratio_ct"` → grid over (cn/ct, ct), cn = ratio·ct (formerly `"cn_ctratio"`); `"ratio_cn"` → grid over (cn/ct, cn), ct = cn/ratio; `"non_ratio"` → grid over (cn, ct) |
| `eps` | stabilization fraction of `RfromT`, eta = eps·max\|F\|, F = G_T·T (e.g. 0.06) |
| `search_space` | `((p_min, p_max), (q_min, q_max))`: p = cn/ct (`"ratio_ct"`, `"ratio_cn"`) or cn (`"non_ratio"`); q = ct (`"ratio_ct"`, `"non_ratio"`) or cn (`"ratio_cn"`) [m/Pa] |
| `nsearch` | `(n_p, n_q)` log-spaced samples of the two axes, e.g. `(50, 50)`; with `"ratio_cn"` a finer cn axis, e.g. `(50, 100)` over cn 1e-11–1e-8 |
| `cc` | `"signed"` or `"abs"`: lag rule of `align_by_window_cc` (model aligned to the data inside `scat_data['time_window']`) |
| `constraint` | must be `None` (not implemented yet) |
| `trace_opts` | `None` (default: all pairs, weights 1) or dict with `"weight"` and/or `"select"`, see below |
| `max_lag` | int or `None`: max \|lag\| [samples] of the alignment, e.g. 35 |

#### `trace_opts`

`None` (default): all pairs are used, with weight 1. Otherwise a dict with up to two entries:

```python
trace_opts = {"weight": "noise_cap",       # "noise" or "noise_cap"
              "select": ("snr", 7.0)}      # drop pairs with SNR < 7
```

- **`"weight"`**: each pair is weighted by its combined noise, w_i = 1 / (σ_R² + σ_T²), where
  σ_R² and σ_T² are the pre-event noise powers of the pair's reflection and transmission
  channels (`noise_ref`, `noise_tr`).
  - `"noise"`: σ_R² and σ_T² as measured.
  - `"noise_cap"`: σ_R² and σ_T² each capped at its own median over all pairs, because the
    pre-event window of some channels contains transient signal.
- **`"select": ("snr", min)`**: a pair is dropped when the SNR of either of its traces
  (reflection or transmission) is below `min`. The SNR uses the same noise as the weight
  (capped with `"noise_cap"`, as measured otherwise).

Weights are rescaled to an average of 1 over the kept pairs. `scat_data` is not modified.

### Model and cost

- Forward model `RfromT`, per pair i and node (cn, ct):
  - `F = G_T · T`, `eta = eps · max|F|`
  - `E_R = E_T · G_R · R · conj(F) / (|F|² + eta²)`
  - where E_T is the observed transmitted spectrum and R, T are from `RPS_coef` at θ_i
- `Csp_i = 0.5 · w_i · Σ_t (d_i − h_i)²`, with h_i the modeled reflection aligned to d_i; no s_hat
- Node cost `Csp_sum = Σ_i Csp_i`; the best node minimizes `Csp_sum`
- `s_hat = Σ_i w_i d_i·h_i / Σ_i w_i h_i·h_i` is computed at every node as a **diagnostic only**
  (≈ 1 where the predicted amplitude matches the data)

### Output

`out` (dict), the same fields as the full pack of `cnct_searchR_obs`:

- Maps (n_p, n_ct): `Csp_sum`, `cost_norm` (= Csp_sum / data_energy), `s_hat` (diagnostic)
- Best node (global, unconstrained): `ratio_sum_best` / `ct_sum_best` / `cn_sum_best` (ratio grid),
  or `cn_sum_best` / `ct_sum_best` (cn, ct grid)
- Per pair (= per reflection point, `scat_data["ref_points"][out["idx"]]`): `Csp`
  (n_tr, n_p, n_q, each pair's cost at every node), its own best node `cn_best`, `ct_best`,
  `ratio_best`, `Csp_min`, `sub100` (100 best nodes), and `lag_samples`, `cc_max`
- Bookkeeping: `idx`, `idx_input`, `idx_dropped`, `trace_w`, `trace_opts`, `snr_pairs` (pair SNR
  of `idx`, `None` without `trace_opts`), `data_energy`, `search_mode`, `eps`,
  `cost` (= `"0.5 w (d - h)^2, no s_hat"`)

Runtime: about 2.8 ms per pair and node (35 pairs × 50 × 50 ≈ 4 min; × 100 cn ≈ 8 min).

**Groups after the search (no re-run):** the node cost is the sum over pairs, so any group's
cost map is `out["Csp"][rows].sum(0)` (rows = positions in `out["idx"]`), e.g. by incidence
angle; its best node is the argmin of that map (`tests/RfromT/test_group_costs.py`).

**Plots / picks of a saved run** (`DAS_plots.py`):
`plot_cnct_search(pkl, which="full", output="cost" | "cost_norm" | "cost_rel" | "s_hat")`,
`pick_ratiosearch(pkl, which="full", output="cost")` (also `"ratio_cn"` runs, x = cn),
`plot_waveform_picks_scat(run_dir, scat_data)` (data vs `RfromT` at each pick; `scat_data`
prepared exactly as for the run).

### Waveform fitness of the best fits: `waveform_fitness`

```python
fit = DASmod.waveform_fitness(run_dir, scat_data, groups=None, medium=None, search_param=None)
```

Measures how well the best-fit modeled reflections match the data, per group of pairs, from a
saved `cnct_search_scat` run (no re-run of the search). For each group, the best node is the
argmin of `out["Csp"][rows].sum(0)` (the same node as `extracting_search`). At that node the
modeled reflection h_i of every pair is rebuilt with `RfromT` and aligned to the data d_i with
`align_by_window_cc`, using the run's own `cc` and `max_lag`, so 0.5·w_i·‖d_i − h_i‖² equals the
saved `Csp`. The comparison uses the whole trace, the same samples as the cost.

| Input | Description |
|---|---|
| `run_dir` | run folder `<out_root>/<name>/` with `<name>.pkl` and `<name>_data.pkl` |
| `scat_data` | prepared **exactly** as for the run (`fft_traces_scat`, `Gr`, `Gt`, `omega`, the same low-pass of the transmitted traces) |
| `groups` | list of pair-index arrays (as in `out["idx"]`), e.g. `optimal_angle_groups(run_dir, scat_data)["K"][4]["groups"]`; `None` = one group with all the run's pairs (global best node) |
| `medium`, `search_param` | `None` = the run's own values (from `<name>_data.pkl`). Overriding them changes the model or alignment, so the result no longer matches the search |

Per pair i (d = observed reflected trace, h = best-fit modeled trace, aligned):

| Measure | Formula | Meaning |
|---|---|---|
| `rho` | d·h / (‖d‖ ‖h‖) | shape match (normalized cross-correlation at the aligned lag); amplitude does not affect it; 1 = same shape |
| `amp` | ‖h‖ / ‖d‖ | model / data RMS amplitude; 1 = same amplitude, 0.5 = model at half the data |
| `S` | 2 d·h / (‖d‖² + ‖h‖²) = rho · 2·amp / (1 + amp²) | shape **and** amplitude in one number (Lin's concordance correlation coefficient for zero-mean traces); 1 = same shape and amplitude |

- `S` is the cross-correlation divided by the **average** energy of the two traces, while
  `rho` divides by their geometric mean. So S ≤ rho, with S = rho only when amp = 1.
- Under- and over-prediction by the same factor give the same S. The amplitude factor falls
  slowly near amp = 1: 0.98 at amp = 0.8, 0.8 at amp = 0.5.
- `rho` and `amp` alone set the normalized L2 cost: ‖d − h‖² / ‖d‖² = 1 − 2·amp·rho + amp².
  For a given rho this is lowest at amp = rho, so the L2 cost barely penalizes a model that
  under-predicts the amplitude when the shape is imperfect. S does penalize it
  (`tests/RfromT/group_cost_check_notes.md`).

#### Output

`fit` (dict):

- `fit["run"]`: run name
- `fit["groups"]`: list, one dict per group (in the order of `groups`):

| Key | Shape | Content |
|---|---|---|
| `pairs` | (n,) | pair indices of the group |
| `angle` | (min, max) | incidence-angle range [deg] |
| `cn`, `ct`, `ratio` | float | group best node (ratio = cn/ct) |
| `cost_norm` | float | group cost at its best node, Σ Csp_i / Σ 0.5·w_i·‖d_i‖² (as `optimal_angle_groups` / `extracting_search`) |
| `rho`, `amp`, `S` | (n,) | per-pair measures above |
| `E` | (n,) | ‖d_i‖² + ‖h_i‖² (energy of each pair, used by `S_pool`) |
| `misfit2` | (n,) | ‖d_i − h_i‖² (unweighted L2 misfit) |
| `w` | (n,) | trace weights of the run |
| `S_mean` | float | mean of S_i: every pair counts equally. **Use this to compare match quality between groups** |
| `S_pool` | float | 2 Σ d_i·h_i / Σ (‖d_i‖² + ‖h_i‖²): the group treated as one joined trace = energy-weighted mean of S_i; strong traces dominate, as in `cost_norm` |
| `rho_mean`, `amp_mean` | float | mean of rho_i and amp_i: which part (shape or amplitude) lowers S |

Example (`weighted_ratio_cn_snr10`, K = 4; `tests/RfromT/test_waveform_fitness.py`):

```python
groups = DASmod.optimal_angle_groups(run_dir, scat_data)["K"][4]["groups"]
fit = DASmod.waveform_fitness(run_dir, scat_data, groups)
```

| group | n | angle (deg) | S_mean | S_pool | rho_mean | amp_mean | cost_norm |
|---|---|---|---|---|---|---|---|
| 0 | 6 | 45.1–43.3 | 0.81 | 0.81 | 0.82 | 0.87 | 0.33 |
| 1 | 7 | 43.0–40.9 | 0.67 | 0.68 | 0.70 | 0.74 | 0.50 |
| 2 | 15 | 40.6–36.3 | 0.64 | 0.67 | 0.70 | 0.68 | 0.46 |
| 3 | 4 | 36.0–35.1 | 0.57 | 0.58 | 0.66 | 0.57 | 0.56 |

Group 2 has a lower cost_norm than group 1 but a lower S_mean: its weaker pairs (19–27) fit
poorly, and the pooled measures (cost_norm, S_pool) give them less weight.

### Save wrapper: `cnct_search_scat`

```python
from cnct_search import cnct_search_scat
out = cnct_search_scat(scat_data, idx, medium, search_param, "<name>")
```

Same inputs as `cnct_search_RfromT`, plus the run name. `search_fn` defaults to
`DASmod.cnct_search_RfromT`. Returns the search result (`out`) and saves it in
`inv_results/<name>/`.

### Example

```python
medium = (vp, vs, rho, wave)
trace_opts = {"weight": "noise_cap", "select": ("snr", 7.0)}
search_param = ("full", "ratio_ct", 0.06, ((0.01, 3.5), (1e-12, 1e-8)), (50, 50), "signed",
                None, trace_opts, 35)
out = DASmod.cnct_search_RfromT(scat_data, idx, medium, search_param)
```

---

## 3) `submit_detached`: run a search in the background

```python
from cnct_search import cnct_search, cnct_search_scat, submit_detached
```

Runs either save wrapper **with the notebook's own variables** in a separate process (the
operating system puts it on a free core). It returns immediately, the notebook stays free, and
the job keeps running after the kernel or VS Code is closed. The call is the wrapper's call
with `submit_detached(<wrapper>, ...)` in front:

```python
# reflection / transmission (DASscatInv)
job = submit_detached(cnct_search_scat, scat_data, idx_range, medium, search_param, "<name>",
                      out_root=path)

# reflection only (DASrefInv)
job = submit_detached(cnct_search, ref_data, inc_data, idx, medium, search_param,
                      DASrefmod.cnct_searchR_obs, "<name>", out_root=path)
```

Optional: `job_name` (default: the run name), `job_dir` (default: the run folder).

### What goes where

Everything goes to the run folder `<out_root>/<name>/` (`out_root` default `./inv_results`):

| File | Content |
|---|---|
| `<name>.pkl`, `<name>_data.pkl`, `<name>_data.json` | search result and run parameters (as a direct wrapper call) |
| `<name>.log` | progress, the wrapper's "done ... saved" line, then "job finished" or "job FAILED" with the error |
| `<name>_job.pkl` | the pickled inputs (wrapper + arguments, ~140 MB with `scat_data`); deleted after a successful run, kept on failure |

`job` is a small dict (not the results): `job["pid"]` (process id), `job["log"]` (log path),
`job["job"]` (job-file path), `job["run_dir"]` (run folder).

How it works: the inputs are pickled to the job file, and a new process in its own session runs
`cnct_search.py` as a script on it (`_run_job`), which calls the wrapper and deletes the job file.

### Checking jobs

In the notebook:

```python
print(job["log"].read_text())                  # last line "job finished ..." = done
import os; os.path.exists(f"/proc/{job['pid']}")   # True while running
```

In a terminal (repo root):

```bash
ps -u $USER -o pid,etime,args | grep "cnct_search.py" | grep -v grep   # running jobs
tail -n 1 inv_results/RfromT/*/*.log                                   # last line of each run's log
ls inv_results/RfromT/*/*_job.pkl                                      # unfinished jobs
```

Several jobs: call `submit_detached` once per run, each with its own run name (one core per
job). A running job can be stopped with `os.kill(job["pid"], 15)`.
