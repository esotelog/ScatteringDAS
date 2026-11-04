"""
Purpose
-------
Generalizes this module's earlier hardcoded single-run script (which only
ran cnct_ratiosearch_Rtime_obs on one fixed set of parameters) into a
reusable function, `cnct_search`, callable from a notebook or another
script. It takes already-prepared search inputs (ref_data, inc_data,
idx_array, medium, search_param -- the exact calling convention shared by
BOTH DASref_modeling.cnct_search_Rtime_Gd_cc_obs and
DASref_modeling.cnct_ratiosearch_Rtime_obs) plus a choice of WHICH of
those two functions to run, and saves the run as a self-contained folder
under ./inv_results/<output_name>/ -- the result pickle and a companion
pickle of the run's own input parameters, so a past run's exact inputs
can be inspected without re-running it.

Tested in tests/test_run_cnct_search.py against real data, for both
search_fn choices, before being placed here.

Input variables (of cnct_search)
------------------------------------
ref_data, inc_data, idx_array, medium, search_param, true_cnct :
    passed straight through to `search_fn` -- see
    cnct_search_Rtime_Gd_cc_obs's / cnct_ratiosearch_Rtime_obs's own
    docstrings in DASref_modeling.py for what each means. NOT re-loaded
    or rebuilt here -- the caller prepares them once (e.g. from
    data/ref_stf_data.pkl, as DASrefInv.ipynb's cells 0-10 do) and can
    reuse them across many cnct_search calls without re-reading that
    pickle each time.
search_fn : callable
    DASrefmod.cnct_search_Rtime_Gd_cc_obs (single-pack, plain (cn, ct)
    unconstrained search) or DASrefmod.cnct_ratiosearch_Rtime_obs
    (two-pack, ratio-parameterized search with an s_hat constraint).
output_name : str
    Base name for the run's folder and files.
out_root : str or Path or None
    Parent directory for the run folder; defaults to
    <repo root>/inv_results.

Output variables (of cnct_search)
--------------------------------------
result : whatever search_fn returns -- a single dict for
    cnct_search_Rtime_Gd_cc_obs, or an (out_full, out_constraint) tuple
    for cnct_ratiosearch_Rtime_obs.
Also written to disk, under ./inv_results/<output_name>/, both pickles:
    <output_name>.pkl : pickled `result`.
    <output_name>_data.pkl : pickled dict of the run's OWN input
        parameters (which search_fn, idx_array, medium, search_param,
        elapsed time, timestamp) -- NOT ref_data/inc_data themselves,
        which are large and shared across many runs rather than
        duplicated per run.
    <output_name>_data.json : the same input parameters as readable JSON
        (arrays -> lists, tuples -> lists), for reviewing a run without
        Python. When the result carries trace bookkeeping (a
        cnct_ratiosearch_Rtime_obs run with trace_opts), it also lists
        idx_used / idx_dropped -- the traces actually fitted / removed.
        The .pkl stays the file code should load (exact types).
"""
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _to_jsonable(obj):
    """
    Purpose: convert a run's input-parameter dict to JSON-serializable types.
    Input: obj -- nested dict/tuple/list of numpy arrays/scalars, str, None.
    Output: same structure with arrays -> lists, numpy scalars -> Python
        scalars, tuples -> lists (JSON has no tuple).
    """
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj



def write_data_json(input_data, result, json_path):
    """
    Purpose: write a run's input parameters as readable JSON (one line per
        field), plus the traces actually fitted/dropped when the result
        records them (a cnct_ratiosearch_Rtime_obs run with trace_opts).
        Used by cnct_search for new runs and by write_run_json for runs
        saved before the JSON existed.
    Input: input_data (dict, the *_data.pkl content); result (the search
        result: a dict, or an (out_full, out_constraint) tuple); json_path
        (str/Path to write).
    Output: None (side effect: writes json_path).
    """
    json_data = dict(input_data)
    first = result[0] if isinstance(result, tuple) else result
    if isinstance(first, dict) and "idx_dropped" in first:
        json_data["idx_used"] = first["idx"]
        json_data["idx_dropped"] = first["idx_dropped"]
    # one line per field (compact values), so long arrays stay readable
    lines = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in _to_jsonable(json_data).items()]
    with open(json_path, "w") as f:
        f.write("{\n" + ",\n".join(lines) + "\n}\n")


def write_run_json(run_dir, overwrite=False):
    """
    Purpose: add <run>_data.json to an EXISTING run folder, built from its
        <run>_data.pkl (and its main <run>.pkl for idx_used/idx_dropped).
    Input: run_dir (str/Path, e.g. ./inv_results/run2_cons3); overwrite
        (bool, replace an existing json).
    Output: Path of the json, or None if the folder has no <run>_data.pkl
        or the json exists and overwrite is False.
    """
    run_dir = Path(run_dir)
    name = run_dir.name
    data_path = run_dir / f"{name}_data.pkl"
    json_path = run_dir / f"{name}_data.json"
    if not data_path.exists() or (json_path.exists() and not overwrite):
        return None
    with open(data_path, "rb") as f:
        input_data = pickle.load(f)
    result = None
    main_pkl = run_dir / f"{name}.pkl"
    if main_pkl.exists():
        with open(main_pkl, "rb") as f:
            result = pickle.load(f)
    write_data_json(input_data, result, json_path)
    return json_path


def cnct_search(ref_data, inc_data, idx_array, medium, search_param,
                     search_fn, output_name, true_cnct=None, out_root=None):
    if out_root is None:
        out_root = ROOT / "inv_results"
    out_root = Path(out_root)
    run_dir = out_root / output_name
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = search_fn(ref_data, inc_data, idx_array, medium, search_param, true_cnct=true_cnct)
    elapsed = time.time() - t0

    # the run's own parameters, NOT ref_data/inc_data (large, shared
    # across runs) -- enough to know what produced <output_name>.pkl
    input_data = {
        "search_fn": search_fn.__name__,
        "idx_array": np.asarray(idx_array),
        "medium": medium,
        "search_param": search_param,
        "elapsed_s": elapsed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    pkl_path = run_dir / f"{output_name}.pkl"
    data_path = run_dir / f"{output_name}_data.pkl"
    json_path = run_dir / f"{output_name}_data.json"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    with open(data_path, "wb") as f:
        pickle.dump(input_data, f)

    write_data_json(input_data, result, json_path)

    print(f"[{search_fn.__name__}] done in {elapsed:.1f}s -> "
          f"saved {pkl_path.name}, {data_path.name}, {json_path.name} in {run_dir}", flush=True)
    return result


def cnct_search_scat(scat_data, idx_array, medium, search_param, output_name,
                     search_fn=None, out_root=None):
    """
    Purpose
    -------
    Save wrapper for the reflection/transmission searches, the counterpart of
    cnct_search for functions with the calling convention
        search_fn(scat_data, idx_array, medium, search_param)
    i.e. DAS_modeling.cnct_search_RfromT (no inc_data, no true_cnct). Runs
    the search and saves it exactly like cnct_search, as a self-contained
    folder ./inv_results/<output_name>/ with
        <output_name>.pkl       : pickled result (the search output dict);
        <output_name>_data.pkl  : the run's own input parameters (search_fn
                                  name, idx_array, medium, search_param,
                                  elapsed time, timestamp) -- NOT scat_data,
                                  which is large and shared across runs;
        <output_name>_data.json : the same as readable JSON, plus idx_used /
                                  idx_dropped (pairs fitted / removed by
                                  trace_opts).
    The saved run loads in DAS_plots.plot_cnct_search like any other run.

    Input
    -----
    scat_data : dict, prepared as in DASscatInv.ipynb (see
        DAS_modeling.cnct_search_RfromT).
    idx_array : array-like of int, pair indices to fit.
    medium : (vp, vs, rho, wave).
    search_param : tuple, see DAS_modeling.cnct_search_RfromT.
    output_name : str, base name of the run folder and files.
    search_fn : callable or None
        None (default) -> DAS_modeling.cnct_search_RfromT.
    out_root : str or Path or None
        Parent directory of the run folder; None -> <repo root>/inv_results.

    Output
    ------
    result : the search output (dict), also written to disk as above.
    """
    if search_fn is None:                                   # default search (imported here: no
        import DAS_model_inversion as DASmi                 # DAS_* dependency for cnct_search)
        search_fn = DASmi.cnct_search_RfromT
    if out_root is None:
        out_root = ROOT / "inv_results"
    run_dir = Path(out_root) / output_name
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = search_fn(scat_data, idx_array, medium, search_param)
    elapsed = time.time() - t0

    # the run's own parameters (not scat_data) -- enough to know what produced <output_name>.pkl
    input_data = {
        "search_fn": search_fn.__name__,
        "idx_array": np.asarray(idx_array),
        "medium": medium,
        "search_param": search_param,
        "elapsed_s": elapsed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    pkl_path = run_dir / f"{output_name}.pkl"
    data_path = run_dir / f"{output_name}_data.pkl"
    json_path = run_dir / f"{output_name}_data.json"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    with open(data_path, "wb") as f:
        pickle.dump(input_data, f)
    write_data_json(input_data, result, json_path)          # same JSON layout as cnct_search

    print(f"[{search_fn.__name__}] done in {elapsed:.1f}s -> "
          f"saved {pkl_path.name}, {data_path.name}, {json_path.name} in {run_dir}", flush=True)
    return result


def submit_detached(wrapper, *args, job_name=None, job_dir=None, **kwargs):
    """
    Purpose
    -------
    Run a save wrapper (cnct_search or cnct_search_scat) with the CALLER's own
    variables (e.g. the notebook's scat_data, search_param) in a separate,
    detached process that keeps running after the notebook kernel / VS Code
    is closed. The wrapper and its arguments are pickled to a job file; a new
    process in its own session (like `setsid nohup`) runs this module as a
    script on it (`python cnct_search.py <job file>`, see _run_job), which
    calls wrapper(*args, **kwargs) -- the run is saved exactly as a direct
    call would save it -- and deletes the job file when it ends (also on failure).

    Input
    -----
    wrapper : callable, cnct_search or cnct_search_scat (any module-level
        function importable from py_files works).
    *args, **kwargs : passed unchanged to wrapper, e.g.
        submit_detached(cnct_search_scat, scat_data, idx, medium,
                        search_param, "run_name", out_root=path)
    job_name : str or None
        Name of the job/log files; None -> the run name (wrapper's output_name).
    job_dir : str or Path or None
        Folder for the job file and log; None -> the run folder itself,
        <out_root>/<output_name> (out_root None -> <repo root>/inv_results),
        so the log sits next to the run's results.

    Output
    ------
    dict: 'pid' (process id), 'log' (Path of the log file
        <run folder>/<output_name>.log: progress and the wrapper's
        "done ... saved" line, or the error), 'job' (Path of the pickled job
        file <run folder>/<output_name>_job.pkl, always removed when the
        job ends, finished or failed), 'run_dir' (the run folder).
    """
    import inspect
    import subprocess
    import sys

    # run folder from the wrapper's own arguments: <out_root>/<output_name>
    bound = inspect.signature(wrapper).bind(*args, **kwargs)
    output_name = bound.arguments.get("output_name")
    out_root = bound.arguments.get("out_root")
    run_dir = (Path(out_root) if out_root is not None else ROOT / "inv_results") / str(output_name)
    job_dir = Path(job_dir) if job_dir is not None else run_dir
    job_dir.mkdir(parents=True, exist_ok=True)
    if job_name is None:
        job_name = str(output_name) if output_name is not None else \
            f"{wrapper.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_path = job_dir / f"{job_name}_job.pkl"           # pickled inputs (removed when done)
    log_path = job_dir / f"{job_name}.log"                # progress / done / error

    with open(job_path, "wb") as f:                         # wrapper (by reference) + its inputs
        pickle.dump({"wrapper": wrapper, "args": args, "kwargs": kwargs}, f)

    with open(log_path, "w") as log:                        # detached: own session, no stdin
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), str(job_path)],   # this module as a script
            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    print(f"submitted {job_name} (pid {proc.pid}); log: {log_path}", flush=True)
    return {"pid": proc.pid, "log": log_path, "job": job_path, "run_dir": run_dir}


def _run_job(job_path):
    """
    Purpose: run one pickled job written by submit_detached (called when this
        module is executed as a script, in the detached process): load the job
        file, call wrapper(*args, **kwargs) (cnct_search / cnct_search_scat,
        which run the search and save it in the run folder), then delete the
        job file. Progress and errors go to the log file opened by
        submit_detached; the job file is ALWAYS deleted, also on failure.
    Input: job_path (str/Path, <run folder>/<output_name>_job.pkl).
    Output: exit code (0 = finished, 1 = failed).
    """
    import traceback
    job_path = Path(job_path)
    t0 = time.time()
    print(f"job {job_path.name} started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    try:
        with open(job_path, "rb") as f:
            job = pickle.load(f)
        job["wrapper"](*job["args"], **job["kwargs"])       # runs and saves the search
    except Exception:
        traceback.print_exc()
        print(f"job FAILED after {time.time() - t0:.1f} s", flush=True)
        return 1
    else:
        print(f"job finished in {time.time() - t0:.1f} s", flush=True)
        return 0
    finally:
        job_path.unlink(missing_ok=True)                    # job file always removed (success or failure)
        print("job file removed", flush=True)


if __name__ == "__main__":
    # detached job started by submit_detached: python cnct_search.py <job file>
    import sys
    sys.path.insert(0, str(ROOT / "py_files"))              # search modules importable
    sys.exit(_run_job(sys.argv[1]))
