"""Score prediction files with the OFFICIAL ESA-ADB metric classes, as timeeval/core/experiments.py
Experiment.evaluate does for this setup (labels_df branch), restricted to channels 41-46.

The metric source files are imported unchanged from an ESA-ADB checkout; only the `timeeval` and
`timeeval.metrics` package __init__ files are bypassed (they pull in docker and dask).
  * labels.csv (time zone dropped), kept if StartTime >= first test timestamp and EndTime <= last,
    the last four columns of anomaly_types.csv merged in, channels 41-46
  * scores -> MinMax per channel (scale_scores) -> astype(uint8)
  * ESAScores(betas=0.5) and ADTQC on the global score (max over channels), ChannelAwareFScore(beta=0.5)
    per channel with the subsystem map from channels.csv; each for {Rare Event, Anomaly} and {Anomaly}
  * the (timestamp, score) series is passed compressed to its change points (first row, every row that
    differs from the previous one, last row), which gives the same events and range as the full series
Each metric runs in its own child process to bound memory.

Usage: ESA_ADB=path/to/ESA-ADB python score.py <work_dir> <pred.npz> [...]   -> <work_dir>/scores/<name>.json
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import types

import numpy as np
import pandas as pd

CHANNELS = ["channel_41", "channel_42", "channel_43", "channel_44", "channel_45", "channel_46"]
SELECTIONS = {"RareEvent+Anomaly": ["Rare Event", "Anomaly"], "Anomaly": ["Anomaly"]}
KEYS = [f"{m}_{s}" for s in SELECTIONS for m in ("ADTQC", "ESAScores", "ChannelAwareFScore")]
BETA = 0.5


def load_truth(src, ts):
    labels_df = pd.read_csv(os.path.join(src, "labels.csv"), parse_dates=["StartTime", "EndTime"])
    labels_df["StartTime"] = labels_df["StartTime"].apply(lambda t: t.tz_localize(None))
    labels_df["EndTime"] = labels_df["EndTime"].apply(lambda t: t.tz_localize(None))
    tmin, tmax = pd.Timestamp(int(ts[0])), pd.Timestamp(int(ts[-1]))
    labels_df = labels_df[(labels_df["StartTime"] >= tmin) & (labels_df["EndTime"] <= tmax)]
    at = pd.read_csv(os.path.join(src, "anomaly_types.csv"))
    cols = at.columns[-4:]
    for col in cols:
        labels_df[col] = ""
    for _, row in at.iterrows():
        labels_df.loc[labels_df["ID"] == row["ID"], cols] = row[cols].values
    ch = pd.read_csv(os.path.join(src, "channels.csv"))
    subsystems = {s: [*v] for s, v in ch.groupby("Subsystem")["Channel"]}
    return labels_df[labels_df["Channel"].isin(CHANNELS)], subsystems


def scale_scores(y):  # Experiment.scale_scores, same logic
    from sklearn.preprocessing import MinMaxScaler
    y = np.asarray(y, dtype=np.float32)
    for i in range(y.shape[-1]):
        mask = np.isinf(y[..., i]) | np.isneginf(y[..., i]) | np.isnan(y[..., i])
        s = y[..., i][~mask]
        if s.size != 0:
            y[..., i][~mask] = MinMaxScaler().fit_transform(s.reshape(-1, 1)).ravel()
    return y


def series(ts, score):
    score = np.asarray(score, dtype=np.uint8)
    keep = np.r_[True, score[1:] != score[:-1]]
    keep[-1] = True
    idx = np.flatnonzero(keep)
    return pd.DataFrame({"Timestamp": pd.to_datetime(ts[idx]), "Score": score[idx]})


def quiet(fn, *a):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return fn(*a)
    except Exception as e:  # the pipeline catches and logs metric exceptions
        return {"EXCEPTION": f"{type(e).__name__}: {e}"}


def one_metric(key, ts, pred):
    """Compute one of KEYS (child process)."""
    esa = os.environ["ESA_ADB"]
    for name, sub in (("timeeval", "timeeval"), ("timeeval.metrics", "timeeval/metrics")):
        mod = types.ModuleType(name)
        mod.__path__ = [os.path.join(esa, sub)]
        sys.modules[name] = mod
    from timeeval.metrics.ESA_ADB_metrics import ESAScores
    from timeeval.metrics.latency_metrics import ADTQC
    from timeeval.metrics.ranking_metrics import ChannelAwareFScore

    y_true, subsystems = load_truth(os.path.join(esa, "data", "ESA-Mission1"), ts)
    y = scale_scores(pred)
    g = series(ts, y.max(axis=1).astype(np.uint8))
    metric, sel_name = key.split("_", 1)
    sel = {"Category": SELECTIONS[sel_name]}
    if metric == "ESAScores":
        return quiet(ESAScores(betas=BETA, select_labels=sel).score, y_true.drop(columns=["Channel"]), g)
    if metric == "ADTQC":
        ytg = y_true.copy()
        ytg["Channel"] = "global"
        return quiet(ADTQC(select_labels=sel).score, ytg, {"global": g})
    yp = {c: series(ts, y[:, k].astype(np.uint8)) for k, c in enumerate(CHANNELS)}
    return quiet(ChannelAwareFScore(beta=BETA, select_labels=sel).score, y_true, yp, subsystems)


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    work = sys.argv[1]
    ts = np.load(os.path.join(work, "data", "84_months.test.npz"))["ts"]
    if os.environ.get("SCORE_ONLY"):  # child: one metric, JSON on stdout
        pred = np.load(sys.argv[2])["pred"]
        print(json.dumps(to_jsonable(one_metric(os.environ["SCORE_ONLY"], ts, pred)), default=str))
        return
    os.makedirs(os.path.join(work, "scores"), exist_ok=True)
    for p in sys.argv[2:]:
        name = os.path.basename(p)[:-4]
        out = {}
        for key in KEYS:
            for attempt in (1, 2):  # one retry: a child can fail to start when the machine is short of memory
                r = subprocess.run([sys.executable, os.path.abspath(__file__), work, p],
                                   env=dict(os.environ, SCORE_ONLY=key), capture_output=True, text=True)
                if r.returncode == 0:
                    break
                print(f"{name} {key}: attempt {attempt} exit {r.returncode}: {r.stderr[-500:]}", flush=True)
                time.sleep(30)
            else:
                sys.exit(f"{name} {key}: failed twice")
            out[key] = json.loads(r.stdout.strip().splitlines()[-1])
            print(name, key, json.dumps(out[key]), flush=True)
        with open(os.path.join(work, "scores", f"{name}.json"), "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
