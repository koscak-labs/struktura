"""struktura Guard on ESA-ADB Mission 1 channels 41-46 (84_months split), and its time-shift controls.

Guard: struktura.Guard (AutoPilot, default configuration, cooldown 50) calibrated on the first block of
20160 train rows (7 days at 30 s) in which all six channels are labelled nominal. Every test row is
pushed in order; the prediction for a channel is 1 at every row where Guard raises an `alarm` or
`rolled_back` event on that channel. Test labels are not read.

Control: the whole prediction matrix rolled in time by one offset, uniform in [10%, 90%] of the
length and the same for every channel. It keeps every alarm, its length and its timing across
channels, and moves only where the alarms fall relative to the labels.

Usage: python guard.py <work_dir> guard            -> <work_dir>/preds/guard.npz (+ .meta.json)
       python guard.py <work_dir> shift <seed>     -> <work_dir>/preds/shift_s<seed>.npz
"""
import json
import os
import sys
import time

import numpy as np

N_CAL = 20160
COOLDOWN = 50


def load(work, split):
    d = np.load(os.path.join(work, "data", f"84_months.{split}.npz"))
    return d["values"], d["labels"]


def first_clean_block(labels, n):
    clean = (labels == 0).all(axis=1)
    run = 0
    for i, c in enumerate(clean):
        run = run + 1 if c else 0
        if run == n:
            return i - n + 1
    raise RuntimeError("no clean block")


def iter_rows(a, chunk=100_000):
    for i in range(0, len(a), chunk):
        yield from a[i:i + chunk].astype(np.float64).tolist()


def guard_run(work):
    import struktura
    v_tr, l_tr = load(work, "train")
    s = first_clean_block(l_tr, N_CAL)
    cal = [v_tr[s:s + N_CAL, k].astype(np.float64).tolist() for k in range(v_tr.shape[1])]
    del v_tr, l_tr
    v, _ = load(work, "test")
    g = struktura.Guard(cal, cooldown=COOLDOWN)
    pred = np.zeros(v.shape, dtype=np.uint8)
    kinds, legs = {}, {}
    t0 = time.time()
    for tick, row in enumerate(iter_rows(v)):
        for e in g.push(row):
            k = e["kind"]
            kinds[k] = kinds.get(k, 0) + 1
            if k in ("alarm", "rolled_back") and e.get("channel") is not None:
                pred[tick, e["channel"]] = 1
                if "leg" in e:
                    legs[e["leg"]] = legs.get(e["leg"], 0) + 1
    meta = {"struktura_version": struktura.__version__, "n_cal": N_CAL, "cooldown": COOLDOWN,
            "cal_row_start": int(s), "test_rows": int(len(v)), "event_kinds": kinds, "alarm_legs": legs,
            "alarm_ticks_per_channel": pred.sum(axis=0).tolist(), "seconds": round(time.time() - t0, 1)}
    return pred, meta


def shift_run(work, seed):
    ref = np.load(os.path.join(work, "preds", "guard.npz"))["pred"]
    n = ref.shape[0]
    off = int(np.random.default_rng(seed).integers(int(0.1 * n), int(0.9 * n) + 1))
    return np.roll(ref, off, axis=0), {"seed": seed, "offset": off}


def main():
    work, kind = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(work, "preds"), exist_ok=True)
    if kind == "guard":
        pred, meta = guard_run(work)
        name = "guard"
    else:
        seed = int(sys.argv[3])
        pred, meta = shift_run(work, seed)
        name = f"shift_s{seed}"
    np.savez_compressed(os.path.join(work, "preds", f"{name}.npz"), pred=pred)
    with open(os.path.join(work, "preds", f"{name}.meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(name, json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
