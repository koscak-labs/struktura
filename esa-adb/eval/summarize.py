"""Guard's official scores against its time-shifted copies.

p = (1 + number of shifts scoring >= Guard) / (1 + number of shifts); with 20 shifts the smallest
possible p is 1/21 = 0.048. ADTQC is undefined (NaN) when nothing is detected; such shifts are left
out of that metric's count.

Usage: python summarize.py <work_dir>
"""
import glob
import json
import math
import os
import sys

METRICS = [("ESAScores", "EW_recall"), ("ESAScores", "EW_F_0.50"), ("ESAScores", "alarming_precision"),
           ("ESAScores", "AFF_F_0.50"), ("ChannelAwareFScore", "channel_F0.50"), ("ADTQC", "Total")]


def value(scores, family, sel, key):
    v = scores.get(f"{family}_{sel}", {}).get(key)
    return None if v is None or math.isnan(float(v)) else float(v)


def main():
    work = sys.argv[1]
    guard = json.load(open(os.path.join(work, "scores", "guard.json")))
    shifts = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(work, "scores", "shift_s*.json")))]
    meta = json.load(open(os.path.join(work, "preds", "guard.meta.json")))
    print(f"struktura {meta['struktura_version']}; guard events {meta['event_kinds']}; "
          f"alarm legs {meta['alarm_legs']}; alarm ticks per channel {meta['alarm_ticks_per_channel']}")
    print(f"{len(shifts)} time-shifted copies")
    print("| labels | metric | guard | shifts: mean | shifts: max | p |")
    print("|---|---|---|---|---|---|")
    for sel in ("Anomaly", "RareEvent+Anomaly"):
        for family, key in METRICS:
            obs = value(guard, family, sel, key)
            cv = [v for v in (value(s, family, sel, key) for s in shifts) if v is not None]
            if obs is None or not cv:
                print(f"| {sel} | {key} | {obs} | | | n/a ({len(cv)} shifts defined) |")
                continue
            p = (1 + sum(v >= obs for v in cv)) / (1 + len(cv))
            print(f"| {sel} | {key} | {obs:.3f} | {sum(cv) / len(cv):.3f} | {max(cv):.3f} | "
                  f"{p:.3f}{'' if len(cv) == len(shifts) else f' ({len(cv)} defined)'} |")


if __name__ == "__main__":
    main()
