"""ESA-ADB Mission 1, channels 41-46: the official preprocessing, restricted to these channels.

Restricted replication of notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py
(kplabs-pl/ESA-ADB, commit aeebcd9). The official script builds one frame with all 76 channels and
11 telecommands over ~7.4M rows per split, which needs far more memory than six channels.

Replicated from the official script:
  * labels.csv parsed with dateutil ignoretz; Category from anomaly_types.csv -> label 1 (Anomaly),
    2 (Rare Event), 3 (other), applied per labels.csv row in file order
  * split: train = index <= split_at, test = index > 2007-01-01
  * 30 s zero-order hold: reindex(date_range(floor(first), ceil(last)), ffill), iloc[0] = first sample
  * "restore annotated samples": for a 30 s bin with >1 raw samples whose last label is nominal and which
    holds an Anomaly/Rare Event sample, the row at bin + 30 s gets the last annotated sample (vectorised)
  * full index = date_range over the first/last resampled timestamp of ALL 76 channels and the 11
    priority>=3 telecommands (utils.find_full_time_range), so every channel file is opened once
  * per channel: reindex to the full index, ffill().bfill() for value and label
Channels 41-46 are not in 4..11, so the official derivative step does not apply to them.

Usage: ESA_ADB=path/to/ESA-ADB python prep.py <out_dir> 84_months 2007-01-01
Writes <out_dir>/84_months.{train,test}.npz: ts (int64 ns), values float32 [n, 6], labels uint8 [n, 6].
"""
import gc
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from dateutil.parser import parse as parse_date

ESA_ADB = os.environ["ESA_ADB"]
sys.path.insert(0, os.path.join(ESA_ADB, "notebooks", "data-prep"))
from utils import AnnotationLabel, encode_telecommands  # noqa: E402  official helpers, unchanged

SRC = os.path.join(ESA_ADB, "data", "ESA-Mission1")
RULE = pd.Timedelta(seconds=30)
SUBSET = ["channel_41", "channel_42", "channel_43", "channel_44", "channel_45", "channel_46"]
TEST_SPLIT = "2007-01-01"
SPLITS = {}


def load_meta():
    labels_df = pd.read_csv(os.path.join(SRC, "labels.csv"), parse_dates=["StartTime", "EndTime"],
                            date_parser=lambda x: parse_date(x, ignoretz=True))
    anomaly_types_df = pd.read_csv(os.path.join(SRC, "anomaly_types.csv"))
    tc = pd.read_csv(os.path.join(SRC, "telecommands.csv"))
    tc = tc.loc[tc["Priority"] >= 3]
    return labels_df, anomaly_types_df, sorted(tc.Telecommand.to_list())


def split_df(df, split_type, split_at):
    if split_type == "train":
        return df[df.index <= parse_date(split_at)].copy()
    return df[df.index > parse_date(TEST_SPLIT)].copy()


def apply_labels(param, param_df, labels_df, anomaly_types_df):
    is_annotated = False
    for _, row in labels_df.iterrows():
        if row["Channel"] == param:
            cat = anomaly_types_df.loc[anomaly_types_df["ID"] == row["ID"]]["Category"].values[0]
            if cat == "Anomaly":
                v = AnnotationLabel.ANOMALY.value
            elif cat == "Rare Event":
                v = AnnotationLabel.RARE_EVENT.value
            else:
                v = AnnotationLabel.GAP.value
            param_df.loc[row["StartTime"]:row["EndTime"], "label"] = v
            is_annotated = True
    return is_annotated


def restore_targets(param_df):
    """Vectorised equivalent of the official restore loop: (target_timestamp, value, label) per bin."""
    idx = param_df.index.values.astype("int64")
    step = RULE.value
    bins = idx - (idx % step)  # floor to 30 s (epoch-aligned; 30 s divides a day)
    lab = param_df["label"].values
    val = param_df["value"].values
    n = len(idx)
    if n == 0:
        return []
    starts = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
    ends = np.r_[starts[1:], n]
    ann = (lab == AnnotationLabel.ANOMALY.value) | (lab == AnnotationLabel.RARE_EVENT.value)
    cs = np.r_[0, np.cumsum(ann)]
    has_ann = (cs[ends] - cs[starts]) > 0
    cand = np.flatnonzero(((ends - starts) > 1) & (lab[ends - 1] == AnnotationLabel.NOMINAL.value) & has_ann)
    out = []
    for g in cand:
        s, e = starts[g], ends[g]
        j = s + np.flatnonzero(ann[s:e])[-1]
        out.append((pd.Timestamp(bins[s]) + RULE, val[j], lab[j]))
    return out


def resample_channel(param_df, is_annotated):
    first = pd.Timestamp(param_df.index[0]).floor(freq=RULE)
    last = pd.Timestamp(param_df.index[-1]).ceil(freq=RULE)
    res = param_df.reindex(pd.date_range(first, last, freq=RULE), method="ffill")
    res.iloc[0] = param_df.iloc[0]
    if is_annotated:
        for ts, v, lab in restore_targets(param_df):
            res.loc[ts] = [v, lab]  # same row assignment as the official script
    return res


def channel_range(param, split_type, labels_df, anomaly_types_df):
    """First/last resampled timestamp of a channel for a split (including the restore edge case)."""
    df = pd.read_pickle(os.path.join(SRC, "channels", f"{param}.zip"))
    df = df.rename(columns={param: "value"})
    df["label"] = np.uint8(0)
    ann = apply_labels(param, df, labels_df, anomaly_types_df)
    df = split_df(df, split_type, SPLITS[split_type])
    if len(df) == 0:
        return None
    first = pd.Timestamp(df.index[0]).floor(freq=RULE)
    last = pd.Timestamp(df.index[-1]).ceil(freq=RULE)
    if ann:
        t = restore_targets(df)
        if t:
            last = max(last, t[-1][0])
    return first, last


def telecommand_range(tcname, split_type):
    df = pd.read_pickle(os.path.join(SRC, "telecommands", f"{tcname}.zip"))
    df["label"] = np.uint8(0)
    df = df.rename(columns={tcname: "value"})
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated()]
    df = encode_telecommands(df, RULE)
    df = split_df(df, split_type, SPLITS[split_type])
    if len(df) == 0:
        return None
    return pd.Timestamp(df.index[0]).floor(freq=RULE), pd.Timestamp(df.index[-1]).ceil(freq=RULE)


def main():
    out_dir, split_name, split_at = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    SPLITS["train"], SPLITS["test"] = split_at, TEST_SPLIT
    labels_df, anomaly_types_df, tcs = load_meta()
    all_params = sorted(os.path.basename(f)[:-4] for f in glob(os.path.join(SRC, "channels", "*.zip")))
    info = {"split_name": split_name, "split_at": split_at, "test_split": TEST_SPLIT,
            "n_channels_scanned": len(all_params), "telecommands": tcs}
    for st in ("train", "test"):
        ranges = []
        for p in all_params:
            ranges.append(channel_range(p, st, labels_df, anomaly_types_df))
            gc.collect()
        ranges += [telecommand_range(t, st) for t in tcs]
        ranges = [r for r in ranges if r]
        start, end = min(r[0] for r in ranges), max(r[1] for r in ranges)
        full_index = pd.date_range(start, end, freq=RULE)
        print(st, "full range", start, end, len(full_index), flush=True)
        vals = np.empty((len(full_index), len(SUBSET)), dtype=np.float32)
        labs = np.empty((len(full_index), len(SUBSET)), dtype=np.uint8)
        for k, p in enumerate(SUBSET):
            df = pd.read_pickle(os.path.join(SRC, "channels", f"{p}.zip"))
            df["label"] = np.uint8(0)
            df = df.rename(columns={p: "value"})
            ann = apply_labels(p, df, labels_df, anomaly_types_df)
            res = resample_channel(split_df(df, st, split_at), ann).reindex(full_index)
            del df
            vals[:, k] = res["value"].ffill().bfill().values.astype(np.float32)
            labs[:, k] = res["label"].ffill().bfill().astype(np.uint8).values
            del res
            gc.collect()
            print(st, p, "done", flush=True)
        fn = os.path.join(out_dir, f"{split_name}.{st}.npz")
        np.savez(fn, ts=full_index.values.astype("int64"), values=vals, labels=labs, channels=np.array(SUBSET))
        info[st] = {"start": str(start), "end": str(end), "rows": int(len(full_index))}
        del vals, labs
        gc.collect()
    with open(os.path.join(out_dir, f"{split_name}.prep_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("DONE", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
