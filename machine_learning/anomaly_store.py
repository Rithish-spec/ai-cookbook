"""Persistent anomaly store.

Rows are identified by a content hash (+ occurrence number for exact duplicates),
so the store survives row reordering, additions and removals in the source file.

    anomaly_store/
      flags.csv      auto-evaluated result per row key (cache)
      overrides.csv  manual decisions: row_key, decision (anomaly|clean), note
      meta.json      rules fingerprint + frozen rule parameters

Each run only evaluates rows the store hasn't seen. Rows that vanished from the
data are pruned from the cache (overrides are kept). Changing anomaly_rules.py
triggers a full re-evaluation; manual overrides always win over the rules.

CLI (row numbers are 0-based positions in the loaded Excel file):
    python anomaly_store.py summary
    python anomaly_store.py flag  12 40 --note "bad entry"   # force anomaly
    python anomaly_store.py allow 7                          # force clean
    python anomaly_store.py reset 7 12                       # drop override
"""
import argparse
import hashlib
import json
import os

import pandas as pd

import anomaly_rules

STORE_DIR = os.path.join(os.path.dirname(__file__), "anomaly_store")
FLAGS_PATH = os.path.join(STORE_DIR, "flags.csv")
OVERRIDES_PATH = os.path.join(STORE_DIR, "overrides.csv")
META_PATH = os.path.join(STORE_DIR, "meta.json")
DUPLICATE_COL = "duplicate_rows"


def row_keys(df):
    """Stable content-hash key per row; repeats get a '#n' occurrence suffix."""
    h = pd.util.hash_pandas_object(df, index=False).astype("uint64").astype(str)
    return h + "#" + h.groupby(h).cumcount().astype(str)


def rules_fingerprint():
    with open(anomaly_rules.__file__, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _load_flags():
    if not os.path.exists(FLAGS_PATH):
        return None
    return pd.read_csv(FLAGS_PATH, index_col="row_key")


def _load_overrides():
    if not os.path.exists(OVERRIDES_PATH):
        return pd.DataFrame(columns=["decision", "note"]).rename_axis("row_key")
    return pd.read_csv(OVERRIDES_PATH, index_col="row_key", keep_default_na=False)


def _evaluate(df, keys, params):
    flags = pd.DataFrame(
        {r.__name__: r(df, params).fillna(True).to_numpy() for r in anomaly_rules.RULES},
        index=keys.to_numpy(),
    )
    flags[DUPLICATE_COL] = keys.str.split("#").str[1].astype(int).gt(0).to_numpy()
    flags.index.name = "row_key"
    return flags


def sync(df, verbose=True):
    """Bring the store up to date with df. Returns (is_anomaly mask aligned to df, keys)."""
    os.makedirs(STORE_DIR, exist_ok=True)
    keys = row_keys(df)
    fp = rules_fingerprint()
    cached = _load_flags()
    meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else {}

    if cached is None or meta.get("fingerprint") != fp:
        if verbose and cached is not None:
            print("Anomaly rules changed -> re-evaluating all rows.")
        params = anomaly_rules.fit_params(df)
        flags = _evaluate(df, keys, params)
        added, removed = len(flags), 0
    else:
        params = meta["params"]
        removed = len(cached.index.difference(pd.Index(keys)))
        flags = cached[cached.index.isin(keys)]
        new_mask = ~keys.isin(flags.index)
        added = int(new_mask.sum())
        if added:
            new = _evaluate(df[new_mask], keys[new_mask], params)
            flags = pd.concat([flags, new])

    if added or removed or cached is None:
        flags.to_csv(FLAGS_PATH)
        json.dump({"fingerprint": fp, "params": params}, open(META_PATH, "w"), indent=2)
    if verbose:
        print(f"Anomaly store: {added} rows evaluated, {removed} pruned, "
              f"{len(keys) - added} reused from cache.")

    auto = flags.any(axis=1).reindex(keys).to_numpy()
    overrides = _load_overrides()["decision"].reindex(keys).to_numpy()
    mask = pd.Series(auto, index=df.index)
    mask[overrides == "anomaly"] = True
    mask[overrides == "clean"] = False
    if verbose:
        print(f"Overrides applied: {(pd.notna(overrides)).sum()} rows.")
    return mask, keys


def set_overrides(df, positions, decision, note=""):
    keys = row_keys(df)
    ov = _load_overrides()
    for p in positions:
        if decision is None:
            ov = ov.drop(index=keys.iloc[p], errors="ignore")
        else:
            ov.loc[keys.iloc[p]] = [decision, note]
    os.makedirs(STORE_DIR, exist_ok=True)
    ov.to_csv(OVERRIDES_PATH)


def summary(df):
    mask, keys = sync(df, verbose=False)
    flags = _load_flags().reindex(keys)
    print(f"{len(df)} rows, {int(mask.sum())} treated as anomalies after overrides.")
    print(flags.sum().to_string())
    print(f"overrides: {_load_overrides()['decision'].value_counts().to_dict()}")


if __name__ == "__main__":
    from proper_training_size import filename

    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["summary", "flag", "allow", "reset"])
    ap.add_argument("rows", nargs="*", type=int)
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    data = pd.read_excel(filename)
    if args.command == "summary":
        summary(data)
    else:
        decision = {"flag": "anomaly", "allow": "clean", "reset": None}[args.command]
        set_overrides(data, args.rows, decision, args.note)
        print(f"{args.command}: updated {len(args.rows)} row(s).")
