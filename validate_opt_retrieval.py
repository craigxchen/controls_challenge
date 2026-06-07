import argparse
from pathlib import Path

import numpy as np

from tinyphysics import run_rollout


VALID_MIN = 5000
VALID_MAX = 9999
TRAIN_MIN = 10000
TRAIN_MAX = 19999


def split_paths(data_path, start, end, step, limit, split):
  if split == "validation":
    lo, hi = VALID_MIN, VALID_MAX
  elif split == "train":
    lo, hi = TRAIN_MIN, TRAIN_MAX
  else:
    raise ValueError("split must be 'validation' or 'train'")
  if start < lo or end > hi:
    raise ValueError(f"{split} split is restricted to {lo:05d}-{hi:05d}; requested {start:05d}-{end:05d}")
  paths = [Path(data_path) / f"{idx:05d}.csv" for idx in range(start, end + 1, step)]
  paths = [p for p in paths if p.exists()]
  if limit is not None:
    paths = paths[:limit]
  if not paths:
    raise ValueError("No files matched requested range")
  return paths


def summarize(rows):
  keys = rows[0].keys()
  summary = {k: float(np.mean([row[k] for row in rows])) for k in keys}
  summary["p95_total"] = float(np.percentile([row["total_cost"] for row in rows], 95))
  return summary


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
  parser.add_argument("--data_path", default="./data")
  parser.add_argument("--controller", default="opt_retrieval")
  parser.add_argument("--baseline", default="mpc")
  parser.add_argument("--split", choices=["validation", "train"], default="validation")
  parser.add_argument("--start", type=int, default=5000)
  parser.add_argument("--end", type=int, default=9999)
  parser.add_argument("--step", type=int, default=100)
  parser.add_argument("--limit", type=int, default=None)
  args = parser.parse_args()

  paths = split_paths(args.data_path, args.start, args.end, args.step, args.limit, args.split)
  results = {}
  for controller in [args.baseline, args.controller]:
    rows = []
    worst = []
    for data_file in paths:
      cost, _, _ = run_rollout(data_file, controller, args.model_path, debug=False)
      rows.append(cost)
      worst.append((data_file.stem, cost["total_cost"], cost["lataccel_cost"], cost["jerk_cost"]))
    results[controller] = summarize(rows)
    print(controller, results[controller])
    print("worst", sorted(worst, key=lambda row: row[1], reverse=True)[:10])


if __name__ == "__main__":
  main()
