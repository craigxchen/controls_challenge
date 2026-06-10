import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from controllers.lib.blackbox_response import (
  A_LAGS,
  DEFAULT_ARTIFACT,
  MODEL_VERSION,
  ROLL_LAGS,
  U_LAGS,
  V_LAGS,
  Y_LAGS,
  LinearBlackboxResponse,
  build_raw_features,
  design_matrix,
)
from tinyphysics import (
  ACC_G,
  CONTEXT_LENGTH,
  CONTROL_START_IDX,
  LATACCEL_RANGE,
  MAX_ACC_DELTA,
  State,
  TinyPhysicsModel,
)


TRAIN_MIN = 10000
TRAIN_MAX = 19999
VALID_MIN = 5000
VALID_MAX = 9999

SPEED_KNOTS = np.array([2.0, 6.0, 10.0, 14.0, 18.0, 24.0, 30.0, 34.0, 38.0], dtype=np.float32)


def split_paths(data_path, start, end, step, limit, split):
  if split == "train":
    lo, hi = TRAIN_MIN, TRAIN_MAX
  elif split == "validation":
    lo, hi = VALID_MIN, VALID_MAX
  else:
    raise ValueError("split must be 'train' or 'validation'")
  if start < lo or end > hi:
    raise ValueError(f"{split} split is restricted to {lo:05d}-{hi:05d}; requested {start:05d}-{end:05d}")
  paths = [Path(data_path) / f"{idx:05d}.csv" for idx in range(start, end + 1, step)]
  paths = [p for p in paths if p.exists()]
  if limit is not None:
    paths = paths[:limit]
  if not paths:
    raise ValueError("No segment files found for requested split/range")
  return paths


def load_processed_data(data_file):
  df = pd.read_csv(data_file)
  return pd.DataFrame({
    "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
    "v_ego": df["vEgo"].values,
    "a_ego": df["aEgo"].values,
    "target_lataccel": df["targetLateralAcceleration"].values,
    "steer_command": -df["steerCommand"].values,
  })


def expected_lataccel(model, states, actions, past_preds, current=None):
  tokens = np.clip(model.tokenizer.encode(past_preds), 0, model.tokenizer.vocab_size - 1)
  raw_states = [list(x) for x in states]
  model_input = {
    "states": np.expand_dims(np.column_stack([actions, raw_states]), axis=0).astype(np.float32),
    "tokens": np.expand_dims(tokens, axis=0).astype(np.int64),
  }
  logits = model.ort_session.run(None, model_input)[0]
  logits = logits[0, -1] / 0.8
  finite = np.isfinite(logits)
  if not np.any(finite):
    return None
  if not np.all(finite):
    idx = int(np.nanargmax(logits))
    pred = float(model.tokenizer.bins[idx])
  else:
    probs = model.softmax(logits, axis=-1)
    pred = float(np.sum(probs * model.tokenizer.bins))
  pred = float(np.clip(pred, LATACCEL_RANGE[0], LATACCEL_RANGE[1]))
  if current is not None:
    pred = float(np.clip(pred, current - MAX_ACC_DELTA, current + MAX_ACC_DELTA))
  return pred


def history(values, size, default):
  values = list(values)
  if not values:
    values = [default]
  out = [float(v) for v in values[-1:-size - 1:-1]]
  while len(out) < size:
    out.append(out[-1])
  return out


def state_lags(values, step_idx, size):
  out = []
  for lag in range(size):
    idx = max(0, step_idx - lag)
    out.append(float(values[idx]))
  return out


def feature_from_hist(data, step_idx, lat_hist, action_hist):
  y_lags = history(lat_hist, Y_LAGS, 0.0)
  u_lags = history(action_hist, U_LAGS, 0.0)
  roll_lags = state_lags(data["roll_lataccel"].values, step_idx, ROLL_LAGS)
  v_lags = state_lags(data["v_ego"].values, step_idx, V_LAGS)
  a_lags = state_lags(data["a_ego"].values, step_idx, A_LAGS)
  return build_raw_features(y_lags, u_lags, roll_lags, v_lags, a_lags, zero_current_action=True)


def policy_action(policy, data, step_idx, prev_action, rng):
  logged = float(data["steer_command"].values[step_idx])
  target = float(data["target_lataccel"].values[step_idx])
  roll = float(data["roll_lataccel"].values[step_idx])
  speed = float(data["v_ego"].values[step_idx])
  if policy == "logged":
    action = logged
  elif policy == "road_proxy":
    raw = 0.45 * (target - roll)
    action = 0.85 * prev_action + 0.15 * raw
  elif policy == "smooth_noise":
    action = 0.94 * prev_action + rng.normal(0.0, 0.18)
  elif policy == "sine":
    phase = rng.uniform(-np.pi, np.pi)
    action = 0.65 * np.sin(0.045 * step_idx + phase) + 0.35 * logged
  elif policy == "pulses":
    center = rng.uniform(CONTROL_START_IDX + 20, CONTROL_START_IDX + 340)
    width = rng.uniform(18.0, 90.0)
    amp = rng.uniform(-1.2, 1.2)
    action = 0.75 * prev_action + 0.25 * logged + amp * np.exp(-((step_idx - center) / width) ** 2)
  elif policy == "speed_sweep":
    raw = np.tanh((speed - 22.0) / 8.0) * np.sin(0.035 * step_idx + rng.uniform(-np.pi, np.pi))
    action = 0.82 * prev_action + 0.18 * (logged + raw)
  else:
    raise ValueError(f"Unknown policy: {policy}")
  return float(np.clip(action, -2.0, 2.0))


def second_derivative(u_minus, y_minus, u0, y0, u_plus, y_plus):
  if min(abs(u0 - u_minus), abs(u_plus - u0), abs(u_plus - u_minus)) < 1e-6:
    return 0.0
  xs = np.asarray([u_minus, u0, u_plus], dtype=float)
  ys = np.asarray([y_minus, y0, y_plus], dtype=float)
  try:
    coef = np.polyfit(xs, ys, deg=2)
  except np.linalg.LinAlgError:
    return 0.0
  return float(2.0 * coef[0])


def collect_rows(model, data_file, policies, seed, fd_eps):
  data = load_processed_data(data_file)
  rows = []
  next_labels = []
  delta_labels = []
  gain_labels = []
  curv_labels = []
  current_actions = []
  currents = []
  speeds = []
  rng = np.random.default_rng(seed + int(data_file.stem))
  states_all = [
    State(row.roll_lataccel, row.v_ego, row.a_ego)
    for row in data[["roll_lataccel", "v_ego", "a_ego"]].itertuples(index=False)
  ]

  for policy in policies:
    lat_hist = data["target_lataccel"].values[:CONTEXT_LENGTH].astype(float).tolist()
    action_hist = data["steer_command"].values[:CONTEXT_LENGTH].astype(float).tolist()
    state_hist = states_all[:CONTEXT_LENGTH]
    current = float(lat_hist[-1])
    prev_action = float(action_hist[-1])

    for step_idx in range(CONTEXT_LENGTH, len(data)):
      state_hist.append(states_all[step_idx])
      if step_idx < CONTROL_START_IDX:
        action = float(data["steer_command"].values[step_idx])
        pred = float(data["target_lataccel"].values[step_idx])
        action_hist.append(action)
      else:
        action = policy_action(policy, data, step_idx, prev_action, rng)
        action_hist.append(action)
        action_context = action_hist[-CONTEXT_LENGTH:]
        state_context = state_hist[-CONTEXT_LENGTH:]
        pred_context = lat_hist[-CONTEXT_LENGTH:]
        pred = expected_lataccel(model, state_context, action_context, pred_context, current=current)
        if pred is None or not np.isfinite(pred):
          pred = current
        else:
          u_plus = float(np.clip(action + fd_eps, -2.0, 2.0))
          u_minus = float(np.clip(action - fd_eps, -2.0, 2.0))
          plus_context = list(action_context)
          minus_context = list(action_context)
          plus_context[-1] = u_plus
          minus_context[-1] = u_minus
          pred_plus = expected_lataccel(model, state_context, plus_context, pred_context, current=current)
          pred_minus = expected_lataccel(model, state_context, minus_context, pred_context, current=current)
          if pred_plus is not None and pred_minus is not None and abs(u_plus - u_minus) > 1e-6:
            rows.append(feature_from_hist(data, step_idx, lat_hist, action_hist))
            next_labels.append(pred)
            delta_labels.append(pred - current)
            gain_labels.append((pred_plus - pred_minus) / (u_plus - u_minus))
            curv_labels.append(second_derivative(u_minus, pred_minus, action, pred, u_plus, pred_plus))
            current_actions.append(action)
            currents.append(current)
            speeds.append(float(data["v_ego"].values[step_idx]))

      lat_hist.append(pred)
      current = float(pred)
      prev_action = float(action)

  return {
    "raw": rows,
    "next": next_labels,
    "delta": delta_labels,
    "gain": gain_labels,
    "curv": curv_labels,
    "u0": current_actions,
    "current": currents,
    "speed": speeds,
  }


def fit_weighted_ridge(design, y, speed, knot, bandwidth, ridge, intercept_scale=0.02):
  w = np.exp(-0.5 * ((speed - knot) / bandwidth) ** 2)
  w = np.maximum(w, 1e-4)
  xw = design * np.sqrt(w[:, None])
  yw = y * np.sqrt(w)
  gram = xw.T @ xw
  reg = ridge * np.eye(gram.shape[0], dtype=np.float64)
  reg[0, 0] = ridge * intercept_scale
  try:
    return np.linalg.solve(gram + reg, xw.T @ yw)
  except np.linalg.LinAlgError:
    return np.linalg.lstsq(gram + reg, xw.T @ yw, rcond=None)[0]


def fit_table(design, y, speed, bandwidth, ridge):
  return np.asarray([
    fit_weighted_ridge(design, y, speed, float(knot), bandwidth, ridge)
    for knot in SPEED_KNOTS
  ], dtype=np.float32)


def predict_table(coef_table, design, speed):
  out = np.zeros(len(design), dtype=np.float64)
  for idx, v in enumerate(speed):
    coef = np.asarray([
      np.interp(v, SPEED_KNOTS, coef_table[:, col])
      for col in range(coef_table.shape[1])
    ], dtype=np.float64)
    out[idx] = float(design[idx] @ coef)
  return out


def merge_collections(collections):
  keys = ("raw", "next", "delta", "gain", "curv", "u0", "current", "speed")
  return {key: np.asarray(sum((list(c[key]) for c in collections), []), dtype=np.float64) for key in keys}


def random_hidden(raw_dim, count, seed):
  if count <= 0:
    return (
      np.zeros((raw_dim, 0), dtype=np.float32),
      np.zeros(0, dtype=np.float32),
    )
  rng = np.random.default_rng(seed)
  weight = rng.normal(0.0, 1.0 / np.sqrt(raw_dim), size=(raw_dim, count)).astype(np.float32)
  bias = rng.uniform(-np.pi, np.pi, size=count).astype(np.float32)
  return weight, bias


def fit_artifact(train, args):
  raw = train["raw"].astype(np.float64)
  feature_mean = np.mean(raw, axis=0)
  feature_scale = np.maximum(np.std(raw, axis=0), 1e-3)
  hidden_weight, hidden_bias = random_hidden(raw.shape[1], args.random_features, args.seed + 991)
  design = design_matrix(raw, feature_mean, feature_scale, hidden_weight, hidden_bias).astype(np.float64)
  speed = train["speed"].astype(np.float64)
  u0 = train["u0"].astype(np.float64)
  delta = train["delta"].astype(np.float64)

  gain_clip = np.percentile(train["gain"], [1.0, 99.0]).astype(np.float64)
  curv_clip = np.percentile(train["curv"], [2.0, 98.0]).astype(np.float64)
  gain_y = np.clip(train["gain"], gain_clip[0], gain_clip[1])
  curv_y = np.clip(train["curv"], curv_clip[0], curv_clip[1])

  gain_coef = fit_table(design, gain_y, speed, args.bandwidth, args.gain_ridge)
  gain_pred = np.clip(predict_table(gain_coef, design, speed), gain_clip[0], gain_clip[1])
  curv_coef = fit_table(design, curv_y, speed, args.bandwidth, args.curv_ridge)
  curv_pred = np.clip(predict_table(curv_coef, design, speed), curv_clip[0], curv_clip[1])
  base_delta = delta - gain_pred * u0 - 0.5 * curv_pred * u0 * u0
  delta_coef = fit_table(design, base_delta, speed, args.bandwidth, args.delta_ridge)

  return {
    "feature_mean": feature_mean,
    "feature_scale": feature_scale,
    "delta_coef": delta_coef,
    "gain_coef": gain_coef,
    "curv_coef": curv_coef,
    "gain_clip": gain_clip,
    "curv_clip": curv_clip,
    "hidden_weight": hidden_weight,
    "hidden_bias": hidden_bias,
  }


def save_artifact(artifact, train_range, args):
  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    out,
    model_version=np.array([MODEL_VERSION], dtype=np.int32),
    speed_knots=SPEED_KNOTS.astype(np.float32),
    delta_coef=artifact["delta_coef"].astype(np.float32),
    gain_coef=artifact["gain_coef"].astype(np.float32),
    curv_coef=artifact["curv_coef"].astype(np.float32),
    feature_mean=artifact["feature_mean"].astype(np.float32),
    feature_scale=artifact["feature_scale"].astype(np.float32),
    hidden_weight=artifact["hidden_weight"].astype(np.float32),
    hidden_bias=artifact["hidden_bias"].astype(np.float32),
    y_lags=np.array([Y_LAGS], dtype=np.int32),
    u_lags=np.array([U_LAGS], dtype=np.int32),
    roll_lags=np.array([ROLL_LAGS], dtype=np.int32),
    v_lags=np.array([V_LAGS], dtype=np.int32),
    a_lags=np.array([A_LAGS], dtype=np.int32),
    max_delta=np.array([MAX_ACC_DELTA], dtype=np.float32),
    lat_min=np.array([LATACCEL_RANGE[0]], dtype=np.float32),
    lat_max=np.array([LATACCEL_RANGE[1]], dtype=np.float32),
    gain_clip=artifact["gain_clip"].astype(np.float32),
    curv_clip=artifact["curv_clip"].astype(np.float32),
    fd_eps=np.array([args.fd_eps], dtype=np.float32),
    train_start=np.array([train_range[0]], dtype=np.int32),
    train_end=np.array([train_range[1]], dtype=np.int32),
  )
  return out


def evaluate_one_step(response_model, data):
  preds = []
  for raw, u0, current, speed in zip(data["raw"], data["u0"], data["current"], data["speed"]):
    preds.append(response_model.predict_from_raw(raw, u0, current, speed, clip_output=True))
  preds = np.asarray(preds, dtype=np.float64)
  y = data["next"].astype(np.float64)
  err = preds - y
  return {
    "rmse": float(np.sqrt(np.mean(err * err))),
    "mae": float(np.mean(np.abs(err))),
    "p95_abs": float(np.percentile(np.abs(err), 95.0)),
  }


def collect_split(model, paths, policies, seed, fd_eps, split_name):
  collections = []
  for idx, path in enumerate(paths, 1):
    c = collect_rows(model, path, policies, seed, fd_eps)
    collections.append(c)
    print(f"{split_name} {idx:04d}/{len(paths):04d} {path.stem} rows={len(c['raw'])}", flush=True)
  return merge_collections(collections)


def fit_model(args):
  train_paths = split_paths(args.data_path, args.train_start, args.train_end, args.train_step, args.limit, "train")
  model = TinyPhysicsModel(args.model_path, False)
  train_policies = tuple(x for x in args.policies.split(",") if x)
  train = collect_split(model, train_paths, train_policies, args.seed, args.fd_eps, "train")
  if len(train["raw"]) == 0:
    raise ValueError("No training rows collected")

  artifact = fit_artifact(train, args)
  out = save_artifact(artifact, (args.train_start, args.train_end), args)
  response = LinearBlackboxResponse(out)
  train_metrics = evaluate_one_step(response, train)
  print(f"Saved {len(train['raw'])} rows to {out}")
  print(
    "train one-step "
    f"rmse={train_metrics['rmse']:.6f} mae={train_metrics['mae']:.6f} p95_abs={train_metrics['p95_abs']:.6f}"
  )
  print(
    "train jacobian clips "
    f"gain=[{artifact['gain_clip'][0]:.4f},{artifact['gain_clip'][1]:.4f}] "
    f"curv=[{artifact['curv_clip'][0]:.4f},{artifact['curv_clip'][1]:.4f}]"
  )

  if not args.skip_validation:
    valid_paths = split_paths(
      args.data_path,
      args.validation_start,
      args.validation_end,
      args.validation_step,
      args.validation_limit,
      "validation",
    )
    valid_policies = tuple(x for x in args.validation_policies.split(",") if x)
    valid = collect_split(model, valid_paths, valid_policies, args.seed + 17, args.fd_eps, "valid")
    valid_metrics = evaluate_one_step(response, valid)
    print(
      "validation one-step "
      f"rmse={valid_metrics['rmse']:.6f} mae={valid_metrics['mae']:.6f} p95_abs={valid_metrics['p95_abs']:.6f}"
    )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
  parser.add_argument("--data_path", default="./data")
  parser.add_argument("--output", default=str(DEFAULT_ARTIFACT))
  parser.add_argument("--train_start", type=int, default=10000)
  parser.add_argument("--train_end", type=int, default=19999)
  parser.add_argument("--train_step", type=int, default=100)
  parser.add_argument("--limit", type=int, default=60)
  parser.add_argument("--policies", default="logged,road_proxy,smooth_noise,sine,pulses,speed_sweep")
  parser.add_argument("--fd_eps", type=float, default=0.18)
  parser.add_argument("--bandwidth", type=float, default=5.5)
  parser.add_argument("--delta_ridge", type=float, default=65.0)
  parser.add_argument("--gain_ridge", type=float, default=120.0)
  parser.add_argument("--curv_ridge", type=float, default=250.0)
  parser.add_argument("--random_features", type=int, default=192)
  parser.add_argument("--seed", type=int, default=20260608)
  parser.add_argument("--skip_validation", action="store_true")
  parser.add_argument("--validation_start", type=int, default=5000)
  parser.add_argument("--validation_end", type=int, default=9999)
  parser.add_argument("--validation_step", type=int, default=250)
  parser.add_argument("--validation_limit", type=int, default=None)
  parser.add_argument("--validation_policies", default="road_proxy,smooth_noise,sine,speed_sweep")
  args = parser.parse_args()
  fit_model(args)


if __name__ == "__main__":
  main()
