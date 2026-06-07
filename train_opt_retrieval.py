import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from controllers import BaseController
from controllers.mpc import Controller as MPCController
from controllers.opt_retrieval import FEATURE_VERSION, build_feature
from tinyphysics import (
  ACC_G,
  CONTEXT_LENGTH,
  CONTROL_START_IDX,
  COST_END_IDX,
  DEL_T,
  FuturePlan,
  LAT_ACCEL_COST_MULTIPLIER,
  STEER_RANGE,
  State,
  TinyPhysicsModel,
  TinyPhysicsSimulator,
)


TRAIN_MIN = 10000
TRAIN_MAX = 19999
VALID_MIN = 5000
VALID_MAX = 9999


@dataclass
class RolloutRecord:
  cost: dict
  states: list
  targets: np.ndarray
  lataccels: np.ndarray
  actions: np.ndarray


class SequenceController(BaseController):
  def __init__(self, actions):
    self.actions = np.asarray(actions, dtype=np.float32)
    self.step_idx = CONTEXT_LENGTH

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    idx = self.step_idx - CONTROL_START_IDX
    self.step_idx += 1
    if idx < 0:
      return 0.0
    if idx >= len(self.actions):
      return float(self.actions[-1])
    return float(self.actions[idx])


def segment_id(path):
  try:
    return int(Path(path).stem)
  except ValueError as exc:
    raise ValueError(f"Segment file must be named like 12345.csv: {path}") from exc


def split_paths(data_path, start, end, step=1, limit=None, split="train"):
  if split == "train":
    lo, hi = TRAIN_MIN, TRAIN_MAX
  elif split == "validation":
    lo, hi = VALID_MIN, VALID_MAX
  else:
    raise ValueError(f"Unknown split: {split}")
  if start < lo or end > hi:
    raise ValueError(f"{split} split is restricted to {lo:05d}-{hi:05d}; requested {start:05d}-{end:05d}")
  paths = [Path(data_path) / f"{idx:05d}.csv" for idx in range(start, end + 1, step)]
  paths = [p for p in paths if p.exists()]
  if limit is not None:
    paths = paths[:limit]
  if not paths:
    raise ValueError("No segment files found for requested split/range")
  return paths


def rollout(model_path, data_file, controller):
  sim = TinyPhysicsSimulator(TinyPhysicsModel(model_path, False), str(data_file), controller, False)
  cost = sim.rollout()
  return RolloutRecord(
    cost=cost,
    states=sim.state_history,
    targets=np.asarray(sim.target_lataccel_history, dtype=np.float32),
    lataccels=np.asarray(sim.current_lataccel_history, dtype=np.float32),
    actions=np.asarray(sim.action_history, dtype=np.float32),
  )


def mpc_rollout(model_path, data_file):
  return rollout(model_path, data_file, MPCController())


def fixed_rollout(model_path, data_file, actions):
  return rollout(model_path, data_file, SequenceController(actions))


def _coef(v):
  return np.array([np.interp(v, MPCController.KNOT_V, MPCController.KNOT_COEF[:, k]) for k in range(7)])


def arx_matrices(data_file, baseline):
  df = pd.read_csv(data_file)
  roll = np.sin(df["roll"].values) * ACC_G
  v_ego = df["vEgo"].values
  target = df["targetLateralAcceleration"].values
  logged_u = -df["steerCommand"].values
  horizon = COST_END_IDX - CONTROL_START_IDX

  a0 = baseline.lataccels[CONTROL_START_IDX - 1]
  am1 = baseline.lataccels[CONTROL_START_IDX - 2]
  past_u = {
    -1: logged_u[CONTROL_START_IDX - 1],
    -2: logged_u[CONTROL_START_IDX - 2],
    -3: logged_u[CONTROL_START_IDX - 3],
  }

  def simulate_basis(active_col=None):
    a_prev2 = float(am1)
    a_prev1 = float(a0)
    out = np.zeros(horizon, dtype=np.float64)
    for k in range(horizon):
      coef = _coef(v_ego[CONTROL_START_IDX + k])
      a1, a2, b1, b2, b3, g, c = coef

      def u_at(offset):
        idx = k + offset
        if idx < 0:
          return float(past_u[idx])
        if active_col is None:
          return 0.0
        return 1.0 if idx == active_col else 0.0

      ak = (
        a1 * a_prev1 + a2 * a_prev2
        + b1 * u_at(0) + b2 * u_at(-1) + b3 * u_at(-2)
        + g * roll[CONTROL_START_IDX + k] + c
      )
      out[k] = ak
      a_prev2, a_prev1 = a_prev1, ak
    return out

  free = simulate_basis(None)
  response = np.zeros((horizon, horizon), dtype=np.float32)
  for col in range(horizon):
    response[:, col] = simulate_basis(col) - free
  return response, free.astype(np.float32), target[CONTROL_START_IDX:COST_END_IDX].astype(np.float32)


def optimize_actions_arx(data_file, baseline, model_path, prior_weight=0.65, du_weight=0.12, ctrl_weight=0.03):
  response, free, target = arx_matrices(data_file, baseline)
  horizon = len(target)
  baseline_u = baseline.actions[CONTROL_START_IDX:COST_END_IDX].astype(np.float32)

  d_lat = response
  d_jerk = (np.eye(horizon, k=0) - np.eye(horizon, k=-1))[1:].astype(np.float32) @ response / DEL_T
  d_action = (np.eye(horizon, k=0) - np.eye(horizon, k=-1)).astype(np.float32)
  d_action[0, :] = 0.0

  w_track = LAT_ACCEL_COST_MULTIPLIER * 100.0 / horizon
  w_jerk = 100.0 / max(1, horizon - 1)
  h_mat = (
    w_track * (d_lat.T @ d_lat)
    + w_jerk * (d_jerk.T @ d_jerk)
    + du_weight * (d_action.T @ d_action)
    + ctrl_weight * np.eye(horizon, dtype=np.float32)
    + prior_weight * np.eye(horizon, dtype=np.float32)
  )
  f_vec = (
    w_track * d_lat.T @ (free - target)
    + prior_weight * (-baseline_u)
  )

  try:
    candidate = np.linalg.solve(h_mat, -f_vec)
  except np.linalg.LinAlgError:
    candidate = baseline_u
  candidate = np.clip(candidate, STEER_RANGE[0], STEER_RANGE[1]).astype(np.float32)

  best_actions = baseline_u
  best = baseline
  for alpha in (1.0, 0.75, 0.5, 0.25):
    actions = np.clip((1.0 - alpha) * baseline_u + alpha * candidate, STEER_RANGE[0], STEER_RANGE[1])
    rec = fixed_rollout(model_path, data_file, actions)
    if rec.cost["total_cost"] < best.cost["total_cost"]:
      best_actions = actions
      best = rec
  return best_actions, best


def smooth_basis(horizon, basis_step):
  knots = list(range(0, horizon, basis_step))
  if knots[-1] != horizon - 1:
    knots.append(horizon - 1)
  x = np.arange(horizon, dtype=np.float32)
  basis = np.zeros((horizon, len(knots)), dtype=np.float32)
  for i, knot in enumerate(knots):
    left = knots[i - 1] if i > 0 else knot
    right = knots[i + 1] if i + 1 < len(knots) else knot
    if knot > left:
      mask = (x >= left) & (x <= knot)
      basis[mask, i] = (x[mask] - left) / max(knot - left, 1)
    basis[x == knot, i] = 1.0
    if right > knot:
      mask = (x >= knot) & (x <= right)
      basis[mask, i] = np.maximum(basis[mask, i], (right - x[mask]) / max(right - knot, 1))
  basis /= np.maximum(np.linalg.norm(basis, axis=0, keepdims=True), 1e-6)
  return basis.astype(np.float32)


def solve_quadratic_update(base_lat, target, basis_response, damping):
  horizon, dims = basis_response.shape
  err = base_lat - target
  jerk = np.diff(base_lat) / DEL_T
  response_jerk = np.diff(basis_response, axis=0) / DEL_T

  w_track = LAT_ACCEL_COST_MULTIPLIER * 100.0 / horizon
  w_jerk = 100.0 / max(1, horizon - 1)
  h_mat = (
    w_track * (basis_response.T @ basis_response)
    + w_jerk * (response_jerk.T @ response_jerk)
    + damping * np.eye(dims, dtype=np.float32)
  )
  f_vec = w_track * (basis_response.T @ err) + w_jerk * (response_jerk.T @ jerk)
  try:
    return np.linalg.solve(h_mat, -f_vec).astype(np.float32)
  except np.linalg.LinAlgError:
    return np.zeros(dims, dtype=np.float32)


def optimize_actions_exact_basis(data_file, baseline, model_path, basis_step=20, fd_eps=0.20,
                                 iterations=2, damping=0.5):
  target = baseline.targets[CONTROL_START_IDX:COST_END_IDX].astype(np.float32)
  horizon = len(target)
  basis = smooth_basis(horizon, basis_step)

  best_actions = baseline.actions[CONTROL_START_IDX:COST_END_IDX].astype(np.float32)
  best = baseline
  for _ in range(iterations):
    base_lat = best.lataccels[CONTROL_START_IDX:COST_END_IDX].astype(np.float32)
    responses = np.zeros((horizon, basis.shape[1]), dtype=np.float32)
    for col in range(basis.shape[1]):
      perturbed = np.clip(best_actions + fd_eps * basis[:, col], STEER_RANGE[0], STEER_RANGE[1])
      rec = fixed_rollout(model_path, data_file, perturbed)
      responses[:, col] = (rec.lataccels[CONTROL_START_IDX:COST_END_IDX] - base_lat) / fd_eps

    delta = solve_quadratic_update(base_lat, target, responses, damping)
    action_delta = basis @ delta
    accepted = False
    for alpha in (1.0, 0.5, 0.25, 0.125):
      candidate = np.clip(best_actions + alpha * action_delta, STEER_RANGE[0], STEER_RANGE[1])
      rec = fixed_rollout(model_path, data_file, candidate)
      if rec.cost["total_cost"] + 1e-6 < best.cost["total_cost"]:
        best_actions = candidate.astype(np.float32)
        best = rec
        accepted = True
        break
    if not accepted:
      break
  return best_actions, best


def future_plan_from_data(data, step_idx):
  return FuturePlan(
    lataccel=data["target_lataccel"].values[step_idx + 1:step_idx + 50].tolist(),
    roll_lataccel=data["roll_lataccel"].values[step_idx + 1:step_idx + 50].tolist(),
    v_ego=data["v_ego"].values[step_idx + 1:step_idx + 50].tolist(),
    a_ego=data["a_ego"].values[step_idx + 1:step_idx + 50].tolist(),
  )


def load_processed_data(data_file):
  df = pd.read_csv(data_file)
  return pd.DataFrame({
    "roll_lataccel": np.sin(df["roll"].values) * ACC_G,
    "v_ego": df["vEgo"].values,
    "a_ego": df["aEgo"].values,
    "target_lataccel": df["targetLateralAcceleration"].values,
  })


def extract_snippets(data_file, teacher, stride, snippet_horizon):
  data = load_processed_data(data_file)
  features = []
  snippets = []
  action_hist = []
  error_hist = []
  integral_error = 0.0
  for step_idx in range(CONTEXT_LENGTH, COST_END_IDX):
    current = float(teacher.lataccels[max(0, step_idx - 1)])
    target = float(teacher.targets[step_idx])
    error = target - current
    integral_error = float(np.clip(integral_error + error, -40.0, 40.0))

    if step_idx >= CONTROL_START_IDX and (step_idx - CONTROL_START_IDX) % stride == 0:
      state_row = data.iloc[step_idx]
      state = State(
        roll_lataccel=state_row["roll_lataccel"],
        v_ego=state_row["v_ego"],
        a_ego=state_row["a_ego"],
      )
      feature = build_feature(
        target, current, state, future_plan_from_data(data, step_idx),
        action_hist, error_hist, integral_error,
      )
      action_start = step_idx
      action_end = min(step_idx + snippet_horizon, len(teacher.actions))
      snippet = np.zeros(snippet_horizon, dtype=np.float32)
      vals = teacher.actions[action_start:action_end]
      snippet[:len(vals)] = vals
      if len(vals) < snippet_horizon and len(vals):
        snippet[len(vals):] = vals[-1]
      features.append(feature)
      snippets.append(snippet)

    action_hist.append(float(teacher.actions[step_idx]))
    error_hist.append(error)
    action_hist = action_hist[-4:]
    error_hist = error_hist[-4:]
  return features, snippets


def random_projection(feature_dim, projection_dim, seed):
  rng = np.random.default_rng(seed)
  proj = rng.normal(0.0, 1.0 / np.sqrt(projection_dim), size=(feature_dim, projection_dim))
  return proj.astype(np.float32)


def kmeans(features, n_centroids, seed, iterations=8):
  rng = np.random.default_rng(seed)
  n_centroids = min(n_centroids, len(features))
  centroids = features[rng.choice(len(features), size=n_centroids, replace=False)].astype(np.float32)
  assignment = np.zeros(len(features), dtype=np.int64)
  for _ in range(iterations):
    assignment = assign_centroids(features, centroids)
    for c in range(n_centroids):
      mask = assignment == c
      if np.any(mask):
        centroids[c] = np.mean(features[mask], axis=0)
  return centroids, assignment


def assign_centroids(features, centroids, batch_size=4096):
  assignment = np.zeros(len(features), dtype=np.int64)
  for start in range(0, len(features), batch_size):
    end = min(start + batch_size, len(features))
    d2 = np.sum((features[start:end, None, :] - centroids[None, :, :]) ** 2, axis=2)
    assignment[start:end] = np.argmin(d2, axis=1)
  return assignment


def build_library(args):
  paths = split_paths(args.data_path, args.train_start, args.train_end, args.train_step, args.num_segments, split="train")
  all_features = []
  all_actions = []
  summaries = []

  for idx, data_file in enumerate(paths, 1):
    baseline = mpc_rollout(args.model_path, data_file)
    if args.teacher == "hybrid":
      arx_actions, arx_teacher = optimize_actions_arx(data_file, baseline, args.model_path)
      teacher_actions, teacher = optimize_actions_exact_basis(
        data_file, arx_teacher, args.model_path,
        basis_step=args.basis_step,
        fd_eps=args.fd_eps,
        iterations=args.teacher_iterations,
        damping=args.teacher_damping,
      )
    elif args.teacher == "exact_basis":
      teacher_actions, teacher = optimize_actions_exact_basis(
        data_file, baseline, args.model_path,
        basis_step=args.basis_step,
        fd_eps=args.fd_eps,
        iterations=args.teacher_iterations,
        damping=args.teacher_damping,
      )
    else:
      teacher_actions, teacher = optimize_actions_arx(data_file, baseline, args.model_path)
    features, snippets = extract_snippets(data_file, teacher, args.snippet_stride, args.snippet_horizon)
    all_features.extend(features)
    all_actions.extend(snippets)
    summaries.append((int(data_file.stem), baseline.cost["total_cost"], teacher.cost["total_cost"], len(features)))
    print(
      f"{idx:04d}/{len(paths):04d} {data_file.stem} "
      f"mpc={baseline.cost['total_cost']:.3f} teacher={teacher.cost['total_cost']:.3f} snippets={len(features)}",
      flush=True,
    )

  feature_matrix = np.asarray(all_features, dtype=np.float32)
  action_matrix = np.asarray(all_actions, dtype=np.float32)
  if args.max_snippets and len(feature_matrix) > args.max_snippets:
    rng = np.random.default_rng(args.seed)
    keep = rng.choice(len(feature_matrix), size=args.max_snippets, replace=False)
    feature_matrix = feature_matrix[keep]
    action_matrix = action_matrix[keep]

  feature_mean = np.mean(feature_matrix, axis=0).astype(np.float32)
  feature_scale = np.std(feature_matrix, axis=0).astype(np.float32)
  feature_scale = np.maximum(feature_scale, 1e-3)
  projection = random_projection(feature_matrix.shape[1], args.projection_dim, args.seed)
  projected = ((feature_matrix - feature_mean) / feature_scale) @ projection
  projected = projected.astype(np.float32)
  centroids, assignment = kmeans(projected, args.centroids, args.seed)
  assignment = assign_centroids(projected, centroids)
  assigned_d = np.sqrt(np.sum((projected - centroids[assignment]) ** 2, axis=1) / projected.shape[1])
  distance_scale = np.array([max(float(np.percentile(assigned_d, 35)), 1e-3)], dtype=np.float32)

  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    out,
    version=np.array([FEATURE_VERSION], dtype=np.int32),
    feature_mean=feature_mean,
    feature_scale=feature_scale,
    projection=projection,
    features=projected.astype(np.float16),
    actions=action_matrix.astype(np.float16),
    centroids=centroids.astype(np.float32),
    feature_centroid=assignment.astype(np.int16),
    distance_scale=distance_scale,
    snippet_horizon=np.array([args.snippet_horizon], dtype=np.int32),
    train_start=np.array([args.train_start], dtype=np.int32),
    train_end=np.array([args.train_end], dtype=np.int32),
    summaries=np.asarray(summaries, dtype=np.float32),
  )
  print(f"Saved {len(feature_matrix)} snippets to {out}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", default="./models/tinyphysics.onnx")
  parser.add_argument("--data_path", default="./data")
  parser.add_argument("--output", default="./controllers/artifacts/opt_retrieval_library.npz")
  parser.add_argument("--train_start", type=int, default=10000)
  parser.add_argument("--train_end", type=int, default=19999)
  parser.add_argument("--train_step", type=int, default=50)
  parser.add_argument("--num_segments", type=int, default=80)
  parser.add_argument("--snippet_stride", type=int, default=4)
  parser.add_argument("--snippet_horizon", type=int, default=10)
  parser.add_argument("--projection_dim", type=int, default=32)
  parser.add_argument("--centroids", type=int, default=128)
  parser.add_argument("--max_snippets", type=int, default=50000)
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--teacher", choices=["arx", "exact_basis", "hybrid"], default="arx")
  parser.add_argument("--basis_step", type=int, default=20)
  parser.add_argument("--fd_eps", type=float, default=0.20)
  parser.add_argument("--teacher_iterations", type=int, default=2)
  parser.add_argument("--teacher_damping", type=float, default=0.5)
  args = parser.parse_args()
  build_library(args)


if __name__ == "__main__":
  main()
