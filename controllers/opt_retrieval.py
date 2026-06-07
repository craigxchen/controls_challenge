from pathlib import Path

import numpy as np

from . import BaseController
from .mpc import Controller as MPCController


CONTROL_START_IDX = 100
CONTEXT_LENGTH = 20
STEER_MIN = -2.0
STEER_MAX = 2.0
FEATURE_VERSION = 1
FUTURE_IDXS = np.array([0, 1, 2, 4, 7, 11, 16, 23, 32, 45], dtype=np.int64)
ACTION_HIST_LEN = 4
ERROR_HIST_LEN = 4


def _safe_at(values, idx, default=0.0):
  if values is None or len(values) == 0:
    return float(default)
  idx = min(int(idx), len(values) - 1)
  return float(values[idx])


def _tail(values, size):
  arr = np.zeros(size, dtype=np.float32)
  if len(values):
    src = np.asarray(values[-size:], dtype=np.float32)
    arr[-len(src):] = src
  return arr


def build_feature(target_lataccel, current_lataccel, state, future_plan,
                  action_hist, error_hist, integral_error):
  """Build the runtime/train feature vector shared by the library and controller."""
  error = float(target_lataccel - current_lataccel)
  future_lat = np.array([_safe_at(future_plan.lataccel, i, target_lataccel) for i in FUTURE_IDXS], dtype=np.float32)
  future_roll = np.array([_safe_at(future_plan.roll_lataccel, i, state.roll_lataccel) for i in FUTURE_IDXS], dtype=np.float32)
  future_v = np.array([_safe_at(future_plan.v_ego, i, state.v_ego) for i in FUTURE_IDXS], dtype=np.float32)
  future_a = np.array([_safe_at(future_plan.a_ego, i, state.a_ego) for i in FUTURE_IDXS], dtype=np.float32)

  actions = _tail(action_hist, ACTION_HIST_LEN)
  errors = _tail(error_hist, ERROR_HIST_LEN)
  lat_delta = future_lat - float(target_lataccel)
  roll_delta = future_roll - float(state.roll_lataccel)
  v_delta = future_v - float(state.v_ego)

  feature = np.concatenate([
    np.array([
      current_lataccel / 5.0,
      target_lataccel / 5.0,
      error / 5.0,
      np.clip(integral_error, -40.0, 40.0) / 40.0,
      state.roll_lataccel / 5.0,
      state.v_ego / 40.0,
      state.a_ego / 5.0,
      np.mean(actions) / 2.0,
      np.std(actions) / 2.0,
      np.mean(errors) / 5.0,
      np.std(errors) / 5.0,
    ], dtype=np.float32),
    actions / 2.0,
    errors / 5.0,
    future_lat / 5.0,
    lat_delta / 5.0,
    future_roll / 5.0,
    roll_delta / 5.0,
    future_v / 40.0,
    v_delta / 20.0,
    future_a / 5.0,
  ]).astype(np.float32)
  return feature


class RetrievalLibrary:
  def __init__(self, artifact_path):
    data = np.load(artifact_path)
    version = int(data.get("version", np.array([0]))[0])
    if version != FEATURE_VERSION:
      raise ValueError(f"Unsupported opt_retrieval artifact version: {version}")

    self.feature_mean = data["feature_mean"].astype(np.float32)
    self.feature_scale = data["feature_scale"].astype(np.float32)
    self.projection = data["projection"].astype(np.float32)
    self.features = data["features"].astype(np.float32)
    self.actions = data["actions"].astype(np.float32)
    self.centroids = data["centroids"].astype(np.float32)
    self.feature_centroid = data["feature_centroid"].astype(np.int64)
    self.distance_scale = float(data["distance_scale"][0])
    self.snippet_horizon = int(data["snippet_horizon"][0])

  def _project(self, feature):
    x = (feature.astype(np.float32) - self.feature_mean) / self.feature_scale
    return x @ self.projection

  def query(self, feature, top_k=16, centroid_k=5):
    z = self._project(feature)
    centroid_d2 = np.sum((self.centroids - z) ** 2, axis=1)
    centroid_k = min(centroid_k, len(self.centroids))
    centroid_ids = np.argpartition(centroid_d2, centroid_k - 1)[:centroid_k]
    mask = np.isin(self.feature_centroid, centroid_ids)
    candidate_idx = np.flatnonzero(mask)
    if len(candidate_idx) < top_k:
      candidate_idx = np.arange(len(self.features))

    d2 = np.sum((self.features[candidate_idx] - z) ** 2, axis=1)
    top_k = min(top_k, len(candidate_idx))
    local = np.argpartition(d2, top_k - 1)[:top_k]
    idx = candidate_idx[local]
    d2 = d2[local]

    d_min = float(np.min(d2))
    temp = max(self.distance_scale ** 2, 1e-3)
    weights = np.exp(-(d2 - d_min) / temp)
    weights /= np.sum(weights)

    snippets = self.actions[idx]
    action0 = float(weights @ snippets[:, 0])
    action_std = float(np.sqrt(weights @ ((snippets[:, 0] - action0) ** 2)))
    dist = float(np.sqrt(d_min / max(1, self.features.shape[1])))
    dist_conf = 1.0 / (1.0 + (dist / max(self.distance_scale, 1e-3)) ** 2)
    agree_conf = 1.0 / (1.0 + (action_std / 0.35) ** 2)
    confidence = float(np.clip(dist_conf * agree_conf, 0.0, 1.0))
    return action0, confidence, {"distance": dist, "action_std": action_std}


class Controller(BaseController):
  """
  Train-snippet retrieval controller.

  The runtime path is intentionally NumPy-only. If the retrieval artifact is
  absent, corrupt, or low-confidence for a state, the controller blends back
  toward the existing MPC fallback.
  """

  def __init__(self, artifact_path=None, retrieval_blend=0.05, smooth=0.25,
               feedback_p=0.0, feedback_d=0.0):
    if artifact_path is None:
      artifact_path = Path(__file__).resolve().parent / "artifacts" / "opt_retrieval_library.npz"
    self.library = None
    try:
      if Path(artifact_path).exists():
        self.library = RetrievalLibrary(artifact_path)
    except Exception:
      self.library = None

    self.fallback = MPCController()
    self.retrieval_blend = retrieval_blend
    self.smooth = smooth
    self.feedback_p = feedback_p
    self.feedback_d = feedback_d

    self.step_idx = CONTEXT_LENGTH
    self.action_hist = []
    self.error_hist = []
    self.integral_error = 0.0
    self.prev_error = 0.0
    self.prev_action = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    fallback_action = float(self.fallback.update(target_lataccel, current_lataccel, state, future_plan))
    error = float(target_lataccel - current_lataccel)
    self.integral_error = float(np.clip(self.integral_error + error, -40.0, 40.0))

    action = fallback_action
    if self.library is not None and self.step_idx >= CONTROL_START_IDX:
      feature = build_feature(
        target_lataccel, current_lataccel, state, future_plan,
        self.action_hist, self.error_hist, self.integral_error,
      )
      retrieved, confidence, _ = self.library.query(feature)
      retrieved += self.feedback_p * error + self.feedback_d * (error - self.prev_error)
      confidence = float(np.clip(confidence * self.retrieval_blend, 0.0, 1.0))
      raw_action = confidence * retrieved + (1.0 - confidence) * fallback_action
      action = self.smooth * self.prev_action + (1.0 - self.smooth) * raw_action

    action = float(np.clip(action, STEER_MIN, STEER_MAX))
    self._commit_history(action, error)
    self._align_fallback_history(action)
    return action

  def _commit_history(self, action, error):
    self.action_hist.append(action)
    self.error_hist.append(error)
    if len(self.action_hist) > ACTION_HIST_LEN:
      self.action_hist = self.action_hist[-ACTION_HIST_LEN:]
    if len(self.error_hist) > ERROR_HIST_LEN:
      self.error_hist = self.error_hist[-ERROR_HIST_LEN:]
    self.prev_action = action
    self.prev_error = error
    self.step_idx += 1

  def _align_fallback_history(self, action):
    # MPC computed its proposal before retrieval blending. Keep its input
    # history close to the action actually returned by this controller.
    if hasattr(self.fallback, "prev_u"):
      self.fallback.prev_u = action
    if hasattr(self.fallback, "u_hist") and self.fallback.u_hist:
      self.fallback.u_hist[0] = action
