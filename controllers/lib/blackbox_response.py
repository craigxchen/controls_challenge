from pathlib import Path

import numpy as np


DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "blackbox_response_model.npz"

MODEL_VERSION = 2
Y_LAGS = 20
U_LAGS = 20
ROLL_LAGS = 8
V_LAGS = 6
A_LAGS = 6
DIFF_LAGS = 8


def _as_history(values, size, default):
  values = list(values) if values is not None else []
  if not values:
    values = [default]
  out = [float(v) for v in values[:size]]
  while len(out) < size:
    out.append(out[-1])
  return np.asarray(out, dtype=float)


def _diffs(values, size):
  values = np.asarray(values, dtype=float)
  if values.size < 2:
    return np.zeros(size, dtype=float)
  diffs = values[:-1] - values[1:]
  if diffs.size >= size:
    return diffs[:size].astype(float)
  return np.pad(diffs, (0, size - diffs.size), mode="edge").astype(float)


def _as_history_batch(values, size, default):
  if values is None:
    return np.full((1, size), float(default), dtype=float)
  values = np.asarray(values, dtype=float)
  if values.ndim == 0:
    values = values.reshape(1, 1)
  elif values.ndim == 1:
    values = values[None, :]
  if values.shape[1] == 0:
    return np.full((values.shape[0], size), float(default), dtype=float)
  if values.shape[1] >= size:
    return values[:, :size].astype(float, copy=False)
  pad = np.repeat(values[:, -1:], size - values.shape[1], axis=1)
  return np.concatenate([values, pad], axis=1).astype(float, copy=False)


def _diffs_batch(values, size):
  values = np.asarray(values, dtype=float)
  if values.ndim == 1:
    values = values[None, :]
  if values.shape[1] < 2:
    return np.zeros((values.shape[0], size), dtype=float)
  diffs = values[:, :-1] - values[:, 1:]
  if diffs.shape[1] >= size:
    return diffs[:, :size].astype(float, copy=False)
  pad = np.repeat(diffs[:, -1:], size - diffs.shape[1], axis=1)
  return np.concatenate([diffs, pad], axis=1).astype(float, copy=False)


def _quiet_matmul(left, right):
  with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
    return left @ right


def build_raw_features(y_lags, u_lags, roll_lags, v_lags, a_lags, zero_current_action=False):
  y = _as_history(y_lags, Y_LAGS, 0.0)
  u = _as_history(u_lags, U_LAGS, 0.0)
  if zero_current_action:
    u = u.copy()
    u[0] = 0.0
  roll = _as_history(roll_lags, ROLL_LAGS, 0.0)
  v = _as_history(v_lags, V_LAGS, 20.0)
  a = _as_history(a_lags, A_LAGS, 0.0)

  y5 = y[:5]
  u5 = u[:5]
  r5 = roll[:5]
  summary = np.asarray([
    np.mean(y5),
    np.std(y5),
    np.max(np.abs(y5)),
    np.mean(u5),
    np.std(u5),
    np.max(np.abs(u5)),
    np.mean(r5),
    np.std(r5),
    np.max(np.abs(r5)),
    y[0] * roll[0],
    y[0] * v[0] / 30.0,
    roll[0] * v[0] / 30.0,
    u[1] * v[0] / 30.0,
    u[1] * roll[0],
    a[0] * v[0] / 30.0,
  ], dtype=float)

  return np.concatenate([
    y,
    _diffs(y, DIFF_LAGS),
    u,
    _diffs(u, DIFF_LAGS),
    roll,
    _diffs(roll, min(4, DIFF_LAGS)),
    v,
    _diffs(v, min(3, DIFF_LAGS)),
    a,
    _diffs(a, min(3, DIFF_LAGS)),
    summary,
  ]).astype(float)


def build_raw_features_batch(y_lags, u_lags, roll_lags, v_lags, a_lags, zero_current_action=False):
  y = _as_history_batch(y_lags, Y_LAGS, 0.0)
  u = _as_history_batch(u_lags, U_LAGS, 0.0)
  if zero_current_action:
    u = u.copy()
    u[:, 0] = 0.0
  roll = _as_history_batch(roll_lags, ROLL_LAGS, 0.0)
  v = _as_history_batch(v_lags, V_LAGS, 20.0)
  a = _as_history_batch(a_lags, A_LAGS, 0.0)

  rows = max(y.shape[0], u.shape[0], roll.shape[0], v.shape[0], a.shape[0])
  if y.shape[0] == 1 and rows > 1:
    y = np.repeat(y, rows, axis=0)
  if u.shape[0] == 1 and rows > 1:
    u = np.repeat(u, rows, axis=0)
  if roll.shape[0] == 1 and rows > 1:
    roll = np.repeat(roll, rows, axis=0)
  if v.shape[0] == 1 and rows > 1:
    v = np.repeat(v, rows, axis=0)
  if a.shape[0] == 1 and rows > 1:
    a = np.repeat(a, rows, axis=0)

  y5 = y[:, :5]
  u5 = u[:, :5]
  r5 = roll[:, :5]
  summary = np.column_stack([
    np.mean(y5, axis=1),
    np.std(y5, axis=1),
    np.max(np.abs(y5), axis=1),
    np.mean(u5, axis=1),
    np.std(u5, axis=1),
    np.max(np.abs(u5), axis=1),
    np.mean(r5, axis=1),
    np.std(r5, axis=1),
    np.max(np.abs(r5), axis=1),
    y[:, 0] * roll[:, 0],
    y[:, 0] * v[:, 0] / 30.0,
    roll[:, 0] * v[:, 0] / 30.0,
    u[:, 1] * v[:, 0] / 30.0,
    u[:, 1] * roll[:, 0],
    a[:, 0] * v[:, 0] / 30.0,
  ]).astype(float)

  return np.concatenate([
    y,
    _diffs_batch(y, DIFF_LAGS),
    u,
    _diffs_batch(u, DIFF_LAGS),
    roll,
    _diffs_batch(roll, min(4, DIFF_LAGS)),
    v,
    _diffs_batch(v, min(3, DIFF_LAGS)),
    a,
    _diffs_batch(a, min(3, DIFF_LAGS)),
    summary,
  ], axis=1).astype(float)


def _feature_offsets():
  off = {}
  idx = 0
  off["y"] = idx
  idx += Y_LAGS
  off["dy"] = idx
  idx += DIFF_LAGS
  off["u"] = idx
  idx += U_LAGS
  off["du"] = idx
  idx += DIFF_LAGS
  off["roll"] = idx
  idx += ROLL_LAGS
  off["droll"] = idx
  idx += 4
  off["v"] = idx
  idx += V_LAGS
  off["dv"] = idx
  idx += 3
  off["a"] = idx
  idx += A_LAGS
  off["da"] = idx
  idx += 3
  off["summary"] = idx
  return off


def _interaction_pairs(raw_dim):
  off = _feature_offsets()
  pairs = (
    (off["y"], off["roll"]),
    (off["y"], off["v"]),
    (off["roll"], off["v"]),
    (off["u"] + 1, off["v"]),
    (off["u"] + 1, off["roll"]),
    (off["u"] + 1, off["a"]),
    (off["dy"], off["v"]),
    (off["du"] + 1, off["v"]),
    (off["a"], off["v"]),
    (off["summary"] + 2, off["summary"] + 8),
  )
  return tuple((i, j) for i, j in pairs if i < raw_dim and j < raw_dim)


def design_matrix(raw_features, feature_mean, feature_scale, hidden_weight=None, hidden_bias=None):
  raw = np.asarray(raw_features, dtype=float)
  one_dim = raw.ndim == 1
  if one_dim:
    raw = raw[None, :]
  x = (raw - feature_mean) / feature_scale
  x = np.clip(x, -8.0, 8.0)
  pieces = [np.ones((x.shape[0], 1), dtype=float), x, 0.5 * x * x]
  for i, j in _interaction_pairs(x.shape[1]):
    pieces.append((x[:, i] * x[:, j])[:, None])
  if hidden_weight is not None and hidden_bias is not None and hidden_weight.size:
    hidden_linear = _quiet_matmul(x, hidden_weight) + hidden_bias
    hidden = np.tanh(np.nan_to_num(hidden_linear, nan=0.0, posinf=20.0, neginf=-20.0))
    pieces.append(hidden)
  design = np.concatenate(pieces, axis=1)
  return design[0] if one_dim else design


class LinearBlackboxResponse:
  """Versioned NumPy response model for the tinyphysics blackbox."""

  def __init__(self, artifact_path=DEFAULT_ARTIFACT):
    data = np.load(artifact_path)
    self.model_version = int(data.get("model_version", np.array([1], dtype=np.int32))[0])
    self.speed_knots = data["speed_knots"].astype(float)
    self.max_delta = float(data.get("max_delta", np.array([0.5], dtype=np.float32))[0])
    self.lat_min = float(data.get("lat_min", np.array([-5.0], dtype=np.float32))[0])
    self.lat_max = float(data.get("lat_max", np.array([5.0], dtype=np.float32))[0])

    if self.model_version >= 2:
      self.delta_coef = data["delta_coef"].astype(float)
      self.gain_coef = data["gain_coef"].astype(float)
      self.curv_coef = data["curv_coef"].astype(float)
      self.feature_mean = data["feature_mean"].astype(float)
      self.feature_scale = data["feature_scale"].astype(float)
      self.hidden_weight = data.get("hidden_weight", np.zeros((self.feature_mean.size, 0), dtype=np.float32)).astype(float)
      self.hidden_bias = data.get("hidden_bias", np.zeros(0, dtype=np.float32)).astype(float)
      self.y_lags = int(data["y_lags"][0])
      self.u_lags = int(data["u_lags"][0])
      self.roll_lags = int(data["roll_lags"][0])
      self.v_lags = int(data["v_lags"][0])
      self.a_lags = int(data["a_lags"][0])
      self.gain_min = float(data["gain_clip"][0])
      self.gain_max = float(data["gain_clip"][1])
      self.curv_min = float(data["curv_clip"][0])
      self.curv_max = float(data["curv_clip"][1])
    else:
      self.coef = data["coef"].astype(float)
      self.feature_mean = data["feature_mean"].astype(float)
      self.feature_scale = data["feature_scale"].astype(float)
      self.y_lags = int(data["y_lags"][0])
      self.u_lags = int(data["u_lags"][0])
      self.roll_lags = int(data["roll_lags"][0])
      self.v_lags = 1
      self.a_lags = int(data["a_lags"][0])

  @classmethod
  def try_load(cls, artifact_path=DEFAULT_ARTIFACT):
    try:
      path = Path(artifact_path)
      if path.exists():
        return cls(path)
    except Exception:
      return None
    return None

  def predict_next(self, y_lags, u_lags, roll_lags, a_lags, v_ego, v_lags=None, clip_output=False):
    if self.model_version < 2:
      features = self._legacy_feature(y_lags, u_lags, roll_lags, a_lags)
      x = (features - self.feature_mean) / self.feature_scale
      coef = self._legacy_coef(v_ego)
      pred = float(coef[0] + x @ coef[1:])
      if clip_output:
        y0 = _as_history(y_lags, self.y_lags, 0.0)[0]
        pred = self._clip_next(pred, y0)
      return pred

    y = _as_history(y_lags, self.y_lags, 0.0)
    u = _as_history(u_lags, self.u_lags, 0.0)
    v = _as_history(v_lags if v_lags is not None else [v_ego], self.v_lags, v_ego)
    raw = build_raw_features(y, u, roll_lags, v, a_lags, zero_current_action=True)
    pred = self.predict_from_raw(raw, u[0], y[0], v[0], clip_output=clip_output)
    return pred

  def predict_from_raw(self, raw_features, current_action, y0, v_ego, clip_output=False):
    if self.model_version < 2:
      raise RuntimeError("predict_from_raw requires a v2 blackbox response artifact")
    design = self._design(raw_features)
    delta = float(_quiet_matmul(self._coef_at(self.delta_coef, v_ego), design))
    gain = float(_quiet_matmul(self._coef_at(self.gain_coef, v_ego), design))
    curv = float(_quiet_matmul(self._coef_at(self.curv_coef, v_ego), design))
    gain = float(np.clip(gain, self.gain_min, self.gain_max))
    curv = float(np.clip(curv, self.curv_min, self.curv_max))
    u0 = float(current_action)
    pred = float(y0 + delta + gain * u0 + 0.5 * curv * u0 * u0)
    if clip_output:
      pred = self._clip_next(pred, y0)
    return pred

  def predict_gain(self, y_lags, u_lags, roll_lags, a_lags, v_ego, v_lags=None):
    if self.model_version < 2:
      features = self._legacy_feature(y_lags, u_lags, roll_lags, a_lags)
      x = (features - self.feature_mean) / self.feature_scale
      coef = self._legacy_coef(v_ego)
      action_offset = self.y_lags
      return float(coef[1 + action_offset])
    y = _as_history(y_lags, self.y_lags, 0.0)
    u = _as_history(u_lags, self.u_lags, 0.0)
    v = _as_history(v_lags if v_lags is not None else [v_ego], self.v_lags, v_ego)
    raw = build_raw_features(y, u, roll_lags, v, a_lags, zero_current_action=True)
    design = self._design(raw)
    gain = float(_quiet_matmul(self._coef_at(self.gain_coef, v[0]), design))
    return float(np.clip(gain, self.gain_min, self.gain_max))

  def response_matrix(self, y_history, u_history, roll_future, a_future, v_future, horizon,
                      base_actions=None, roll_history=None, a_history=None, v_history=None, fd_eps=0.12):
    horizon = int(horizon)
    if base_actions is None:
      base_actions = np.zeros(horizon, dtype=float)
    else:
      base_actions = np.asarray(base_actions, dtype=float)[:horizon].copy()
      if base_actions.size < horizon:
        base_actions = np.pad(base_actions, (0, horizon - base_actions.size), mode="edge")
    base_actions = np.clip(base_actions, -2.0, 2.0)
    matrix = np.zeros((horizon, horizon), dtype=float)
    if self.model_version >= 2 and horizon > 0:
      action_batch = np.repeat(base_actions[None, :], horizon + 1, axis=0)
      denoms = np.zeros(horizon, dtype=float)
      for col in range(horizon):
        old = action_batch[col + 1, col]
        new = float(np.clip(old + fd_eps, -2.0, 2.0))
        if abs(new - old) < 1e-6:
          new = float(np.clip(old - fd_eps, -2.0, 2.0))
        denoms[col] = new - old
        if abs(denoms[col]) >= 1e-6:
          action_batch[col + 1, col] = new
      rollouts = self.simulate_batch(
        y_history, u_history, roll_future, a_future, v_future, action_batch,
        roll_history=roll_history, a_history=a_history, v_history=v_history,
      )
      free = rollouts[0]
      for col in range(horizon):
        if abs(denoms[col]) >= 1e-6:
          matrix[:, col] = (rollouts[col + 1] - free) / denoms[col]
      return matrix, free

    free = self.simulate(
      y_history, u_history, roll_future, a_future, v_future, base_actions,
      roll_history=roll_history, a_history=a_history, v_history=v_history,
    )
    for col in range(horizon):
      actions = base_actions.copy()
      old = actions[col]
      new = float(np.clip(old + fd_eps, -2.0, 2.0))
      if abs(new - old) < 1e-6:
        new = float(np.clip(old - fd_eps, -2.0, 2.0))
      denom = new - old
      if abs(denom) < 1e-6:
        continue
      actions[col] = new
      pert = self.simulate(
        y_history, u_history, roll_future, a_future, v_future, actions,
        roll_history=roll_history, a_history=a_history, v_history=v_history,
      )
      matrix[:, col] = (pert - free) / denom
    return matrix, free

  def simulate(self, y_history, u_history, roll_future, a_future, v_future, actions,
               roll_history=None, a_history=None, v_history=None):
    if self.model_version >= 2:
      return self.simulate_batch(
        y_history, u_history, roll_future, a_future, v_future, np.asarray(actions, dtype=float)[None, :],
        roll_history=roll_history, a_history=a_history, v_history=v_history,
      )[0]

    actions = np.asarray(actions, dtype=float)
    horizon = int(actions.size)
    y_hist = _as_history(y_history, self.y_lags, 0.0)
    u_past = _as_history(u_history, max(self.u_lags - 1, 1), 0.0)
    roll_future = np.asarray(roll_future, dtype=float)
    a_future = np.asarray(a_future, dtype=float)
    v_future = np.asarray(v_future, dtype=float)
    roll_history = _as_history(roll_history, self.roll_lags, roll_future[0] if roll_future.size else 0.0)
    a_history = _as_history(a_history, self.a_lags, a_future[0] if a_future.size else 0.0)
    v_history = _as_history(v_history, self.v_lags, v_future[0] if v_future.size else 20.0)
    out = np.zeros(horizon, dtype=float)

    for step in range(horizon):
      u_lags = [float(actions[step])]
      for lag in range(1, self.u_lags):
        prev_step = step - lag
        if prev_step >= 0:
          u_lags.append(float(actions[prev_step]))
        else:
          idx = lag - step - 1
          u_lags.append(float(u_past[idx] if idx < len(u_past) else u_past[-1]))

      roll_lags = self._series_lags(roll_future, roll_history, step, self.roll_lags)
      a_lags = self._series_lags(a_future, a_history, step, self.a_lags)
      v_lags = self._series_lags(v_future, v_history, step, self.v_lags)
      pred = self.predict_next(y_hist, u_lags, roll_lags, a_lags, v_lags[0], v_lags=v_lags, clip_output=True)
      out[step] = pred
      y_hist = np.concatenate([[pred], y_hist[:-1]])
    return out

  def simulate_batch(self, y_history, u_history, roll_future, a_future, v_future, action_batch,
                     roll_history=None, a_history=None, v_history=None):
    if self.model_version < 2:
      rows = [
        self.simulate(
          y_history, u_history, roll_future, a_future, v_future, actions,
          roll_history=roll_history, a_history=a_history, v_history=v_history,
        )
        for actions in np.asarray(action_batch, dtype=float)
      ]
      return np.asarray(rows, dtype=float)

    actions = np.asarray(action_batch, dtype=float)
    if actions.ndim == 1:
      actions = actions[None, :]
    batch, horizon = actions.shape
    y_hist = np.repeat(_as_history(y_history, self.y_lags, 0.0)[None, :], batch, axis=0)
    u_past = _as_history(u_history, max(self.u_lags - 1, 1), 0.0)
    roll_future = np.asarray(roll_future, dtype=float)
    a_future = np.asarray(a_future, dtype=float)
    v_future = np.asarray(v_future, dtype=float)
    roll_history = _as_history(roll_history, self.roll_lags, roll_future[0] if roll_future.size else 0.0)
    a_history = _as_history(a_history, self.a_lags, a_future[0] if a_future.size else 0.0)
    v_history = _as_history(v_history, self.v_lags, v_future[0] if v_future.size else 20.0)
    out = np.zeros((batch, horizon), dtype=float)

    for step in range(horizon):
      u_lags = np.empty((batch, self.u_lags), dtype=float)
      u_lags[:, 0] = actions[:, step]
      for lag in range(1, self.u_lags):
        prev_step = step - lag
        if prev_step >= 0:
          u_lags[:, lag] = actions[:, prev_step]
        else:
          idx = lag - step - 1
          u_lags[:, lag] = float(u_past[idx] if idx < len(u_past) else u_past[-1])

      roll_lags = self._series_lags(roll_future, roll_history, step, self.roll_lags)
      a_lags = self._series_lags(a_future, a_history, step, self.a_lags)
      v_lags = self._series_lags(v_future, v_history, step, self.v_lags)
      raw = build_raw_features_batch(
        y_hist,
        u_lags,
        np.asarray(roll_lags, dtype=float),
        np.asarray(v_lags, dtype=float),
        np.asarray(a_lags, dtype=float),
        zero_current_action=True,
      )
      design = self._design(raw)
      v0 = float(v_lags[0])
      delta = _quiet_matmul(design, self._coef_at(self.delta_coef, v0))
      gain = _quiet_matmul(design, self._coef_at(self.gain_coef, v0))
      curv = _quiet_matmul(design, self._coef_at(self.curv_coef, v0))
      delta = np.nan_to_num(delta, nan=0.0, posinf=self.max_delta, neginf=-self.max_delta)
      gain = np.clip(np.nan_to_num(gain, nan=0.0, posinf=self.gain_max, neginf=self.gain_min),
                     self.gain_min, self.gain_max)
      curv = np.clip(np.nan_to_num(curv, nan=0.0, posinf=self.curv_max, neginf=self.curv_min),
                     self.curv_min, self.curv_max)
      u0 = actions[:, step]
      pred = y_hist[:, 0] + delta + gain * u0 + 0.5 * curv * u0 * u0
      pred = np.clip(pred, y_hist[:, 0] - self.max_delta, y_hist[:, 0] + self.max_delta)
      pred = np.clip(pred, self.lat_min, self.lat_max)
      out[:, step] = pred
      y_hist = np.concatenate([pred[:, None], y_hist[:, :-1]], axis=1)
    return out

  def _coef_at(self, coef_table, v_ego):
    v = float(v_ego)
    if v <= self.speed_knots[0]:
      return coef_table[0]
    if v >= self.speed_knots[-1]:
      return coef_table[-1]
    hi = int(np.searchsorted(self.speed_knots, v, side="right"))
    lo = hi - 1
    alpha = (v - self.speed_knots[lo]) / (self.speed_knots[hi] - self.speed_knots[lo])
    return (1.0 - alpha) * coef_table[lo] + alpha * coef_table[hi]

  def _design(self, raw_features):
    return design_matrix(
      raw_features,
      self.feature_mean,
      self.feature_scale,
      hidden_weight=self.hidden_weight,
      hidden_bias=self.hidden_bias,
    )

  def _legacy_coef(self, v_ego):
    v = float(v_ego)
    if v <= self.speed_knots[0]:
      return self.coef[0]
    if v >= self.speed_knots[-1]:
      return self.coef[-1]
    hi = int(np.searchsorted(self.speed_knots, v, side="right"))
    lo = hi - 1
    alpha = (v - self.speed_knots[lo]) / (self.speed_knots[hi] - self.speed_knots[lo])
    return (1.0 - alpha) * self.coef[lo] + alpha * self.coef[hi]

  def _legacy_feature(self, y_lags, u_lags, roll_lags, a_lags):
    return np.asarray(
      list(_as_history(y_lags, self.y_lags, 0.0))
      + list(_as_history(u_lags, self.u_lags, 0.0))
      + list(_as_history(roll_lags, self.roll_lags, 0.0))
      + list(_as_history(a_lags, self.a_lags, 0.0)),
      dtype=float,
    )

  def _clip_next(self, pred, current):
    pred = float(np.clip(pred, current - self.max_delta, current + self.max_delta))
    return float(np.clip(pred, self.lat_min, self.lat_max))

  @staticmethod
  def _series_lags(future, history, step, size):
    future = np.asarray(future, dtype=float)
    history = np.asarray(history, dtype=float)
    out = []
    for lag in range(size):
      idx = step - lag
      if idx >= 0:
        if future.size:
          out.append(float(future[min(idx, future.size - 1)]))
        else:
          out.append(float(history[0] if history.size else 0.0))
      else:
        hidx = -idx - 1
        out.append(float(history[min(hidx, max(0, history.size - 1))] if history.size else 0.0))
    return out
