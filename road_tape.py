from dataclasses import dataclass
from pathlib import Path

import numpy as np


ACC_G = 9.81
DEFAULT_DT = 0.1
DEFAULT_OFFSETS = (0, 1, 2, 3, 5, 8, 13, 21, 34, 49)
EPS = 1e-6


def _as_float_array(values):
  return np.asarray(values, dtype=np.float32)


def _gradient(values, dt):
  values = _as_float_array(values)
  if values.size < 2:
    return np.zeros_like(values)
  return np.gradient(values, dt).astype(np.float32)


def _safe_curvature(lataccel, v_ego):
  speed_sq = np.maximum(_as_float_array(v_ego) ** 2, 1.0)
  return (_as_float_array(lataccel) / speed_sq).astype(np.float32)


def _clip_window(signal, start, horizon):
  signal = _as_float_array(signal)
  if signal.size == 0:
    return np.zeros(horizon, dtype=np.float32)
  start = int(max(0, start))
  end = min(signal.size, start + horizon)
  window = signal[start:end]
  if window.size < horizon:
    pad_value = window[-1] if window.size else signal[-1]
    window = np.pad(window, (0, horizon - window.size), constant_values=float(pad_value))
  return window.astype(np.float32)


def _lagged_corr(a, b, max_lag=10):
  a = _as_float_array(a)
  b = _as_float_array(b)
  if a.size < 4 or b.size < 4:
    return 0.0, 0.0
  n = min(a.size, b.size)
  a = a[:n]
  b = b[:n]
  best_corr = 0.0
  best_lag = 0
  for lag in range(-max_lag, max_lag + 1):
    if lag < 0:
      aa, bb = a[-lag:], b[:n + lag]
    elif lag > 0:
      aa, bb = a[:n - lag], b[lag:]
    else:
      aa, bb = a, b
    if aa.size < 4 or np.std(aa) < EPS or np.std(bb) < EPS:
      corr = 0.0
    else:
      corr = float(np.corrcoef(aa, bb)[0, 1])
    if abs(corr) > abs(best_corr):
      best_corr = corr
      best_lag = lag
  return float(best_corr), float(best_lag)


def _event_summary(window, dt):
  window = _as_float_array(window)
  if window.size == 0:
    return np.zeros(9, dtype=np.float32)

  abs_window = np.abs(window)
  peak_idx = int(np.argmax(abs_window))
  pos_idx = int(np.argmax(window))
  neg_idx = int(np.argmin(window))
  roughness = float(np.mean(np.abs(np.diff(window, n=2)))) if window.size > 2 else 0.0
  return np.asarray([
    float(window[0]),
    float(window[-1]),
    float(np.mean(window)),
    float(np.mean(abs_window)),
    float(window[peak_idx]),
    float(peak_idx * dt),
    float(pos_idx * dt),
    float(neg_idx * dt),
    roughness,
  ], dtype=np.float32)


@dataclass
class RoadTape:
  target_lataccel: np.ndarray
  roll_lataccel: np.ndarray
  v_ego: np.ndarray
  a_ego: np.ndarray
  dt: float = DEFAULT_DT

  def __post_init__(self):
    self.target_lataccel = _as_float_array(self.target_lataccel)
    self.roll_lataccel = _as_float_array(self.roll_lataccel)
    self.v_ego = _as_float_array(self.v_ego)
    self.a_ego = _as_float_array(self.a_ego)
    self._validate_lengths()

  def _validate_lengths(self):
    lengths = {
      self.target_lataccel.size,
      self.roll_lataccel.size,
      self.v_ego.size,
      self.a_ego.size,
    }
    if len(lengths) != 1:
      raise ValueError("RoadTape channels must have matching lengths")

  @property
  def size(self):
    return int(self.target_lataccel.size)

  @property
  def steer_demand_proxy(self):
    return (self.target_lataccel - self.roll_lataccel).astype(np.float32)

  @property
  def channel_names(self):
    return (
      "target_lataccel",
      "roll_lataccel",
      "steer_demand_proxy",
      "v_ego",
      "a_ego",
      "target_curvature_proxy",
      "roll_curvature_proxy",
      "demand_curvature_proxy",
      "target_slope",
      "roll_slope",
      "demand_slope",
      "v_slope",
    )

  def matrix(self):
    demand = self.steer_demand_proxy
    return np.column_stack([
      self.target_lataccel,
      self.roll_lataccel,
      demand,
      self.v_ego,
      self.a_ego,
      _safe_curvature(self.target_lataccel, self.v_ego),
      _safe_curvature(self.roll_lataccel, self.v_ego),
      _safe_curvature(demand, self.v_ego),
      _gradient(self.target_lataccel, self.dt),
      _gradient(self.roll_lataccel, self.dt),
      _gradient(demand, self.dt),
      _gradient(self.v_ego, self.dt),
    ]).astype(np.float32)

  def window(self, start_idx, horizon=50):
    start_idx = int(start_idx)
    return {
      "target_lataccel": _clip_window(self.target_lataccel, start_idx, horizon),
      "roll_lataccel": _clip_window(self.roll_lataccel, start_idx, horizon),
      "steer_demand_proxy": _clip_window(self.steer_demand_proxy, start_idx, horizon),
      "v_ego": _clip_window(self.v_ego, start_idx, horizon),
      "a_ego": _clip_window(self.a_ego, start_idx, horizon),
    }

  def feature_vector(self, start_idx, horizon=50, offsets=DEFAULT_OFFSETS):
    window = self.window(start_idx, horizon)
    offsets = np.asarray(offsets, dtype=np.int32)
    offsets = np.clip(offsets, 0, horizon - 1)

    pieces = []
    for name in ("target_lataccel", "roll_lataccel", "steer_demand_proxy", "v_ego"):
      pieces.append(window[name][offsets])

    for name in ("target_lataccel", "roll_lataccel", "steer_demand_proxy", "v_ego"):
      pieces.append(_event_summary(window[name], self.dt))

    corr, lag = _lagged_corr(window["target_lataccel"], window["roll_lataccel"])
    pieces.append(np.asarray([corr, lag * self.dt], dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32)


def build_road_tape_from_arrays(target_lataccel, roll_lataccel, v_ego, a_ego, dt=DEFAULT_DT):
  return RoadTape(
    target_lataccel=target_lataccel,
    roll_lataccel=roll_lataccel,
    v_ego=v_ego,
    a_ego=a_ego,
    dt=dt,
  )


def build_road_tape_from_dataframe(df, dt=DEFAULT_DT):
  roll_lataccel = np.sin(df["roll"].to_numpy(dtype=np.float32)) * ACC_G
  return build_road_tape_from_arrays(
    target_lataccel=df["targetLateralAcceleration"].to_numpy(dtype=np.float32),
    roll_lataccel=roll_lataccel.astype(np.float32),
    v_ego=df["vEgo"].to_numpy(dtype=np.float32),
    a_ego=df["aEgo"].to_numpy(dtype=np.float32),
    dt=dt,
  )


def build_road_tape_from_csv(csv_path, dt=DEFAULT_DT):
  import pandas as pd

  return build_road_tape_from_dataframe(pd.read_csv(Path(csv_path)), dt=dt)


class OnlineRoadTape:
  """
  Reconstructs the same road-tape channels from the controller API.

  The full CSV helper is useful for training and diagnostics. At runtime, this
  class builds the visible prefix of that tape from the current state plus the
  rolling future_plan without using data paths or segment IDs.
  """

  def __init__(self, dt=DEFAULT_DT, start_idx=0):
    self.dt = dt
    self.step_idx = int(start_idx)
    self.target = []
    self.roll = []
    self.v_ego = []
    self.a_ego = []

  def update(self, target_lataccel, state, future_plan):
    self._set_value(self.target, self.step_idx, target_lataccel)
    self._set_value(self.roll, self.step_idx, state.roll_lataccel)
    self._set_value(self.v_ego, self.step_idx, state.v_ego)
    self._set_value(self.a_ego, self.step_idx, state.a_ego)

    if future_plan is None:
      self.step_idx += 1
      return
    self._merge_future(self.target, future_plan.lataccel)
    self._merge_future(self.roll, future_plan.roll_lataccel)
    self._merge_future(self.v_ego, future_plan.v_ego)
    self._merge_future(self.a_ego, future_plan.a_ego)
    self.step_idx += 1

  def _merge_future(self, dst, values):
    if values is None:
      return
    for offset, value in enumerate(values, start=self.step_idx + 1):
      self._set_value(dst, offset, value)

  @staticmethod
  def _set_value(dst, idx, value):
    idx = int(idx)
    if idx < 0 or not np.isfinite(value):
      return
    if len(dst) <= idx:
      dst.extend([float("nan")] * (idx + 1 - len(dst)))
    dst[idx] = float(value)

  def tape(self):
    target = self._finite_filled(self.target)
    roll = self._finite_filled(self.roll)
    v_ego = self._finite_filled(self.v_ego)
    a_ego = self._finite_filled(self.a_ego)
    n = min(target.size, roll.size, v_ego.size, a_ego.size)
    return build_road_tape_from_arrays(target[:n], roll[:n], v_ego[:n], a_ego[:n], dt=self.dt)

  def feature_vector(self, start_idx=None, horizon=50, offsets=DEFAULT_OFFSETS):
    if start_idx is None:
      start_idx = self.step_idx
    return self.tape().feature_vector(start_idx, horizon=horizon, offsets=offsets)

  @staticmethod
  def _finite_filled(values):
    arr = _as_float_array(values)
    if arr.size == 0:
      return arr
    finite = np.isfinite(arr)
    if np.all(finite):
      return arr
    if not np.any(finite):
      return np.zeros_like(arr)
    first = int(np.flatnonzero(finite)[0])
    arr[:first] = arr[first]
    for idx in range(first + 1, arr.size):
      if not np.isfinite(arr[idx]):
        arr[idx] = arr[idx - 1]
    return arr
