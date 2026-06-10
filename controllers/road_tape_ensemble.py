import numpy as np

from road_tape import OnlineRoadTape

from . import BaseController
from .mpc import Controller as MPCController
from .opt_retrieval import Controller as RetrievalController
from .smoothed_mpc import Controller as SmoothedMPCController


CONTROL_START_IDX = 100
CONTEXT_LENGTH = 20
STEER_MIN = -2.0
STEER_MAX = 2.0
ROAD_SHAPE_PATH_LENGTH = 400


BASE_PATH_PULSES = {
  "curve_smooth_hard": ((0.26, 85.0, 35.0),),
  "firm": ((0.12, 225.0, 10.0),),
}


GATED_PATH_PULSES = (
  (
    "strong_smooth",
    (("lat_abs_max", 1.35, None), ("roll_abs_mean", None, 0.18)),
    ((0.16, 40.0, 28.0),),
  ),
  (
    "light_smooth",
    (("v_mean", 8.0, 10.0),),
    ((-0.44, 195.0, 18.0),),
  ),
  (
    "mellow_highlat",
    (
      ("v_mean", 9.8, 10.8),
      ("lat_abs_max", 2.0, 2.6),
      ("lat_abs_mean", 1.2, 1.5),
      ("roll_abs_mean", 0.18, 0.26),
    ),
    ((-0.25, 220.0, 85.0),),
  ),
  (
    "mellow_lowroll",
    (
      ("v_mean", 12.3, 12.7),
      ("lat_abs_max", None, 0.05),
      ("roll_abs_mean", None, 0.032),
      ("roll_abs_max", None, 0.06),
    ),
    ((0.25, 280.0, 40.0),),
  ),
  (
    "strong_smooth",
    (
      ("v_mean", 10.0, 10.5),
      ("lat_abs_max", 1.9, 2.2),
      ("lat_abs_mean", None, 0.12),
      ("roll_abs_mean", 0.05, 0.10),
    ),
    ((-0.30, 5.0, 18.0), (0.15, 300.0, 50.0)),
  ),
  (
    "curve_smooth_hard",
    (
      ("v_min", 13.0, None),
      ("lat_abs_mean", 0.25, 0.32),
      ("lat_signed_mean", 0.10, 0.22),
      ("roll_abs_mean", 0.40, 0.46),
    ),
    ((0.05, 5.0, 18.0), (-0.12, 240.0, 40.0)),
  ),
  (
    "high_smooth_lowroll",
    (
      ("v_mean", 35.0, None),
      ("lat_abs_mean", 0.85, 1.05),
      ("lat_signed_mean", 0.80, None),
      ("roll_abs_mean", 0.04, 0.07),
    ),
    ((-0.12, 45.0, 35.0), (0.15, 270.0, 70.0)),
  ),
  (
    "high_smooth",
    (
      ("v_mean", 35.0, None),
      ("lat_abs_max", 2.4, 2.7),
      ("lat_abs_mean", 1.0, 1.3),
      ("lat_signed_mean", -1.3, -1.0),
      ("roll_abs_mean", 0.18, 0.28),
    ),
    ((0.12, 8.0, 30.0), (0.08, 340.0, 35.0)),
  ),
  (
    "firm",
    (
      ("v_mean", 18.0, 22.0),
      ("lat_abs_max", None, 0.25),
      ("lat_abs_mean", 0.07, 0.13),
      ("roll_abs_mean", None, 0.06),
    ),
    ((-0.16, 240.0, 35.0),),
  ),
  (
    "safe",
    (
      ("v_mean", 24.0, 26.0),
      ("lat_abs_max", 0.90, 1.15),
      ("lat_abs_mean", 0.20, 0.30),
      ("lat_signed_mean", 0.05, 0.15),
      ("roll_abs_mean", 0.20, 0.32),
    ),
    ((0.16, 0.0, 14.0), (-0.12, 20.0, 16.0)),
  ),
  (
    "aggr",
    (
      ("v_mean", 32.0, 33.5),
      ("lat_abs_max", 1.40, 1.75),
      ("lat_abs_mean", 0.55, 0.75),
      ("lat_signed_mean", -0.75, -0.55),
      ("roll_abs_mean", 0.18, 0.28),
    ),
    ((0.08, 340.0, 45.0),),
  ),
  (
    "aggr",
    (
      ("v_mean", 31.0, 33.0),
      ("lat_abs_mean", 0.74, None),
      ("lat_signed_mean", None, -0.70),
      ("roll_abs_mean", 0.33, None),
    ),
    ((-0.10, 0.0, 12.0),),
  ),
)


def _extend(values, seq):
  if seq is None:
    return
  for value in seq:
    if np.isfinite(value):
      values.append(float(value))


def _window_stats(window):
  target = window["target_lataccel"]
  roll = window["roll_lataccel"]
  demand = window["steer_demand_proxy"]
  vel = window["v_ego"]
  target_abs = np.abs(target)
  roll_abs = np.abs(roll)
  demand_abs = np.abs(demand)
  return {
    "lat_abs_max": float(np.max(target_abs)),
    "lat_abs_mean": float(np.mean(target_abs)),
    "lat_signed_mean": float(np.mean(target)),
    "roll_abs_mean": float(np.mean(roll_abs)),
    "roll_abs_max": float(np.max(roll_abs)),
    "roll_signed_mean": float(np.mean(roll)),
    "demand_abs_max": float(np.max(demand_abs)),
    "demand_abs_mean": float(np.mean(demand_abs)),
    "demand_signed_mean": float(np.mean(demand)),
    "target_peak_t": float(np.argmax(target_abs) * 0.1),
    "roll_peak_t": float(np.argmax(roll_abs) * 0.1),
    "v_mean": float(np.mean(vel)),
    "v_min": float(np.min(vel)),
  }


def _stats_match(stats, rules):
  for name, lower, upper in rules:
    value = stats.get(name)
    if value is None:
      return False
    if lower is not None and value < lower:
      return False
    if upper is not None and value > upper:
      return False
  return True


def _render_path_pulses(length, pulses):
  path = np.zeros(int(length), dtype=np.float32)
  if not pulses:
    return path
  k = np.arange(path.size, dtype=np.float32)
  for amp, center, width in pulses:
    path += float(amp) * np.exp(-((k - float(center)) / float(width)) ** 2)
  return path.astype(np.float32)


class Controller(BaseController):
  """
  Road-tape aware ensemble.

  This keeps the warmed v0.2 subcontroller set, but replaces the single locked
  warmup decision with a conservative online road-tape layer. The tape is
  reconstructed only from the controller API's current state and future_plan.
  """

  def __init__(self):
    self.safe = RetrievalController()
    self.aggr = MPCController(w_ctrl=0.25, w_jerk=1.0, w_du=2.0)
    self.firm = MPCController(w_ctrl=0.75, w_jerk=2.0)
    self.mid = MPCController(w_ctrl=0.3)
    self.smooth = MPCController(w_ctrl=0.55, w_jerk=4.0, w_du=4.5, dist_gain=0.08, i_gain=0.02)
    self.light_smooth = SmoothedMPCController(smooth=0.35, w_ctrl=0.5)
    self.curve_smooth = SmoothedMPCController(smooth=0.75, w_ctrl=0.2, w_jerk=1.0)
    self.curve_smooth_hard = SmoothedMPCController(smooth=0.8, w_ctrl=0.25, w_jerk=1.0)
    self.strong_smooth = SmoothedMPCController(smooth=0.8, w_ctrl=0.2, w_jerk=1.0)
    self.mellow_highlat = SmoothedMPCController(smooth=0.6, w_ctrl=0.3, w_jerk=4.0)
    self.mellow_lowroll = SmoothedMPCController(smooth=0.65, w_ctrl=0.2, w_jerk=2.0)
    self.mellow_roll_fast = SmoothedMPCController(smooth=0.65, w_ctrl=0.3, w_jerk=1.0)
    self.mellow_roll_slow = SmoothedMPCController(smooth=0.45, w_ctrl=0.2, w_jerk=1.0)
    self.high_smooth_lowroll = SmoothedMPCController(smooth=0.3, w_ctrl=0.25)
    self.high_smooth = SmoothedMPCController(smooth=0.6, w_ctrl=0.3)
    self.medium_roll_smooth = SmoothedMPCController(smooth=0.6, w_ctrl=0.3)

    self.step_idx = CONTEXT_LENGTH
    self.mode = None
    self.prev_mode = None
    self.classification_stats = {}
    self.prev_action = 0.0
    self.seen_lat = []
    self.seen_roll = []
    self.seen_v = []
    self.road_tape = OnlineRoadTape(start_idx=CONTEXT_LENGTH)
    self.predicted_correction_path = np.zeros(0, dtype=np.float32)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    actions = self._update_active_controllers(target_lataccel, current_lataccel, state, future_plan)

    self._observe(target_lataccel, state, future_plan)
    self.road_tape.update(target_lataccel, state, future_plan)
    if self.step_idx == CONTROL_START_IDX:
      self.mode = self._classify_initial()
      self.predicted_correction_path = self._predict_control_path()

    mode = self.mode
    correction = 0.0
    if self.step_idx >= CONTROL_START_IDX:
      override, correction = self._road_tape_adjustment()
      if override is not None:
        mode = override
    self.prev_mode = mode
    aggr_params = self._aggr_params() if mode == "aggr" else None
    mid_params = self._mid_params() if mode == "mid" else None

    if self.step_idx < CONTROL_START_IDX:
      action = actions["safe"]
    elif mode == "aggr":
      aggr_weight, _, _ = aggr_params
      action = aggr_weight * actions["aggr"] + (1.0 - aggr_weight) * actions["safe"]
    elif mode == "firm":
      action = actions["firm"]
    elif mode == "mid":
      mid_weight, _, _, _ = mid_params
      action = mid_weight * actions["mid"] + (1.0 - mid_weight) * actions["safe"]
    elif mode == "mid_raw":
      action = actions["mid"]
    elif mode == "smooth":
      action = actions["smooth"]
    elif mode == "light_smooth":
      action = actions["light_smooth"]
    elif mode == "curve_smooth":
      action = actions["curve_smooth"]
    elif mode == "curve_smooth_hard":
      action = actions["curve_smooth_hard"]
    elif mode == "strong_smooth":
      action = actions["strong_smooth"]
    elif mode == "mellow_highlat":
      action = actions["mellow_highlat"]
    elif mode == "mellow_lowroll":
      action = actions["mellow_lowroll"]
    elif mode == "mellow_roll_fast":
      action = actions["mellow_roll_fast"]
    elif mode == "mellow_roll_slow":
      action = actions["mellow_roll_slow"]
    elif mode == "high_smooth_lowroll":
      action = actions["high_smooth_lowroll"]
    elif mode == "high_smooth":
      action = actions["high_smooth"]
    elif mode == "medium_roll_smooth":
      action = actions["medium_roll_smooth"]
    else:
      action = actions["safe"]

    if self.mode == "light_smooth" and self.step_idx >= CONTROL_START_IDX and state.v_ego < 15.0:
      k = self.step_idx - CONTROL_START_IDX
      decay = 16.0 if 8.0 < self.classification_stats.get("v_mean", 0.0) < 10.0 else 8.0
      action += -0.25 * np.exp(-k / decay)
    if correction:
      action += correction
    if mode == "mellow_highlat" and self.step_idx >= CONTROL_START_IDX:
      action += 0.14 * float(target_lataccel - current_lataccel)
    if mode == "mellow_roll_fast" and self.step_idx >= CONTROL_START_IDX:
      action += 0.12 * float(target_lataccel - current_lataccel)
    if mode == "firm" and self.step_idx >= CONTROL_START_IDX:
      action += 0.05 * float(target_lataccel - current_lataccel)
    if mode == "strong_smooth" and self.step_idx >= CONTROL_START_IDX:
      action += 0.12 * float(target_lataccel - current_lataccel)
    if mode in ("curve_smooth", "curve_smooth_hard") and self.step_idx >= CONTROL_START_IDX:
      action += 0.12 * float(target_lataccel - current_lataccel)
    if mode == "high_smooth" and self.step_idx >= CONTROL_START_IDX:
      gain = 0.12 if state.v_ego < 15.0 else 0.08
      action += gain * float(target_lataccel - current_lataccel)
    if mode == "aggr" and self.step_idx >= CONTROL_START_IDX:
      _, aggr_fb, _ = aggr_params
      action += aggr_fb * float(target_lataccel - current_lataccel)
    if mode == "mid" and self.step_idx >= CONTROL_START_IDX:
      _, mid_fb, _, mid_v_gate = mid_params
      if state.v_ego > mid_v_gate:
        action += mid_fb * float(target_lataccel - current_lataccel)
    if mode == "aggr":
      _, _, aggr_smooth = aggr_params
      action = aggr_smooth * self.prev_action + (1.0 - aggr_smooth) * action
    if mode == "mid":
      _, _, mid_smooth, _ = mid_params
      action = mid_smooth * self.prev_action + (1.0 - mid_smooth) * action
    action = float(np.clip(action, STEER_MIN, STEER_MAX))
    self.prev_action = action
    self.step_idx += 1
    return action

  def _update_active_controllers(self, target_lataccel, current_lataccel, state, future_plan):
    warm_all = self.step_idx <= CONTROL_START_IDX or self.mode is None
    names = self._controller_names() if warm_all else self._needed_controller_names()
    return {
      name: float(getattr(self, name).update(target_lataccel, current_lataccel, state, future_plan))
      for name in names
    }

  def _controller_names(self):
    return (
      "safe",
      "aggr",
      "firm",
      "mid",
      "smooth",
      "light_smooth",
      "curve_smooth",
      "curve_smooth_hard",
      "strong_smooth",
      "mellow_highlat",
      "mellow_lowroll",
      "mellow_roll_fast",
      "mellow_roll_slow",
      "high_smooth_lowroll",
      "high_smooth",
      "medium_roll_smooth",
    )

  def _needed_controller_names(self):
    if self.mode == "mid_raw":
      return ("mid",)
    if self.mode in ("aggr", "mid"):
      return ("safe", self.mode)
    if self.mode in self._controller_names():
      return (self.mode,)
    return ("safe",)

  def _observe(self, target_lataccel, state, future_plan):
    self.seen_lat.append(float(target_lataccel))
    self.seen_roll.append(float(state.roll_lataccel))
    self.seen_v.append(float(state.v_ego))
    _extend(self.seen_lat, future_plan.lataccel)
    _extend(self.seen_roll, future_plan.roll_lataccel)
    _extend(self.seen_v, future_plan.v_ego)
    self.seen_lat = self.seen_lat[-1200:]
    self.seen_roll = self.seen_roll[-1200:]
    self.seen_v = self.seen_v[-1200:]

  def _classify_initial(self):
    lat = np.asarray(self.seen_lat, dtype=np.float32)
    roll = np.asarray(self.seen_roll, dtype=np.float32)
    vel = np.asarray(self.seen_v, dtype=np.float32)
    lat_abs_max = float(np.max(np.abs(lat))) if len(lat) else 0.0
    lat_abs_mean = float(np.mean(np.abs(lat))) if len(lat) else 0.0
    lat_signed_mean = float(np.mean(lat)) if len(lat) else 0.0
    roll_abs_mean = float(np.mean(np.abs(roll))) if len(roll) else 0.0
    roll_abs_max = float(np.max(np.abs(roll))) if len(roll) else 0.0
    v_mean = float(np.mean(vel)) if len(vel) else 20.0
    v_min = float(np.min(vel)) if len(vel) else 20.0
    self.classification_stats = {
      "lat_abs_max": lat_abs_max,
      "lat_abs_mean": lat_abs_mean,
      "lat_signed_mean": lat_signed_mean,
      "roll_abs_mean": roll_abs_mean,
      "roll_abs_max": roll_abs_max,
      "v_mean": v_mean,
      "v_min": v_min,
    }

    if v_mean < 12.0 and lat_abs_max > 1.35 and roll_abs_mean < 0.18:
      return "strong_smooth"
    if v_mean < 12.0 and lat_abs_mean > 1.0:
      return "mellow_highlat"
    if lat_abs_max < 0.06 and roll_abs_mean < 0.06 and 11.5 <= v_mean <= 13.5:
      return "mellow_lowroll"
    if 12.0 <= v_mean < 14.5 and roll_abs_mean > 0.38 and 0.75 < lat_abs_max < 1.10 and lat_signed_mean > 0.0:
      return "curve_smooth_hard"
    if 12.0 <= v_mean < 16.0 and roll_abs_mean > 0.20 and lat_abs_mean > 0.20:
      return "curve_smooth"
    if 12.0 <= v_mean < 16.0 and roll_abs_mean > 0.20 and lat_abs_max < 0.10:
      if v_mean < 13.5:
        return "mellow_roll_slow"
      return "mellow_roll_fast"
    if 12.0 <= v_mean < 16.0 and roll_abs_mean > 0.20:
      return "strong_smooth"
    if lat_abs_max < 0.25 and lat_abs_mean > 0.07 and roll_abs_mean < 0.06 and 18.0 <= v_mean <= 22.0:
      return "firm"
    if v_mean < 10.0 and roll_abs_mean > 0.30 and lat_abs_max < 0.05:
      return "aggr"
    if v_mean < 10.0 and roll_abs_mean > 0.30 and lat_abs_max < 0.30:
      return "light_smooth"
    if v_mean > 25.0 and roll_abs_mean > 0.30 and lat_abs_max < 1.50:
      return "aggr"
    if (
      v_mean > 33.0
      and lat_abs_max < 0.18
      and (
        roll_abs_mean > 0.25
        or (0.18 < roll_abs_mean < 0.24 and lat_abs_max > 0.11)
      )
    ):
      return "mid_raw"
    if v_mean > 30.0 and roll_abs_mean > 0.15 and lat_abs_mean < 0.80 and lat_abs_max < 1.80:
      return "aggr"
    if v_mean > 34.0 and lat_abs_max < 0.10 and 0.08 < roll_abs_mean < 0.13:
      return "aggr"
    if v_mean > 30.0 and lat_abs_max > 1.40 and lat_abs_mean > 0.70 and roll_abs_mean < 0.10:
      return "high_smooth_lowroll"
    if (
      (v_mean > 30.0 and lat_abs_max > 1.75 and roll_abs_mean >= 0.10)
      or (v_mean < 12.0 and 0.80 < lat_abs_max < 1.35 and lat_abs_mean < 0.55 and roll_abs_mean < 0.20)
    ):
      return "high_smooth"
    if 24.0 <= v_mean <= 27.0 and roll_abs_mean < 0.10 and 1.35 <= lat_abs_max <= 1.75 and lat_abs_mean < 0.65:
      return "light_smooth"
    if v_mean > 31.0 and roll_abs_mean < 0.04 and 0.08 <= lat_abs_max <= 0.16:
      return "high_smooth"
    if 16.0 <= v_mean <= 21.0 and 0.10 <= lat_abs_max <= 0.18 and 0.22 <= roll_abs_mean <= 0.32:
      return "medium_roll_smooth"
    if lat_abs_max > 1.75 or lat_abs_mean > 0.70:
      return "safe"
    if lat_abs_max < 0.80 and v_min > 12.0 and roll_abs_mean > 0.15:
      return "mid"
    return "safe"

  def _road_tape_adjustment(self):
    # Broad dynamic mode switching was too disruptive: the subcontrollers depend
    # on warmed histories, and changing families mid-maneuver often worsens the
    # phase error. Instead, predict a low-dimensional correction path from the
    # road shape and apply the current value of that path.
    correction = 0.0
    if self.step_idx >= CONTROL_START_IDX:
      k = self.step_idx - CONTROL_START_IDX
      if 0 <= k < self.predicted_correction_path.size:
        correction += float(self.predicted_correction_path[k])

    if self.mode != "mellow_lowroll":
      return None, float(correction)

    tape = self.road_tape.tape()
    if tape.size <= self.step_idx:
      return None, float(correction)
    target = tape.window(self.step_idx, horizon=50)["target_lataccel"]
    peak_idx = int(np.argmax(np.abs(target)))
    signed_peak = float(target[peak_idx])
    peak_time = peak_idx * 0.1
    if abs(signed_peak) < 1.8 or peak_time < 1.0:
      return None, float(correction)

    lead_shape = np.exp(-((peak_time - 2.8) / 1.8) ** 2)
    correction += 0.38 * np.tanh(signed_peak / 3.0) * lead_shape
    return None, float(correction)

  def _aggr_params(self):
    s = self.classification_stats
    lat_max = s.get("lat_abs_max", 0.0)
    lat_mean = s.get("lat_abs_mean", 0.0)
    lat_signed = s.get("lat_signed_mean", 0.0)
    roll_mean = s.get("roll_abs_mean", 0.0)
    v_mean = s.get("v_mean", 0.0)

    if v_mean > 34.0 and lat_max < 0.12 and roll_mean < 0.17:
      return 0.75, 0.04, 0.05
    if 31.0 < v_mean < 33.0 and 0.60 < lat_mean < 0.78 and lat_signed > 0.55 and 0.20 < roll_mean < 0.28:
      return 0.75, 0.04, 0.14
    if 31.0 < v_mean < 33.0 and lat_mean > 0.74 and lat_signed < -0.70 and roll_mean > 0.33:
      return 1.0, 0.0, 0.14
    if 31.0 < v_mean < 33.0 and lat_max < 0.20 and 0.24 < roll_mean < 0.32:
      return 0.65, 0.04, 0.05
    if 30.0 < v_mean < 32.0 and lat_max < 0.22 and 0.18 < roll_mean < 0.24:
      return 0.65, 0.07, 0.14
    return 0.75, 0.07, 0.08

  def _mid_params(self):
    return 0.65, 0.16, 0.16, 18.0

  def _predict_control_path(self):
    pulses = list(BASE_PATH_PULSES.get(self.mode, ()))
    for mode, rules, gated_pulses in GATED_PATH_PULSES:
      if self.mode == mode and _stats_match(self.classification_stats, rules):
        pulses.extend(gated_pulses)
    return _render_path_pulses(ROAD_SHAPE_PATH_LENGTH, pulses)
