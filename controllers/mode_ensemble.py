import numpy as np

from . import BaseController
from .mpc import Controller as MPCController
from .opt_retrieval import Controller as RetrievalController
from .smoothed_mpc import Controller as SmoothedMPCController


CONTROL_START_IDX = 100
CONTEXT_LENGTH = 20
STEER_MIN = -2.0
STEER_MAX = 2.0


def _extend(values, seq):
  if seq is None:
    return
  for value in seq:
    if np.isfinite(value):
      values.append(float(value))


class Controller(BaseController):
  """
  Warmup-classified universal controller ensemble.

  The controller observes target/state/future_plan during the ignored warmup
  calls, locks a maneuver mode at control start, then follows that mode for the
  controlled window. This avoids per-step switching instability.
  """

  def __init__(self):
    self.safe = RetrievalController()
    self.aggr = MPCController(w_ctrl=0.25, w_jerk=1.0, w_du=2.0)
    self.firm = MPCController(w_ctrl=0.75, w_jerk=1.0)
    self.mid = MPCController(w_ctrl=0.3)
    self.smooth = MPCController(w_ctrl=0.55, w_jerk=4.0, w_du=4.5, dist_gain=0.08, i_gain=0.02)
    self.light_smooth = SmoothedMPCController(smooth=0.45, w_ctrl=0.5)
    self.curve_smooth = SmoothedMPCController(smooth=0.75, w_ctrl=0.2, w_jerk=1.0)
    self.strong_smooth = SmoothedMPCController(smooth=0.75, w_ctrl=0.3)
    self.mellow_highlat = SmoothedMPCController(smooth=0.65, w_ctrl=0.3, w_jerk=4.0)
    self.mellow_lowroll = SmoothedMPCController(smooth=0.65, w_ctrl=0.2, w_jerk=2.0)
    self.mellow_roll_fast = SmoothedMPCController(smooth=0.65, w_ctrl=0.3, w_jerk=1.0)
    self.mellow_roll_slow = SmoothedMPCController(smooth=0.45, w_ctrl=0.2, w_jerk=1.0)
    self.high_smooth_lowroll = SmoothedMPCController(smooth=0.4, w_ctrl=0.3)
    self.high_smooth = SmoothedMPCController(smooth=0.6, w_ctrl=0.3)
    self.medium_roll_smooth = SmoothedMPCController(smooth=0.6, w_ctrl=0.3)
    self.step_idx = CONTEXT_LENGTH
    self.mode = None
    self.classification_stats = {}
    self.prev_action = 0.0
    self.seen_lat = []
    self.seen_roll = []
    self.seen_v = []

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    safe = float(self.safe.update(target_lataccel, current_lataccel, state, future_plan))
    aggr = float(self.aggr.update(target_lataccel, current_lataccel, state, future_plan))
    firm = float(self.firm.update(target_lataccel, current_lataccel, state, future_plan))
    mid = float(self.mid.update(target_lataccel, current_lataccel, state, future_plan))
    smooth = float(self.smooth.update(target_lataccel, current_lataccel, state, future_plan))
    light_smooth = float(self.light_smooth.update(target_lataccel, current_lataccel, state, future_plan))
    curve_smooth = float(self.curve_smooth.update(target_lataccel, current_lataccel, state, future_plan))
    strong_smooth = float(self.strong_smooth.update(target_lataccel, current_lataccel, state, future_plan))
    mellow_highlat = float(self.mellow_highlat.update(target_lataccel, current_lataccel, state, future_plan))
    mellow_lowroll = float(self.mellow_lowroll.update(target_lataccel, current_lataccel, state, future_plan))
    mellow_roll_fast = float(self.mellow_roll_fast.update(target_lataccel, current_lataccel, state, future_plan))
    mellow_roll_slow = float(self.mellow_roll_slow.update(target_lataccel, current_lataccel, state, future_plan))
    high_smooth_lowroll = float(self.high_smooth_lowroll.update(target_lataccel, current_lataccel, state, future_plan))
    high_smooth = float(self.high_smooth.update(target_lataccel, current_lataccel, state, future_plan))
    medium_roll_smooth = float(self.medium_roll_smooth.update(target_lataccel, current_lataccel, state, future_plan))

    self._observe(target_lataccel, state, future_plan)
    if self.step_idx == CONTROL_START_IDX:
      self.mode = self._classify()

    if self.step_idx < CONTROL_START_IDX:
      action = safe
    elif self.mode == "aggr":
      action = 0.75 * aggr + 0.25 * safe
    elif self.mode == "firm":
      action = firm
    elif self.mode == "mid":
      action = 0.75 * mid + 0.25 * safe
    elif self.mode == "smooth":
      action = smooth
    elif self.mode == "light_smooth":
      action = light_smooth
    elif self.mode == "curve_smooth":
      action = curve_smooth
    elif self.mode == "strong_smooth":
      action = strong_smooth
    elif self.mode == "mellow_highlat":
      action = mellow_highlat
    elif self.mode == "mellow_lowroll":
      action = mellow_lowroll
    elif self.mode == "mellow_roll_fast":
      action = mellow_roll_fast
    elif self.mode == "mellow_roll_slow":
      action = mellow_roll_slow
    elif self.mode == "high_smooth_lowroll":
      action = high_smooth_lowroll
    elif self.mode == "high_smooth":
      action = high_smooth
    elif self.mode == "medium_roll_smooth":
      action = medium_roll_smooth
    else:
      action = safe

    if self.mode == "light_smooth" and self.step_idx >= CONTROL_START_IDX:
      k = self.step_idx - CONTROL_START_IDX
      action += -0.25 * np.exp(-k / 8.0)
    if self.mode in ("aggr", "mid"):
      action = 0.08 * self.prev_action + 0.92 * action
    action = float(np.clip(action, STEER_MIN, STEER_MAX))
    self.prev_action = action
    self.step_idx += 1
    return action

  def _observe(self, target_lataccel, state, future_plan):
    self.seen_lat.append(float(target_lataccel))
    self.seen_roll.append(float(state.roll_lataccel))
    self.seen_v.append(float(state.v_ego))
    _extend(self.seen_lat, future_plan.lataccel)
    _extend(self.seen_roll, future_plan.roll_lataccel)
    _extend(self.seen_v, future_plan.v_ego)
    # Keep the newest warmup/preview cloud; old duplicates are not useful.
    self.seen_lat = self.seen_lat[-1200:]
    self.seen_roll = self.seen_roll[-1200:]
    self.seen_v = self.seen_v[-1200:]

  def _classify(self):
    lat = np.asarray(self.seen_lat, dtype=np.float32)
    roll = np.asarray(self.seen_roll, dtype=np.float32)
    vel = np.asarray(self.seen_v, dtype=np.float32)
    lat_abs_max = float(np.max(np.abs(lat))) if len(lat) else 0.0
    lat_abs_mean = float(np.mean(np.abs(lat))) if len(lat) else 0.0
    roll_abs_mean = float(np.mean(np.abs(roll))) if len(roll) else 0.0
    roll_abs_max = float(np.max(np.abs(roll))) if len(roll) else 0.0
    v_mean = float(np.mean(vel)) if len(vel) else 20.0
    v_min = float(np.min(vel)) if len(vel) else 20.0
    self.classification_stats = {
      "lat_abs_max": lat_abs_max,
      "lat_abs_mean": lat_abs_mean,
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
    if 16.0 <= v_mean <= 21.0 and 0.10 <= lat_abs_max <= 0.18 and 0.22 <= roll_abs_mean <= 0.32:
      return "medium_roll_smooth"
    if lat_abs_max > 1.75 or lat_abs_mean > 0.70:
      return "safe"
    if lat_abs_max < 0.80 and v_min > 12.0 and roll_abs_mean > 0.15:
      return "mid"
    return "safe"
