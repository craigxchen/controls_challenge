import numpy as np

from . import BaseController
from .mpc import Controller as MPCController


STEER_MIN = -2.0
STEER_MAX = 2.0


class Controller(BaseController):
  """
  MPC with a thin outer action smoother.

  This keeps the same model-predictive proposal as ``mpc`` but damps single-step
  action changes before they enter the simulator. It is useful as a robust
  candidate in ensembles and as a quick baseline for jerk-sensitive segments.
  """

  def __init__(self, smooth=0.4, **mpc_kwargs):
    self.mpc = MPCController(**mpc_kwargs)
    self.smooth = smooth
    self.prev_action = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    raw = float(self.mpc.update(target_lataccel, current_lataccel, state, future_plan))
    action = self.smooth * self.prev_action + (1.0 - self.smooth) * raw
    action = float(np.clip(action, STEER_MIN, STEER_MAX))
    self.prev_action = action
    if hasattr(self.mpc, "prev_u"):
      self.mpc.prev_u = action
    if hasattr(self.mpc, "u_hist") and self.mpc.u_hist:
      self.mpc.u_hist[0] = action
    return action
