import numpy as np

from . import BaseController
from .mpc import Controller as MPCController
from .mpc import _solve_spd


CONTEXT_LENGTH = 20
STEER_MIN = -2.0
STEER_MAX = 2.0


class Controller(BaseController):
  """
  Direct quadratic action optimizer over the visible road tape.

  This borrows the leaderboard-style idea of optimizing the action sequence
  directly, but keeps the runtime universal: no segment IDs, no data paths, no
  per-segment lookup. Each update builds a time-varying ARX response matrix
  from the current state plus future_plan speed/roll, solves the full visible
  action vector, and applies the first action.
  """

  def __init__(self, horizon=48, w_track=1.0, w_jerk=2.2, w_du=2.6,
               w_ctrl=0.42, w_plan=0.08, dist_gain=0.08, i_gain=0.035,
               i_clip=0.45, smooth=0.04):
    self.H = int(horizon)
    self.w_track = float(w_track)
    self.w_jerk = float(w_jerk)
    self.w_du = float(w_du)
    self.w_ctrl = float(w_ctrl)
    self.w_plan = float(w_plan)
    self.dist_gain = float(dist_gain)
    self.i_gain = float(i_gain)
    self.i_clip = float(i_clip)
    self.smooth = float(smooth)

    self.Tout = np.eye(self.H) - np.eye(self.H, k=-1)
    self.Tin = np.eye(self.H) - np.eye(self.H, k=-1)

    self.step_idx = CONTEXT_LENGTH
    self.a_m1 = None
    self.u_hist = [0.0, 0.0, 0.0]
    self.d_hat = 0.0
    self.i_term = 0.0
    self.prev_pred = None
    self.prev_u = 0.0
    self.prev_action = 0.0
    self.prev_plan = np.zeros(0, dtype=np.float32)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    a0 = float(current_lataccel)
    if self.prev_pred is not None:
      self.d_hat += self.dist_gain * (a0 - self.prev_pred)
    self.i_term += self.i_gain * (a0 - float(target_lataccel))
    self.i_term = float(np.clip(self.i_term, -self.i_clip, self.i_clip))
    bias = self.d_hat + self.i_term

    target = np.asarray(future_plan.lataccel, dtype=float)
    roll = np.asarray(future_plan.roll_lataccel, dtype=float)
    v_ego = np.asarray(future_plan.v_ego, dtype=float)
    H = min(self.H, target.size, roll.size, v_ego.size)
    if H < 1:
      self._commit(a0, 0.0, None)
      return 0.0

    target = target[:H]
    roll = roll[:H]
    v_ego = v_ego[:H]
    response, free = self._affine_response(H, a0, roll, v_ego, bias)
    plan_prior = self._plan_prior(H)

    Tout = self.Tout[:H, :H]
    Tin = self.Tin[:H, :H]
    c0 = np.zeros(H)
    c0[0] = a0
    e0 = np.zeros(H)
    e0[0] = 1.0

    mean_err = free - target
    jerk_free = Tout @ free - c0
    jerk_mat = Tout @ response
    q = (
      self.w_track * (response.T @ response)
      + self.w_jerk * (jerk_mat.T @ jerk_mat)
      + self.w_du * (Tin.T @ Tin)
      + self.w_ctrl * np.eye(H)
      + self.w_plan * np.eye(H)
    )
    f = (
      self.w_track * (response.T @ mean_err)
      + self.w_jerk * (jerk_mat.T @ jerk_free)
      - self.w_du * self.prev_u * (Tin.T @ e0)
      - self.w_plan * plan_prior
    )

    plan = np.clip(_solve_spd(q, -f), STEER_MIN, STEER_MAX)
    action = float(plan[0])
    if self.smooth > 0.0:
      action = self.smooth * self.prev_action + (1.0 - self.smooth) * action
    action = float(np.clip(action, STEER_MIN, STEER_MAX))
    pred0 = float(response[0] @ plan + free[0])
    self.prev_plan = plan.astype(np.float32)
    self._commit(a0, action, pred0)
    return action

  def _affine_response(self, H, a0, roll, v_ego, bias):
    free = self._simulate(H, a0, roll, v_ego, bias, active_col=None)
    matrix = np.zeros((H, H), dtype=float)
    for col in range(H):
      matrix[:, col] = self._simulate(H, a0, roll, v_ego, bias, active_col=col) - free
    return matrix, free

  def _simulate(self, H, a0, roll, v_ego, bias, active_col):
    am1 = self.a_m1 if self.a_m1 is not None else a0
    a_prev2, a_prev1 = float(am1), float(a0)
    out = np.zeros(H, dtype=float)
    for k in range(H):
      coef = self._coef(v_ego[k])
      a1, a2, b1, b2, b3, g, c = coef

      def u_at(offset):
        idx = k + offset
        if idx < 0:
          hist_idx = -idx - 1
          return float(self.u_hist[hist_idx] if hist_idx < len(self.u_hist) else self.u_hist[-1])
        return 1.0 if active_col is not None and idx == active_col else 0.0

      ak = (
        a1 * a_prev1 + a2 * a_prev2
        + b1 * u_at(0) + b2 * u_at(-1) + b3 * u_at(-2)
        + g * roll[k] + c + bias
      )
      out[k] = ak
      a_prev2, a_prev1 = a_prev1, ak
    return out

  def _plan_prior(self, H):
    if self.prev_plan.size:
      shifted = self.prev_plan[1:].astype(float)
      tail = float(self.prev_plan[-1])
      prior = np.concatenate([shifted, np.full(max(0, H - shifted.size), tail, dtype=float)])
      return np.clip(prior[:H], STEER_MIN, STEER_MAX)
    return np.full(H, self.prev_u, dtype=float)

  @staticmethod
  def _coef(v):
    return np.array([np.interp(v, MPCController.KNOT_V, MPCController.KNOT_COEF[:, k]) for k in range(7)])

  def _commit(self, a0, u0, pred):
    self.a_m1 = float(a0)
    self.prev_pred = pred
    self.prev_u = float(u0)
    self.prev_action = float(u0)
    self.u_hist = [float(u0), self.u_hist[0], self.u_hist[1]]
    self.step_idx += 1
