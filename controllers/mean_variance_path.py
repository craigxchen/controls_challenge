import numpy as np

from . import BaseController
from .lib.blackbox_response import LinearBlackboxResponse
from .mpc import Controller as MPCController
from .mpc import _solve_spd


CONTEXT_LENGTH = 20
CONTROL_START_IDX = 100
STEER_MIN = -2.0
STEER_MAX = 2.0
ROLL_HISTORY_LIMIT = 24
U_HISTORY_LIMIT = 24
Y_HISTORY_LIMIT = 24


class Controller(BaseController):
  """
  Direct continuous whole-path optimizer.

  Each update solves for the continuous steering vector u[0:H] directly. By
  default the response matrix comes from the conservative ARX approximation; a
  train-fitted blackbox response artifact can be enabled for experiments, but
  remains opt-in until the horizon optimizer is constrained tightly enough.
  """

  def __init__(self, horizon=35, w_track=1.0, w_var=0.65, w_jerk=2.5,
               w_du=2.5, w_ctrl=0.45, dist_gain=0.08, i_gain=0.04,
               i_clip=0.45, use_learned_model=False):
    if use_learned_model:
      horizon = min(int(horizon), 14)
    self.H = horizon
    self.w_track = w_track
    self.w_var = w_var
    self.w_jerk = w_jerk
    self.w_du = w_du
    self.w_ctrl = w_ctrl
    self.dist_gain = dist_gain
    self.i_gain = i_gain
    self.i_clip = i_clip

    self.Tout = np.eye(horizon) - np.eye(horizon, k=-1)
    self.Tin = np.eye(horizon) - np.eye(horizon, k=-1)

    self.step_idx = CONTEXT_LENGTH
    self.a_m1 = None
    self.u_hist = [0.0, 0.0, 0.0]
    self.y_hist = []
    self.d_hat = 0.0
    self.i_term = 0.0
    self.prev_pred = None
    self.prev_u = 0.0
    self.planned_path = np.zeros(0, dtype=np.float32)
    self.response_model = LinearBlackboxResponse.try_load() if use_learned_model else None
    self.roll_hist = []
    self.v_hist = []
    self.aego_hist = []

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    a0 = float(current_lataccel)
    if self.response_model is not None and self.step_idx < CONTROL_START_IDX:
      u0 = 0.0
      self._commit(a0, u0, None, state)
      return u0

    if self.prev_pred is not None:
      self.d_hat += self.dist_gain * (a0 - self.prev_pred)
    self.i_term += self.i_gain * (a0 - float(target_lataccel))
    self.i_term = float(np.clip(self.i_term, -self.i_clip, self.i_clip))
    bias = self.d_hat + self.i_term

    target = np.asarray(future_plan.lataccel, dtype=float)
    roll = np.asarray(future_plan.roll_lataccel, dtype=float)
    a_future = np.asarray(future_plan.a_ego, dtype=float)
    v_future = np.asarray(future_plan.v_ego, dtype=float)
    if self.response_model is not None:
      target = np.concatenate([[float(target_lataccel)], target])
      roll = np.concatenate([[float(state.roll_lataccel)], roll])
      a_future = np.concatenate([[float(state.a_ego)], a_future])
      v_future = np.concatenate([[float(state.v_ego)], v_future])
    H = min(self.H, target.size, roll.size, a_future.size, v_future.size)
    if H < 1:
      u0 = 0.0
      self._commit(a0, u0, None, state)
      return u0

    target = target[:H]
    roll = roll[:H]
    a_future = a_future[:H]
    v_future = v_future[:H]
    if self.response_model is not None:
      mean_m, mean_d = self._learned_response(H, a0, roll, a_future, v_future, bias)
      dm = np.zeros((0, H, H), dtype=float)
      dd = np.zeros((0, H), dtype=float)
      weights = np.zeros((0, 1, 1), dtype=float)
    else:
      models, model_weights = self._model_ensemble(float(state.v_ego))
      matrices = []
      frees = []
      for coef in models:
        h = self._impulse(coef, H)
        matrices.append(self._response_matrix(h, H))
        frees.append(self._free_response(coef, H, a0, roll, bias))
      matrices = np.asarray(matrices, dtype=float)
      frees = np.asarray(frees, dtype=float)
      weights = model_weights[:, None, None]

      mean_m = np.sum(weights * matrices, axis=0)
      mean_d = np.sum(weights[:, :, 0] * frees, axis=0)
      dm = matrices - mean_m[None, :, :]
      dd = frees - mean_d[None, :]

    Tout = self.Tout[:H, :H]
    Tin = self.Tin[:H, :H]
    c0 = np.zeros(H)
    c0[0] = a0
    e0 = np.zeros(H)
    e0[0] = 1.0

    risk = 1.0 + 0.025 * max(float(state.v_ego) - 20.0, 0.0)
    risk += 0.10 * min(abs(float(state.roll_lataccel)), 4.0)
    var_w = self.w_var * risk

    q = (
      self.w_track * (mean_m.T @ mean_m)
      + self.w_jerk * ((Tout @ mean_m).T @ (Tout @ mean_m))
      + self.w_du * (Tin.T @ Tin)
      + self.w_ctrl * np.eye(H)
    )
    f = (
      self.w_track * mean_m.T @ (mean_d - target)
      + self.w_jerk * (Tout @ mean_m).T @ (Tout @ mean_d - c0)
      - self.w_du * self.prev_u * (Tin.T @ e0)
    )

    for weight, m_delta, d_delta in zip(weights[:, 0, 0], dm, dd):
      q += var_w * float(weight) * (m_delta.T @ m_delta)
      f += var_w * float(weight) * (m_delta.T @ d_delta)
      jm = Tout @ m_delta
      jd = Tout @ d_delta
      q += 0.25 * var_w * float(weight) * (jm.T @ jm)
      f += 0.25 * var_w * float(weight) * (jm.T @ jd)

    u = _solve_spd(q, -f)
    self.planned_path = np.clip(u, STEER_MIN, STEER_MAX).astype(np.float32)
    u0 = float(self.planned_path[0])
    pred0 = float(mean_m[0] @ u + mean_d[0])
    self._commit(a0, u0, pred0, state)
    return u0

  def _learned_response(self, H, a0, roll_future, a_future, v_future, bias):
    y_history = [a0] + self.y_hist
    base_actions = self._base_actions(H)
    matrix, free = self.response_model.response_matrix(
      y_history=y_history,
      u_history=self.u_hist,
      roll_future=roll_future,
      a_future=a_future,
      v_future=v_future,
      horizon=H,
      base_actions=base_actions,
      roll_history=self.roll_hist,
      a_history=self.aego_hist,
      v_history=self.v_hist,
    )
    return matrix, free - matrix @ base_actions + bias

  def _base_actions(self, H):
    if self.planned_path.size:
      shifted = self.planned_path[1:].astype(float)
      tail = float(self.planned_path[-1])
      base = np.concatenate([shifted, np.full(max(0, H - shifted.size), tail, dtype=float)])
      return np.clip(base[:H], STEER_MIN, STEER_MAX)
    return np.full(H, self.prev_u, dtype=float)

  def _model_ensemble(self, v_ego):
    base = self._coef(v_ego)
    models = [base]
    weights = [0.44]
    for speed_shift, gain_scale, roll_scale, weight in (
      (-2.5, 0.92, 1.05, 0.14),
      (2.5, 1.08, 0.95, 0.14),
      (0.0, 0.86, 1.12, 0.10),
      (0.0, 1.14, 0.88, 0.10),
      (-4.0, 1.00, 1.00, 0.04),
      (4.0, 1.00, 1.00, 0.04),
    ):
      coef = self._coef(v_ego + speed_shift)
      coef = coef.copy()
      coef[2:5] *= gain_scale
      coef[5] *= roll_scale
      models.append(coef)
      weights.append(weight)
    weights = np.asarray(weights, dtype=float)
    weights /= np.sum(weights)
    return np.asarray(models, dtype=float), weights

  def _coef(self, v):
    return np.array([np.interp(v, MPCController.KNOT_V, MPCController.KNOT_COEF[:, k]) for k in range(7)])

  def _impulse(self, coef, H):
    a1, a2, b1, b2, b3 = coef[:5]
    h = np.zeros(H + 1)
    if H >= 1:
      h[1] = b1
    if H >= 2:
      h[2] = a1 * h[1] + b2
    if H >= 3:
      h[3] = a1 * h[2] + a2 * h[1] + b3
    for k in range(4, H + 1):
      h[k] = a1 * h[k - 1] + a2 * h[k - 2]
    return h[1:]

  def _response_matrix(self, impulse, H):
    matrix = np.zeros((H, H))
    for off in range(H):
      idx = np.arange(H - off)
      matrix[idx + off, idx] = impulse[off]
    return matrix

  def _free_response(self, coef, H, a0, roll_future, bias):
    a1, a2, b1, b2, b3, g, c = coef
    um1, um2, um3 = self.u_hist[:3]
    am1 = self.a_m1 if self.a_m1 is not None else a0
    a_prev2, a_prev1 = am1, a0
    out = np.zeros(H)
    for k in range(1, H + 1):
      ul1 = um1 if k == 1 else (um2 if k == 2 else (um3 if k == 3 else 0.0))
      ul2 = um2 if k == 1 else (um3 if k == 2 else 0.0)
      ul3 = um3 if k == 1 else 0.0
      roll_k = roll_future[k - 1] if k - 1 < len(roll_future) else roll_future[-1]
      ak = a1 * a_prev1 + a2 * a_prev2 + b1 * ul1 + b2 * ul2 + b3 * ul3 + g * roll_k + c + bias
      out[k - 1] = ak
      a_prev2, a_prev1 = a_prev1, ak
    return out

  def _commit(self, a0, u0, pred, state=None):
    self.y_hist = [a0] + self.y_hist[:Y_HISTORY_LIMIT - 1]
    self.a_m1 = a0
    self.prev_pred = pred
    self.prev_u = u0
    self.u_hist = [u0] + self.u_hist[:U_HISTORY_LIMIT - 1]
    if state is not None:
      self.roll_hist = [float(state.roll_lataccel)] + self.roll_hist[:ROLL_HISTORY_LIMIT - 1]
      self.v_hist = [float(state.v_ego)] + self.v_hist[:ROLL_HISTORY_LIMIT - 1]
      self.aego_hist = [float(state.a_ego)] + self.aego_hist[:ROLL_HISTORY_LIMIT - 1]
    self.step_idx += 1
