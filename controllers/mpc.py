from . import BaseController
import numpy as np


def _solve_spd(matrix, rhs):
  """Small Cholesky solver avoiding slow generic LAPACK startup on tiny QPs."""
  matrix = np.asarray(matrix, dtype=float)
  rhs = np.asarray(rhs, dtype=float)
  n = matrix.shape[0]
  chol = np.zeros_like(matrix)

  for i in range(n):
    for j in range(i + 1):
      acc = matrix[i, j]
      if j:
        acc -= float(chol[i, :j] @ chol[j, :j])
      if i == j:
        chol[i, j] = np.sqrt(max(acc, 1e-9))
      else:
        chol[i, j] = acc / chol[j, j]

  y = np.zeros(n)
  for i in range(n):
    acc = rhs[i]
    if i:
      acc -= float(chol[i, :i] @ y[:i])
    y[i] = acc / chol[i, i]

  x = np.zeros(n)
  for i in range(n - 1, -1, -1):
    acc = y[i]
    if i + 1 < n:
      acc -= float(chol[i + 1:, i] @ x[i + 1:])
    x[i] = acc / chol[i, i]
  return x


class Controller(BaseController):
  """
  v0.2 -- offset-free linear MPC on a speed-scheduled ARX plant model.

  Plant model (identified by least-squares on dithered rollouts, piecewise in
  speed -> coefficients interpolated by v_ego):
      a[t] = a1 a[t-1] + a2 a[t-2] + b1 u[t-1] + b2 u[t-2] + b3 u[t-3]
             + g roll[t] + c
  DC gain rises ~1.2 (low v) -> 1.5 (high v); the model carries the real lag,
  so we no longer need v0.1's detuned gain hack.

  Each step we rebuild the impulse-response matrix M for the current speed,
  predict a = M u + free_response(IC, future roll, c, bias) over the future
  plan, and minimize  w_track||a-r||^2 + w_jerk||Da||^2 + w_du||Du||^2  in
  closed form (receding horizon, apply first action).

  Bias is killed two ways:
    * d_hat: offset-free output-disturbance estimate from the one-step
      prediction error (integral action; near-zero now the model is accurate)
    * i_term: a slow tracking-error integral backstop, anti-wound.
  """

  # ARX coefficients [a1, a2, b1, b2, b3, g, c] at speed knots (m/s)
  KNOT_V = np.array([11.0, 25.5, 33.0])
  KNOT_COEF = np.array([
    [0.486, 0.117, 0.060, 0.038, 0.385, 0.375,  0.033],
    [0.537, 0.011, 0.007, 0.142, 0.484, 0.525, -0.009],
    [0.598, 0.024, 0.314, 0.072, 0.181, 0.435,  0.002],
  ])

  def __init__(self, horizon=25, w_track=1.0, w_jerk=2.0, w_du=3.0,
               w_ctrl=0.5, dist_gain=0.1, i_gain=0.05, i_clip=0.5):
    self.H = horizon
    self.w_track = w_track
    self.w_jerk = w_jerk
    self.w_du = w_du
    self.w_ctrl = w_ctrl
    self.dist_gain = dist_gain
    self.i_gain = i_gain
    self.i_clip = i_clip

    self.Tout = np.eye(horizon) - np.eye(horizon, k=-1)
    self.Tin = np.eye(horizon) - np.eye(horizon, k=-1)

    # state
    self.a_m1 = None          # a[t-1]
    self.u_hist = [0.0, 0.0, 0.0]   # [u[t-1], u[t-2], u[t-3]]
    self.d_hat = 0.0
    self.i_term = 0.0
    self.prev_pred = None
    self.prev_u = 0.0

  def _coef(self, v):
    return np.array([np.interp(v, self.KNOT_V, self.KNOT_COEF[:, k]) for k in range(7)])

  def _impulse(self, coef, H):
    a1, a2, b1, b2, b3 = coef[:5]
    h = np.zeros(H + 1)              # h[1..H]
    if H >= 1: h[1] = b1
    if H >= 2: h[2] = a1 * h[1] + b2
    if H >= 3: h[3] = a1 * h[2] + a2 * h[1] + b3
    for k in range(4, H + 1):
      h[k] = a1 * h[k - 1] + a2 * h[k - 2]
    return h[1:]

  def _free_response(self, coef, H, a0, roll_future, bias):
    a1, a2, b1, b2, b3, g, c = coef
    um1, um2, um3 = self.u_hist
    am1 = self.a_m1 if self.a_m1 is not None else a0
    a_prev2, a_prev1 = am1, a0       # a[k-2], a[k-1]
    out = np.zeros(H)
    for k in range(1, H + 1):
      # zero future input -> only past actions contribute, at k=1,2,3
      ul1 = um1 if k == 1 else (um2 if k == 2 else (um3 if k == 3 else 0.0))
      ul2 = um2 if k == 1 else (um3 if k == 2 else 0.0)
      ul3 = um3 if k == 1 else 0.0
      roll_k = roll_future[k - 1] if k - 1 < len(roll_future) else (roll_future[-1] if len(roll_future) else 0.0)
      ak = a1 * a_prev1 + a2 * a_prev2 + b1 * ul1 + b2 * ul2 + b3 * ul3 + g * roll_k + c + bias
      out[k - 1] = ak
      a_prev2, a_prev1 = a_prev1, ak
    return out

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    a0 = current_lataccel

    # --- integral updates (bias killers) ---
    if self.prev_pred is not None:
      self.d_hat += self.dist_gain * (a0 - self.prev_pred)        # offset-free
    self.i_term += self.i_gain * (a0 - target_lataccel)           # tracking-error integral
    self.i_term = float(np.clip(self.i_term, -self.i_clip, self.i_clip))
    bias = self.d_hat + self.i_term

    future = future_plan.lataccel
    H = min(self.H, len(future))
    coef = self._coef(state.v_ego)

    if H < 1:
      h1 = max(coef[2], 1e-3)
      u0 = float(np.clip((target_lataccel - bias - coef[6]) / h1, -2, 2))
      self._commit(a0, u0, None)
      return u0

    roll_future = np.asarray(future_plan.roll_lataccel[:H], dtype=float)
    h = self._impulse(coef, H)
    M = np.zeros((H, H))
    for off in range(H):                            # M[r,c]=h[r-c]: lower-tri Toeplitz
      idx = np.arange(H - off)
      M[idx + off, idx] = h[off]
    d_free = self._free_response(coef, H, a0, roll_future, bias)
    r = np.asarray(future[:H], dtype=float)

    Tout = self.Tout[:H, :H]; Tin = self.Tin[:H, :H]
    bt = d_free - r
    TM = Tout @ M
    c0 = np.zeros(H); c0[0] = a0
    bj = Tout @ d_free - c0
    e0 = np.zeros(H); e0[0] = 1.0

    Hqp = (self.w_track * (M.T @ M) + self.w_jerk * (TM.T @ TM)
           + self.w_du * (Tin.T @ Tin) + self.w_ctrl * np.eye(H))
    f = (self.w_track * (M.T @ bt) + self.w_jerk * (TM.T @ bj)
         - self.w_du * self.prev_u * (Tin.T @ e0))

    u = _solve_spd(Hqp, -f)
    u0 = float(np.clip(u[0], -2, 2))
    self._commit(a0, u0, float(M[0] @ u + d_free[0]))
    return u0

  def _commit(self, a0, u0, pred):
    self.a_m1 = a0                  # becomes a[t-1] for the next call
    self.prev_pred = pred
    self.prev_u = u0
    self.u_hist = [u0, self.u_hist[0], self.u_hist[1]]
