from . import BaseController
import numpy as np


class Controller(BaseController):
  """
  Offset-free linear MPC.

  Plant approximation (identified by probing tinyphysics):
      a[k] = (1-beta)*a[k-1] + beta*G*u[k-1-delay]
  i.e. lataccel is a delayed first-order response to steer, static gain G.

  Each step we predict the lataccel trajectory over the future plan,
      a = M @ u + decay*a0 + d_hat
  and minimize the challenge's own quadratic cost plus a control-rate term:
      w_track*||a-r||^2 + w_jerk*||Da||^2 + w_du*||Du||^2
  in closed form (normal equations), applying only the first action
  (receding horizon).

  d_hat is an estimated output disturbance updated from the one-step
  prediction error -> integral action -> zero steady-state offset, which
  absorbs gain mismatch and road-roll disturbance (this is what a plain
  MPC lacks vs. a PID's integral term).
  """

  def __init__(self, horizon=25, gain=4.4, beta=0.35, delay=0,
               w_track=1.0, w_jerk=2.0, w_du=0.25, w_ctrl=1e-3, dist_gain=0.4):
    # NOTE: `gain` is a tuning knob, deliberately set above the measured static
    # gain (~1.9) and `delay` left at 0 -- both detune the closed loop for
    # smoothness; the disturbance estimate (dist_gain) supplies the integral
    # action that removes the resulting steady-state bias. Tuned on 20 segments.
    self.H = horizon
    self.w_track = w_track
    self.w_jerk = w_jerk
    self.w_du = w_du
    self.w_ctrl = w_ctrl
    self.dist_gain = dist_gain

    H = horizon
    # impulse response of the plant to a unit u at step 0, over a[1..H]
    imp = np.zeros(H); a = 0.0
    for k in range(H):
      drive = 1.0 if (k - delay) >= 0 else 0.0
      a = (1 - beta) * a + beta * gain * drive
      imp[k] = a
    M = np.zeros((H, H))
    for j in range(H):
      M[j:, j] = imp[:H - j]
    self.M = M
    self.decay = (1 - beta) ** np.arange(1, H + 1)   # free response of a0
    self.Tout = np.eye(H) - np.eye(H, k=-1)          # output first-difference
    self.Tin = np.eye(H) - np.eye(H, k=-1)           # input first-difference

    # integral / disturbance estimate state
    self.d_hat = 0.0
    self.prev_pred = None   # model's one-step-ahead prediction made last step
    self.prev_u = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    a0 = current_lataccel

    # --- offset-free update: correct disturbance from one-step prediction error ---
    if self.prev_pred is not None:
      self.d_hat += self.dist_gain * (a0 - self.prev_pred)

    future = future_plan.lataccel
    H = min(self.H, len(future))
    if H < 1:
      u0 = (target_lataccel - self.d_hat) / max(self.M[0, 0], 1e-3)
      u0 = float(np.clip(u0, -2, 2))
      self.prev_u = u0
      self.prev_pred = None
      return u0

    M = self.M[:H, :H]
    Tout = self.Tout[:H, :H]
    Tin = self.Tin[:H, :H]
    decay = self.decay[:H]
    d_free = decay * a0 + self.d_hat            # free response + disturbance
    r = np.asarray(future[:H], dtype=float)

    # residuals (all linear in u)
    bt = d_free - r                              # tracking:  M u + bt
    TM = Tout @ M; bj = Tout @ d_free            # jerk:      TM u + bj  (a0 coupling below)
    c = np.zeros(H); c[0] = a0; bj = bj - c
    e0 = np.zeros(H); e0[0] = 1.0                # input-rate couples u0 to prev_u

    Hqp = (self.w_track * (M.T @ M)
           + self.w_jerk * (TM.T @ TM)
           + self.w_du * (Tin.T @ Tin)
           + self.w_ctrl * np.eye(H))
    f = (self.w_track * (M.T @ bt)
         + self.w_jerk * (TM.T @ bj)
         - self.w_du * self.prev_u * (Tin.T @ e0))

    u = np.linalg.solve(Hqp, -f)
    u0 = float(np.clip(u[0], -2, 2))

    # one-step-ahead prediction for next disturbance update
    self.prev_pred = float(M[0] @ u + d_free[0])
    self.prev_u = u0
    return u0
