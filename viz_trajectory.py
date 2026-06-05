"""
Bird's-eye trajectory utilities for tinyphysics rollouts.

The sim only predicts lateral acceleration, never position, so we reconstruct
the planar path kinematically from (v_ego, lataccel):

    a_turn = lataccel - roll_lataccel      # drop the gravity/road-bank artifact
    omega  = -a_turn / v_ego               # yaw rate (right-positive lataccel -> clockwise)
    theta  = cumulative integral of omega
    x,y    = cumulative integral of v_ego * (cos theta, sin theta)

Frame: x = forward (m), y = left (m). Reconstruction is an open-loop double
integration, so it is meaningful as a *relative* comparison (controller vs
controller, or vs target), not a survey-grade map.

Use as a util:
    from viz_trajectory import plot_birdseye
    fig, runs = plot_birdseye("./data/00000.csv", controllers=["pid", "mpc"])

Or from the CLI:
    .venv/bin/python viz_trajectory.py --data_path ./data/00000.csv --controllers pid mpc
"""
import argparse
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tinyphysics import (TinyPhysicsModel, TinyPhysicsSimulator,
                         CONTROL_START_IDX, COST_END_IDX, DEL_T)

DEFAULT_MODEL = "./models/tinyphysics.onnx"
PALETTE = ["#c0392b", "#27ae60", "#8e44ad", "#e67e22", "#16a085"]  # per-controller colors


def reconstruct_path(v_ego, lataccel, roll_lataccel):
  """Bird's-eye (x_forward, y_left) path from a lataccel/speed rollout."""
  a_turn = np.asarray(lataccel) - np.asarray(roll_lataccel)
  omega = -a_turn / np.maximum(np.asarray(v_ego), 1e-3)
  theta = np.cumsum(omega) * DEL_T
  x = np.cumsum(v_ego * np.cos(theta)) * DEL_T
  y = np.cumsum(v_ego * np.sin(theta)) * DEL_T
  return x, y


def run_controller(data_path, controller, model=None, model_path=DEFAULT_MODEL):
  """Run one controller on one segment; return its histories + cost."""
  model = model or TinyPhysicsModel(model_path, debug=False)
  ctrl = importlib.import_module(f"controllers.{controller}").Controller()
  sim = TinyPhysicsSimulator(model, str(data_path), controller=ctrl, debug=False)
  cost = sim.rollout()
  return {
    "cost": cost,
    "v_ego": np.array([s.v_ego for s in sim.state_history]),
    "roll": np.array([s.roll_lataccel for s in sim.state_history]),
    "target": np.array(sim.target_lataccel_history),
    "current": np.array(sim.current_lataccel_history),
    "action": np.array(sim.action_history),
  }


def plot_birdseye(data_path, controllers=("pid", "mpc"), model=None,
                  model_path=DEFAULT_MODEL, out=None):
  """
  Overlay the target path and each controller's reconstructed bird's-eye path
  (top), plus the lataccel time series (bottom). Returns (fig, runs).
  Controllers share the same per-segment RNG seed, so the comparison is fair.
  """
  model = model or TinyPhysicsModel(model_path, debug=False)
  runs = {c: run_controller(data_path, c, model=model) for c in controllers}

  ref = next(iter(runs.values()))                      # target/speed are identical across runs
  tx, ty = reconstruct_path(ref["v_ego"], ref["target"], ref["roll"])

  fig = plt.figure(figsize=(14, 9), constrained_layout=True)
  gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1])
  ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

  ax0.plot(tx, ty, color="#2980b9", lw=2.5, label="target path")
  ax1.plot(ref["target"], color="#2980b9", lw=2, label="target")
  for i, (name, r) in enumerate(runs.items()):
    color = PALETTE[i % len(PALETTE)]
    cx, cy = reconstruct_path(r["v_ego"], r["current"], r["roll"])
    label = f"{name} (cost {r['cost']['total_cost']:.0f})"
    ax0.plot(cx, cy, color=color, lw=1.6, alpha=0.85, label=label)
    ax1.plot(r["current"], color=color, lw=1.2, alpha=0.85, label=label)

  ci = min(CONTROL_START_IDX, len(tx) - 1)
  ax0.scatter([tx[ci]], [ty[ci]], color="black", zorder=5, s=40, label="control start")
  ax0.scatter([tx[0]], [ty[0]], color="green", zorder=5, s=40, label="start")
  ax0.set_aspect("equal")
  ax0.set_xlabel("forward (m)"); ax0.set_ylabel("left (m)")
  ax0.set_title(f"Bird's-eye trajectory — {data_path}")
  ax0.legend(loc="best")

  ax1.axvline(CONTROL_START_IDX, color="black", ls="--", alpha=0.6)
  ax1.axvline(COST_END_IDX, color="gray", ls=":", alpha=0.6)
  ax1.set_xlabel("step (10 FPS)"); ax1.set_ylabel("lat accel (m/s^2)")
  ax1.legend(loc="upper right", ncol=len(runs) + 1, fontsize=8)

  if out:
    fig.savefig(out, dpi=110)
  return fig, runs


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--data_path", default="./data/00000.csv")
  p.add_argument("--controllers", nargs="+", default=["pid", "mpc"])
  p.add_argument("--model_path", default=DEFAULT_MODEL)
  p.add_argument("--out", default="comparison.png")
  args = p.parse_args()

  fig, runs = plot_birdseye(args.data_path, controllers=args.controllers,
                            model_path=args.model_path, out=args.out)
  for name, r in runs.items():
    c = r["cost"]
    print(f"{name:8} total={c['total_cost']:7.2f}  lataccel={c['lataccel_cost']:.3f}  jerk={c['jerk_cost']:.2f}")
  print(f"saved {args.out}")
