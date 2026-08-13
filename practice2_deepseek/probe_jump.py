"""Open-loop max-force jump probe. Determines the physical ceiling of biped jumping.

   python probe_jump.py

Sweeps crouch depth/duration, explode duration, and launch angle to find
max achievable horizontal range from a single maximal jump.
"""
import numpy as np
import gymnasium as gym
import os

FRAME_SKIP = 10
DT = FRAME_SKIP * 0.002

CROUCH_DEPTHS = [-0.5, -1.0]        # thigh/leg action during crouch
CROUCH_STEPS = [10, 20]              # crouch duration in env steps
EXPLODE_ACTIONS = [1.0]              # thigh/leg action during explode
EXPLODE_STEPS = [10, 20]             # explode duration
FOOT_PUSH = [1.0, 0.5]              # foot action during explode


def _make_env():
    return gym.make(
        "Walker2d-v5",
        render_mode=None,
        exclude_current_positions_from_observation=False,
        frame_skip=FRAME_SKIP,
    )


def run_jump(env, crouch, crouch_steps, thigh_leg_explode, foot_explode, explode_steps):
    env.reset(seed=0)
    base = env.unwrapped
    xs, zs, vxs, vzs = [], [], [], []

    # crouch
    for _ in range(crouch_steps):
        action = np.array([crouch, crouch, 0.0, crouch, crouch, 0.0], dtype=np.float32)
        env.step(action)

    # explode
    for _ in range(explode_steps):
        action = np.array([thigh_leg_explode, thigh_leg_explode, foot_explode,
                           thigh_leg_explode, thigh_leg_explode, foot_explode], dtype=np.float32)
        env.step(action)

    # coast (no action, let gravity handle it)
    done = False
    for _ in range(500):
        action = np.zeros(6, dtype=np.float32)
        env.step(action)
        x = float(base.data.qpos[0])
        z = float(base.data.qpos[1])
        vx = float(base.data.qvel[0])
        vz = float(base.data.qvel[1])
        xs.append(x)
        zs.append(z)
        vxs.append(vx)
        vzs.append(vz)
        if z < 0.5:
            break

    if not xs:
        return None

    launch_vx = max(abs(v) for v in vxs[:5]) if len(vxs) >= 5 else 0.0
    launch_vz = max(abs(v) for v in vzs[:5]) if len(vzs) >= 5 else 0.0
    max_z = max(zs)
    final_x = max(xs)
    flight_time = len(xs) * DT

    return {
        "crouch": crouch, "crouch_steps": crouch_steps,
        "thigh_explode": thigh_leg_explode, "foot_explode": foot_explode,
        "explode_steps": explode_steps,
        "launch_vx": round(launch_vx, 3),
        "launch_vz": round(launch_vz, 3),
        "max_z": round(max_z, 3),
        "range": round(final_x, 3),
        "flight_time": round(flight_time, 3),
    }


def main():
    env = _make_env()

    results = []
    for crouch in CROUCH_DEPTHS:
        for crouch_s in CROUCH_STEPS:
            for explode_a in EXPLODE_ACTIONS:
                for explode_s in EXPLODE_STEPS:
                    for foot in FOOT_PUSH:
                        r = run_jump(env, crouch, crouch_s, explode_a, foot, explode_s)
                        if r:
                            results.append(r)

    results.sort(key=lambda r: -r["range"])

    print(f"{'crouch':>7} {'cr_steps':>9} {'th_expl':>8} {'ft_expl':>8} "
          f"{'ex_steps':>9} {'vx':>7} {'vz':>7} {'max_z':>7} {'range':>8} {'t_flight':>9}")
    print("-" * 90)
    for r in results[:15]:
        print(f"{r['crouch']:7.1f} {r['crouch_steps']:9d} {r['thigh_explode']:8.1f} "
              f"{r['foot_explode']:8.1f} {r['explode_steps']:9d} "
              f"{r['launch_vx']:7.3f} {r['launch_vz']:7.3f} {r['max_z']:7.3f} "
              f"{r['range']:8.3f} {r['flight_time']:9.3f}")

    best = results[0]
    print(f"\n=== BEST === range={best['range']:.3f}m, vx={best['launch_vx']:.3f}, "
          f"vz={best['launch_vz']:.3f}, max_z={best['max_z']:.3f}, "
          f"flight={best['flight_time']:.3f}s")

    if best["range"] < 1.0:
        print("\n[VERDICT] Single jump range < 1.0m — max-force jump idea is likely NOT viable.")
        print("          Proceed only if bounding (chained jumps) can compensate.")
    elif best["range"] < 2.0:
        print("\n[VERDICT] 1-2m range — modest. Bounding gait might help on small bumps.")
    else:
        print("\n[VERDICT] >2m range — strong. Bounding gait is physically promising.")

    env.close()


if __name__ == "__main__":
    main()