"""채점기와 동일한 조건(고정 지형/frame_skip/noise 0, seed 0, greedy)으로 1회 주행하며
x 진행 로그를 남긴다. 어디서 정체·낙상하는지 진단용.

  python diag_run.py --model <ckpt.zip> [--every 25]
"""
import argparse

import numpy as np

import evaluate as ev


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--xml", default=ev.CHALLENGE_XML)
    p.add_argument("--every", type=int, default=25)
    args = p.parse_args()

    ev.check_inputs(args.xml, args.model)
    ev.install_locked_make(args.xml)
    module = ev.import_student_module()
    model = ev.load_model(args.model)
    env, chosen = ev.build_student_env(module, model.observation_space.shape[0])
    print(f"env: {chosen}")

    obs, _ = env.reset(seed=ev.SEED)
    base = env.unwrapped
    prev_x = float(base.data.qpos[0])
    stalled_since = None
    best_x = prev_x
    for step in range(1, ev.MAX_STEPS + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs = env.step(action)[0]
        x = float(base.data.qpos[0])
        z = float(base.data.qpos[1])
        best_x = max(best_x, x)
        if step % args.every == 0:
            speed = (x - prev_x) / (args.every * ev.FRAME_SKIP * 0.002)
            nxt = [b for b in getattr(env, "bumps", []) if x < b["back_x"]][:1]
            nxt_s = (f" next={nxt[0]['name']}@{nxt[0]['x']:.1f} h={nxt[0]['height']:.2f}"
                     if nxt else "")
            print(f"step {step:4d}  x={x:7.2f}  z={z:5.2f}  v={speed:5.2f} m/s{nxt_s}")
            prev_x = x
        if not (ev.HEALTHY_Z_MIN < z < ev.HEALTHY_Z_MAX):
            print(f"!! fell at step {step}, x={x:.2f}, z={z:.2f}")
            break
        if x < best_x - 1e-9 + 0.0:
            pass
        if x > best_x - 0.05:
            stalled_since = step
        elif stalled_since is not None and step - stalled_since > 150:
            pass
    print(f"best_x = {best_x:.2f}")
    env.close()


if __name__ == "__main__":
    main()
