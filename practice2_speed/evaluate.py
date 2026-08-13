"""Local grader — reproduces the challenge server's score on your own machine.

    python evaluate.py --model checkpoints/bump_challenge/walker_model_500000_steps.zip
    python evaluate.py --model model.zip --video run.mp4
    python evaluate.py --model model.zip --render

Run it from the project root, next to `custom_walker2d.py` and `asset/`.

The number it prints is the number the server will publish. Grading is a single
deterministic episode — reset noise is off and the policy is greedy — so the
same (code, checkpoint) pair always scores exactly the same, here and there.

If this script says your submission fails, the server will say the same thing.
Fix it here first; a failed run does not place on the leaderboard.

--------------------------------------------------------------------------
How grading works
--------------------------------------------------------------------------
Two phases, exactly as on the server:

  Phase 1  Your `CustomEnvWrapper` is imported and stepped. Your
           `custom_observation`, `custom_reward`, `custom_terminated` and
           `custom_truncated` all run, just as they did in training. This phase
           produces *only the sequence of actions your policy chose*. It scores
           nothing.

  Phase 2  A fresh, official environment is built with none of your code
           loaded, and those actions are applied to it. The distance, the fall
           check and the video all come from this run.

That is why editing the simulator inside `step()` cannot raise your score:
phase 2 never sees it. What survives is what your torques really do.

The terrain, `frame_skip` and `reset_noise_scale` are forced to the server's
values no matter what your `__init__` asks for. Everything else — how you build
the observation, how you compute the reward, when you flag the episode
terminated — is yours, because grading reads none of it.
"""
import argparse
import hashlib
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# =========================================================================
# These must match the server. Do not change them to make your score look
# better — the server uses its own copy and yours would simply disagree.
# =========================================================================
MAX_STEPS = 1000              # 1000 * frame_skip 10 * 0.002 s = 20 s of simulated time
FRAME_SKIP = 10
RESET_NOISE_SCALE = 0.0
HEALTHY_Z_MIN = 0.5           # torso below this counts as fallen
HEALTHY_Z_MAX = 10.0
GOAL_X = 101.9                # x past which the course counts as finished
SEED = 0
CHALLENGE_XML = "asset/custom_walker2d_bumps.xml"

# md5 of the official terrain, announced with the challenge. The script always
# prints the md5 of the file it actually used; set this and it also warns when
# the two differ, which means you are not looking at the course you will be
# graded on. Leave it empty to only print.
OFFICIAL_XML_MD5 = "338efa83e0e7f70abf80b2ba514322cb"

VIDEO_FPS = 25
VIDEO_EVERY_N_FRAMES = 2
VIDEO_WIDTH = 480
VIDEO_HEIGHT = 360
VIDEO_CRF = 28

# A wrapper with many boolean switches has exponentially many configurations, so
# the search over them is bounded, plausible combinations first.
MAX_CANDIDATES = 64


class Rejected(Exception):
    """The submission cannot be graded. Carries a message you can act on."""


# --- the official environment -------------------------------------------
# Captured before gym.make is replaced below, so student code can only ever
# reach the replacement.
import gymnasium as gym                                          # noqa: E402

_REAL_MAKE = gym.make


def _forced_kwargs(xml):
    """The settings a submission does not get a say in.

    Exactly the ones that change the physics or the starting state, which is
    what would make two runs incomparable.
    """
    return dict(
        xml_file=os.path.abspath(xml),
        frame_skip=FRAME_SKIP,
        reset_noise_scale=RESET_NOISE_SCALE,
        max_episode_steps=MAX_STEPS + 10,
    )


def install_locked_make(xml):
    """Replace gym.make so the terrain and the physics are always the server's.

    Your own keyword arguments still flow through for anything that only shapes
    the observation — notably `exclude_current_positions_from_observation`,
    because forcing it would silently shift every index your policy was trained
    against.
    """
    def locked_make(*_args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "id"}
        kwargs.update(_forced_kwargs(xml))
        kwargs["render_mode"] = None          # phase 1 never renders
        return _REAL_MAKE("Walker2d-v5", **kwargs)

    gym.make = locked_make


def make_official(xml, render_mode=None):
    """The scorer's own environment.

    The observation layout is irrelevant here: this phase only applies actions
    and reads the simulator state. The base env never ends the episode on its
    own — the scorer decides when the walker has fallen.
    """
    return _REAL_MAKE(
        "Walker2d-v5",
        **_forced_kwargs(xml),
        exclude_current_positions_from_observation=False,
        render_mode=render_mode,
        width=VIDEO_WIDTH,
        height=VIDEO_HEIGHT,
        healthy_z_range=(-1e6, 1e6),
        terminate_when_unhealthy=False,
    )


# --- phase 1: your code produces an action sequence ----------------------

def import_student_module():
    sys.path.insert(0, os.getcwd())
    try:
        import custom_walker2d
    except Exception as exc:
        raise Rejected(f"custom_walker2d.py를 import할 수 없습니다: {exc!r}")
    if not hasattr(custom_walker2d, "CustomEnvWrapper"):
        raise Rejected("custom_walker2d.py에 CustomEnvWrapper 클래스가 없습니다.")
    return custom_walker2d


def load_model(ckpt):
    from stable_baselines3 import PPO

    # Tolerate checkpoints saved by a slightly different SB3 version: the
    # schedules are the only pickled callables that commonly break.
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    try:
        return PPO.load(ckpt, device="cpu", custom_objects=custom_objects)
    except Exception as exc:
        raise Rejected(f"체크포인트를 로드할 수 없습니다: {exc!r}")


def _candidate_kwargs(cls):
    """Constructor arguments to try, best guess first.

    Every boolean switch the constructor declares is enumerated rather than
    assuming a naming convention, and the checkpoint's observation width then
    picks the right combination. Terrain switches are preferred on; other
    switches tend to be curriculum or randomisation, which have no place in a
    deterministic evaluation.
    """
    import inspect
    import itertools

    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return [{}]

    flags = [name for name, p in params.items()
             if name != "self" and isinstance(p.default, bool)][:8]
    takes_render_mode = "render_mode" in params

    def terrainish(name):
        return any(w in name.lower() for w in ("bump", "challenge", "terrain", "height"))

    combos = []
    for values in itertools.product([True, False], repeat=len(flags)):
        chosen = dict(zip(flags, values))
        rank = (-sum(1 for f, v in chosen.items() if terrainish(f) and v),
                sum(1 for f, v in chosen.items() if not terrainish(f) and v))
        combos.append((rank, chosen))

    combos.sort(key=lambda item: item[0])
    out = [{"render_mode": None, **c} if takes_render_mode else dict(c)
           for _, c in combos[:MAX_CANDIDATES]]
    return out or [{"render_mode": None} if takes_render_mode else {}]


def build_student_env(module, expected_obs):
    """Build your wrapper in the configuration your checkpoint was trained on.

    `expected_obs` is the observation width stored in the policy, which makes
    this a check rather than a guess: a configuration either reproduces the
    layout the policy was trained against or it is the wrong one.

    Define `make_eval_env()` in custom_walker2d.py to skip the search entirely
    and say exactly how the grading environment should be built.
    """
    if hasattr(module, "make_eval_env"):
        try:
            return module.make_eval_env(), {"make_eval_env": True}
        except Exception as exc:
            raise Rejected(f"make_eval_env() 실행 중 오류: {exc!r}")

    cls = module.CustomEnvWrapper
    tried, first_error = [], None
    for kwargs in _candidate_kwargs(cls):
        try:
            env = cls(**kwargs)
            obs, _ = env.reset(seed=SEED)
        except Exception as exc:
            first_error = first_error or exc
            continue
        width = int(np.size(obs))
        tried.append((kwargs, width))
        if expected_obs is None or width == expected_obs:
            return env, kwargs

    if not tried:
        raise Rejected("CustomEnvWrapper를 생성할 수 없습니다 (생성자를 확인하세요). "
                       f"마지막 오류: {first_error!r}")
    options = ", ".join(f"{kw}→{w}차원" for kw, w in tried[:6])
    raise Rejected(
        f"체크포인트는 관측 {expected_obs}차원으로 학습됐는데, 제출한 코드로 만들 수 있는 "
        f"관측 차원이 일치하지 않습니다 (시도: {options}). 학습에 쓴 설정과 같은 환경이 "
        "기본값으로 만들어지도록 하거나, make_eval_env() 함수를 정의해 주세요.")


def collect_actions(env, model):
    """Step your environment, keeping every action the policy chose."""
    try:
        obs, _ = env.reset(seed=SEED)
    except Exception as exc:
        raise Rejected(f"env.reset 중 오류: {exc!r}")

    actions, crashed = [], None
    for _ in range(MAX_STEPS):
        try:
            action, _ = model.predict(obs, deterministic=True)
        except Exception as exc:
            raise Rejected("정책이 관측값을 처리하지 못했습니다. 학습 때와 observation "
                           f"차원이 다른지 확인하세요: {exc!r}")
        actions.append(np.asarray(action, dtype=np.float32).ravel())

        try:
            obs = env.step(action)[0]
        except Exception as exc:
            if not actions:
                raise Rejected(f"env.step 중 오류: {exc!r}")
            crashed = exc      # keep what we have; the replay scores it honestly
            break

        # Purely an optimisation: a shorter action sequence replays the same.
        try:
            z = float(env.unwrapped.data.qpos[1])
        except Exception:
            continue
        if not (HEALTHY_Z_MIN < z < HEALTHY_Z_MAX):
            break

    stacked = np.stack(actions) if actions else np.zeros((0, 6), dtype=np.float32)
    return stacked, crashed


# --- phase 2: a clean replay decides the score ---------------------------

class VideoWriter:
    """Streams frames straight into ffmpeg so the whole clip is never held."""

    def __init__(self, path):
        self.path = path
        self.proc = None
        self.failed = False

    def add(self, frame):
        if self.failed:
            return
        try:
            if self.proc is None:
                self._start(frame.shape[1], frame.shape[0])
            self.proc.stdin.write(frame.tobytes())
        except Exception:
            self.failed = True
            self.close()

    def _start(self, width, height):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(VIDEO_FPS), "-i", "-",
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", str(VIDEO_CRF), "-preset", "veryfast", str(self.path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def close(self):
        if self.proc is None:
            return False
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=60)
        except Exception:
            self.proc.kill()
            self.failed = True
        self.proc = None
        return not self.failed and Path(self.path).exists()


def install_safe_mouse_wheel(env, min_distance=0.5, max_distance=50.0):
    """Replace Gymnasium's incompatible MuJoCo 3.11 scroll callback.

    Gymnasium 1.1.1 calls `mjv_moveCamera` with six arguments while MuJoCo 3.11 accepts
    five, so the built-in wheel handler raises TypeError and kills the viewer the first
    time you scroll. This sets `cam.distance` directly instead.

    Best effort throughout: `--render` is a convenience, and nothing about the viewer may
    cost you a score, so a missing glfw or an unexpected viewer layout just means no
    workaround rather than a failed run.
    """
    try:
        import glfw
    except Exception:
        return False

    viewer = getattr(getattr(env.unwrapped, "mujoco_renderer", None), "viewer", None)
    window = getattr(viewer, "window", None)
    if window is None or getattr(viewer, "_safe_scroll_callback_installed", False):
        return False

    def safe_scroll_callback(_window, _x_offset, y_offset):
        try:
            scroll = float(y_offset)
            distance = float(viewer.cam.distance)
            if not (math.isfinite(scroll) and math.isfinite(distance)):
                return
            scroll = max(-5.0, min(5.0, scroll))
            viewer.cam.distance = max(
                min_distance,
                min(max_distance, distance * math.exp(-0.12 * scroll)),
            )
        except Exception:
            # Exceptions must not escape a GLFW callback and close the viewer.
            return

    # Keep a Python reference to the callback for the lifetime of the viewer.
    viewer._safe_scroll_callback = safe_scroll_callback
    viewer._safe_scroll_callback_installed = True
    glfw.set_scroll_callback(window, safe_scroll_callback)
    return True


def replay(xml, actions, video=None, render=False):
    """Apply the actions to an environment built here, with no student code."""
    mode = "human" if render else ("rgb_array" if video is not None else None)
    env = make_official(xml, render_mode=mode)
    base = env.unwrapped
    env.reset(seed=SEED)

    if render:
        # The viewer does not exist until the first render, so the wheel fix goes here.
        env.render()
        install_safe_mouse_wheel(env)

    best_x, steps = 0.0, 0
    reason = "no_actions" if len(actions) == 0 else "timeout"

    for steps, action in enumerate(actions, start=1):
        env.step(np.clip(action, env.action_space.low, env.action_space.high))

        x = float(base.data.qpos[0])
        z = float(base.data.qpos[1])
        best_x = max(best_x, x)

        if render:
            env.render()
        elif video is not None and steps % VIDEO_EVERY_N_FRAMES == 0:
            try:
                frame = env.render()
                if frame is not None:
                    video.add(np.ascontiguousarray(frame, dtype=np.uint8))
            except Exception:
                pass       # a broken renderer must never cost you your score

        if not (HEALTHY_Z_MIN < z < HEALTHY_Z_MAX):
            reason = "fall"
            break
        if x >= GOAL_X:
            reason = "goal"
            break

    if reason == "timeout" and steps < MAX_STEPS:
        # The rollout stopped early for a reason the replay did not reproduce.
        reason = "ended_early"

    sim_time = steps * FRAME_SKIP * float(base.model.opt.timestep)
    env.close()
    return {
        "score": round(best_x, 4),
        "final_x": round(float(base.data.qpos[0]), 4),
        "final_z": round(float(base.data.qpos[1]), 4),
        "steps": steps,
        "sim_time": round(sim_time, 3),
        "avg_speed": round(best_x / sim_time, 4) if sim_time > 0 else 0.0,
        "end_reason": reason,
    }


# --- reporting -----------------------------------------------------------

REASON_LABEL = {
    "goal": "완주",
    "fall": "넘어짐",
    "timeout": "시간 초과 (정상 종료)",
    "ended_early": "주행 중단 — 실패",
    "no_actions": "행동 없음 — 실패",
}


def check_inputs(xml, ckpt):
    """Fail early, and with a sentence you can act on, rather than mid-import."""
    if not Path("custom_walker2d.py").exists():
        raise Rejected("현재 폴더에 custom_walker2d.py가 없습니다. "
                       "learning.py를 돌리던 프로젝트 루트에서 실행하세요.")
    if not Path(ckpt).exists():
        raise Rejected(f"체크포인트를 찾을 수 없습니다: {ckpt}")

    path = Path(xml)
    if not path.exists():
        raise Rejected(f"지형 파일이 없습니다: {xml} (프로젝트 루트에서 실행하세요)")

    digest = hashlib.md5(path.read_bytes()).hexdigest()
    print(f"[지형] {xml}  md5={digest}")
    if OFFICIAL_XML_MD5 and digest != OFFICIAL_XML_MD5:
        print(f"!! 경고: 공식 지형(md5 {OFFICIAL_XML_MD5})과 다른 파일입니다.", file=sys.stderr)
        print("   서버는 공식 지형으로 채점하므로 아래 점수는 참고가 되지 않습니다.",
              file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="채점 서버와 동일한 방식으로 로컬 채점")
    parser.add_argument("--model", required=True, help="제출할 PPO 체크포인트 (.zip)")
    parser.add_argument("--xml", default=CHALLENGE_XML, help="공식 지형 XML")
    parser.add_argument("--video", default=None, help="주행 영상을 저장할 mp4 경로 (ffmpeg 필요)")
    parser.add_argument("--render", action="store_true", help="실시간 창으로 보기")
    args = parser.parse_args()

    try:
        check_inputs(args.xml, args.model)
        install_locked_make(args.xml)
        module = import_student_module()

        # The checkpoint comes first: its observation width is what tells us
        # which constructor configuration the policy was trained against.
        model = load_model(args.model)
        expected = None
        space = getattr(model, "observation_space", None)
        shape = getattr(space, "shape", None) if space is not None else None
        if shape and len(shape) == 1:
            expected = int(shape[0])

        env, chosen = build_student_env(module, expected)
        print(f"[phase 1] env built with {chosen}, obs={expected}")

        actions, crashed = collect_actions(env, model)
        print(f"[phase 1] collected {len(actions)} actions")
        if crashed is not None:
            print(f"[phase 1] step()에서 예외 발생: {crashed!r}")
    except Rejected as exc:
        print(f"\n채점 실패: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:                     # noqa: BLE001 - report, don't trace
        print(f"\n채점 실패: 코드 실행 중 예외가 발생했습니다: {exc!r}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    video = VideoWriter(args.video) if args.video and not args.render else None
    result = replay(args.xml, actions, video=video, render=args.render)
    has_video = bool(video is not None and video.close())

    failed = result["end_reason"] in ("ended_early", "no_actions")
    print(f"\n{'=' * 46}")
    if failed:
        print(f"  결과      실패 ({REASON_LABEL[result['end_reason']]})")
        print(f"  거리      {result['score']:.2f} m  ← 순위에 반영되지 않습니다")
    else:
        print(f"  점수      {result['score']:.2f} m")
        print(f"  종료 사유  {REASON_LABEL.get(result['end_reason'], result['end_reason'])}")
    print(f"  생존 스텝  {result['steps']} / {MAX_STEPS}")
    print(f"  생존 시간  {result['sim_time']:.2f} 초")
    print(f"  평균 속도  {result['avg_speed']:.2f} m/s")
    print(f"  최종 위치  x = {result['final_x']:.2f} m, z = {result['final_z']:.2f} m")
    print(f"{'=' * 46}")

    if failed:
        print("\n주행이 끝까지 진행되지 않았습니다. 넘어지지도, 완주하지도 않은 채 중단됐다는")
        print("뜻이고, 보통 step()이나 custom_observation()이 도중에 예외를 던진 경우입니다.")
        print("완료되지 않은 주행은 리더보드에 올라가지 않으니 반드시 고쳐서 제출하세요.")
    if has_video:
        print(f"\n영상: {args.video}")
    elif args.video and not args.render:
        print("\n영상 저장 실패 — ffmpeg가 설치되어 있는지 확인하세요 (sudo apt install ffmpeg).",
              file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
