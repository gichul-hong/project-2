import sys

sys.argv = ["dump_map"]
from custom_walker2d import CustomEnvWrapper  # noqa: E402

e = CustomEnvWrapper(render_mode=None, bump_challenge=True)
print(f"n bumps = {len(e.bumps)}")
for b in e.bumps:
    print(f"{b['name']:8s} x={b['x']:7.2f} front={b['front_x']:7.2f} back={b['back_x']:7.2f} "
          f"hw={b['half_width']:.2f} h={b['height']:.2f} rel={b['rel_height']:.2f} big={b['big']}")
