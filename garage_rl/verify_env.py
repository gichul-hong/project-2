"""
환경 파일 무결성 검증.

    python verify_env.py

env/ 아래 파일의 SHA-256 해시를 config/env_manifest.sha256 과 비교합니다.
실습이 이상하게 돌면 먼저 이걸 실행해서 환경을 건드리지 않았는지 확인하세요.
env/ 를 고치면 문제 자체가 달라져 다른 사람 결과와 비교할 수 없습니다.

조교용:
    python verify_env.py --write     # 현재 파일 기준으로 manifest 재생성
"""

import argparse
import hashlib
import os
import sys

MANIFEST = os.path.join("config", "env_manifest.sha256")
# garage_3.py 는 challenge 로 따로 배포하므로, 있을 때만 검증합니다.
TARGETS = [
    os.path.join("env", name)
    for name in ("_core.py", "garage_1.py", "garage_2.py")
]
_G3 = os.path.join("env", "garage_3.py")
if os.path.exists(_G3):
    TARGETS.append(_G3)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest():
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as f:
        for path in TARGETS:
            f.write(f"{sha256(path)}  {path}\n")
    print(f"wrote {MANIFEST}")
    for path in TARGETS:
        print(f"  {sha256(path)}  {path}")


def check():
    if not os.path.exists(MANIFEST):
        print(f"[FAIL] manifest 가 없습니다: {MANIFEST}")
        return False

    expected = {}
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if line:
                digest, path = line.split(None, 1)
                expected[path.strip()] = digest

    ok = True
    for path in TARGETS:
        if path not in expected:
            print(f"[FAIL] manifest 에 항목이 없습니다: {path}")
            ok = False
            continue
        if not os.path.exists(path):
            print(f"[FAIL] 파일이 없습니다: {path}")
            ok = False
            continue
        actual = sha256(path)
        if actual == expected[path]:
            print(f"[ OK ] {path}")
        else:
            print(f"[FAIL] {path} 가 변경되었습니다")
            print(f"       기대: {expected[path]}")
            print(f"       실제: {actual}")
            ok = False

    extra = set(expected) - set(TARGETS)
    for path in sorted(extra):
        print(f"[WARN] manifest 에 있으나 검증 대상이 아닙니다: {path}")

    print()
    if ok:
        print("환경 파일이 원본과 같습니다.")
    else:
        print("환경 파일이 원본과 다릅니다. 배포본의 env/ 를 다시 받아 덮어쓰세요.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help="조교용: manifest 재생성")
    a = ap.parse_args()
    if a.write:
        write_manifest()
    else:
        sys.exit(0 if check() else 1)
