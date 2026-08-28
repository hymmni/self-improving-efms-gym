#!/usr/bin/env python3
"""
Merge a feature branch into main — opt-in 헬퍼.

이 스크립트는 **명시적으로 직접 실행할 때만** 동작한다. execute.py(step 실행기)는
main을 절대 건드리지 않으며, main 병합은 사용자가 이 헬퍼로 수행하거나 수동으로 한다.

수행 시퀀스 (사용자 지정):
    1. local main 으로 checkout 후 origin/main 을 fast-forward pull
    2. feature 브랜치에서 main 을 rebase
    3. main 으로 돌아와 feature 브랜치를 --no-ff 로 병합

안전 장치:
    - 워킹트리가 더러우면 중단한다.
    - rebase 충돌 시 자동으로 `git rebase --abort` 후 안내하고 중단한다 (사람이 해결).
    - force push는 절대 하지 않는다. main push는 --push 를 줬을 때만, 그것도 일반 push.
    - 파괴적 단계 전 확인을 요구한다 (--yes 로 생략 가능).

Usage:
    python3 scripts/merge_to_main.py [feat-branch] [--push] [--yes]

    feat-branch 를 생략하면 현재 브랜치를 사용한다.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = "main"


def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"  ERROR: git {' '.join(args)}")
        if r.stderr.strip():
            print(f"  {r.stderr.strip()}")
        sys.exit(1)
    return r


def current_branch() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def has_origin_main() -> bool:
    r = git("rev-parse", "--verify", "--quiet", "refs/remotes/origin/main", check=False)
    return r.returncode == 0


def confirm(prompt: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        ans = ""
    if ans not in ("y", "yes"):
        print("  중단됨.")
        sys.exit(130)


def main():
    parser = argparse.ArgumentParser(description="Merge a feature branch into main (opt-in).")
    parser.add_argument("feat_branch", nargs="?", help="병합할 feature 브랜치 (생략 시 현재 브랜치)")
    parser.add_argument("--push", action="store_true", help="병합 후 origin/main 으로 push")
    parser.add_argument("--yes", action="store_true", help="확인 프롬프트 생략")
    args = parser.parse_args()

    feat = args.feat_branch or current_branch()
    if feat == MAIN:
        print(f"  ERROR: '{MAIN}' 브랜치는 병합 대상이 될 수 없습니다. feature 브랜치를 지정하세요.")
        sys.exit(1)

    if git("rev-parse", "--verify", "--quiet", feat, check=False).returncode != 0:
        print(f"  ERROR: 브랜치 '{feat}' 가 존재하지 않습니다.")
        sys.exit(1)

    # 워킹트리가 깨끗한지 확인 (rebase/merge 전 필수).
    if git("status", "--porcelain").stdout.strip():
        print("  ERROR: 워킹트리에 커밋되지 않은 변경이 있습니다. commit 하거나 stash 후 다시 시도하세요.")
        sys.exit(1)

    print(f"\n  Merge plan: {feat} → {MAIN}")
    print(f"    1) {MAIN}: pull --ff-only origin/{MAIN}")
    print(f"    2) {feat}: rebase {MAIN}")
    print(f"    3) {MAIN}: merge --no-ff {feat}")
    if args.push:
        print(f"    4) push origin/{MAIN}")
    confirm("  진행할까요?", args.yes)

    # 1. local main 에서 origin/main 을 fast-forward pull.
    git("checkout", MAIN)
    if has_origin_main():
        git("pull", "--ff-only", "origin", MAIN)
    else:
        print(f"  WARN: origin/{MAIN} 이 없어 pull 을 건너뜁니다.")

    # 2. feature 브랜치에서 main 을 rebase. 충돌 시 abort 후 중단.
    git("checkout", feat)
    r = git("rebase", MAIN, check=False)
    if r.returncode != 0:
        print(f"  ✗ rebase 충돌이 발생했습니다. 자동으로 abort 합니다.")
        if r.stderr.strip():
            print(f"  {r.stderr.strip()}")
        git("rebase", "--abort", check=False)
        print(f"  → 충돌을 직접 해결하세요:")
        print(f"      git checkout {feat} && git rebase {MAIN}")
        print(f"      (충돌 해결 후) git rebase --continue")
        print(f"    그런 다음 이 스크립트를 다시 실행하세요.")
        sys.exit(1)

    # 3. main 으로 돌아와 --no-ff 병합.
    git("checkout", MAIN)
    merge_msg = f"Merge branch '{feat}' into {MAIN}"
    git("merge", "--no-ff", feat, "-m", merge_msg)
    print(f"  ✓ Merged: {merge_msg}")

    # 4. (옵션) push.
    if args.push:
        git("push", "origin", MAIN)
        print(f"  ✓ Pushed origin/{MAIN}")

    print(f"\n  완료. 현재 브랜치: {current_branch()}")


if __name__ == "__main__":
    main()
