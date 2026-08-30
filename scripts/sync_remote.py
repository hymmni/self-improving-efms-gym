#!/usr/bin/env python3
"""
SSH로 접속 가능한 원격 서버에 있는 같은 레포와 data/checkpoints/outputs/results를
rsync로 동기화한다 (ADR-002: 이 디렉토리들은 .gitignore 대상이라 git으로는 옮길 수 없음).

동기화 대상은 아래 4종류 디렉토리뿐이다(코드/설정은 git으로 관리하므로 건드리지 않는다):
    data/, checkpoints/, outputs/, results/
루트와 각 projects/<name>/ 아래 양쪽 모두를 후보로 스캔한다.

이 스크립트는 보통 **호스트(컨테이너 밖)**에서 실행한다 — SSH 키는 대개 호스트에만
있고, 이미지에는 rsync/openssh-client가 안 깔려 있다. 컨테이너 안에서 쓰려면
`apt-get install -y rsync openssh-client`를 직접 해야 한다.

Usage:
    python3 scripts/sync_remote.py push user@gpuserver:/home/user/self-improving-gym
    python3 scripts/sync_remote.py pull user@gpuserver:/home/user/self-improving-gym --dry-run
    python3 scripts/sync_remote.py pull user@gpuserver:/home/user/self-improving-gym \\
        --only checkpoints --project grasp_carry

    push = 로컬 → 원격, pull = 원격 → 로컬.
    <remote> 은 rsync 타깃 형식: user@host:/절대/경로/까지의/레포루트 (트레일링 슬래시 없이).

안전 장치:
    - 기본은 추가만 한다(rsync에 --delete 없음). 원격/로컬에만 있는 파일을 지우려면
      --delete를 명시해야 하며, 그 경우 실행 전 확인을 요구한다(--yes로 생략 가능).
    - --dry-run은 rsync -n으로 그대로 전달되어 실제 전송 없이 계획만 보여준다.
    - 각 방향에서 존재하지 않는 후보 디렉토리는 조용히 건너뛴다(에러 아님).
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYNC_DIR_NAMES = ["data", "checkpoints", "outputs", "results"]


def discover_candidates(root: Path, project_filter: list[str] | None) -> list[Path]:
    """루트 및 projects/<name>/ 아래에서 동기화 후보 디렉토리(상대경로 기준)를 나열한다.
    실제 존재 여부는 확인하지 않는다 — pull로 로컬에 아직 없는 디렉토리를 만들 수도 있어야 하기 때문."""
    candidates = []
    if project_filter is None or "root" in project_filter:
        candidates += [root / name for name in SYNC_DIR_NAMES]

    projects_dir = root / "projects"
    if projects_dir.is_dir():
        for proj in sorted(p for p in projects_dir.iterdir() if p.is_dir()):
            if project_filter is not None and proj.name not in project_filter:
                continue
            candidates += [proj / name for name in SYNC_DIR_NAMES]
    return candidates


def build_ssh_cmd(port: str | None, identity: str | None) -> str:
    parts = ["ssh"]
    if port:
        parts += ["-p", port]
    if identity:
        parts += ["-i", identity]
    return " ".join(shlex.quote(p) for p in parts)


def remote_dir_exists(ssh_cmd: str, host: str, path: str) -> bool:
    r = subprocess.run(
        [*shlex.split(ssh_cmd), host, f"test -d {shlex.quote(path)}"],
        capture_output=True,
    )
    return r.returncode == 0


def ensure_remote_dir(ssh_cmd: str, host: str, path: str) -> None:
    subprocess.run(
        [*shlex.split(ssh_cmd), host, f"mkdir -p {shlex.quote(path)}"],
        check=True,
    )


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
    parser = argparse.ArgumentParser(
        description="SSH 원격 서버와 data/checkpoints/outputs/results를 rsync로 동기화한다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("direction", choices=["push", "pull"], help="push=로컬→원격, pull=원격→로컬")
    parser.add_argument("remote", help="user@host:/절대/경로/레포루트 (rsync 타깃 형식)")
    parser.add_argument(
        "--only",
        help=f"쉼표로 구분된 대상 디렉토리 이름 (기본: 전체 = {','.join(SYNC_DIR_NAMES)})",
    )
    parser.add_argument(
        "--project",
        help="쉼표로 구분된 project 이름(예: grasp_carry,square_assembly) + 'root'. 생략 시 전체.",
    )
    parser.add_argument("--dry-run", action="store_true", help="실제 전송 없이 rsync -n으로 계획만 출력")
    parser.add_argument("--delete", action="store_true", help="목적지에만 있는 파일 삭제(rsync --delete, 위험)")
    parser.add_argument("--port", help="SSH 포트")
    parser.add_argument("--identity", "-i", help="SSH 개인키 경로")
    parser.add_argument("--yes", action="store_true", help="확인 프롬프트 생략")
    args = parser.parse_args()

    if ":" not in args.remote:
        print("  ERROR: --remote는 'user@host:/path' 형식이어야 합니다.")
        sys.exit(1)
    host, _, remote_root = args.remote.partition(":")
    if not remote_root.startswith("/"):
        print("  ERROR: 원격 경로는 절대경로여야 합니다 (예: user@host:/home/user/self-improving-gym).")
        sys.exit(1)

    only = set(n.strip() for n in args.only.split(",")) if args.only else set(SYNC_DIR_NAMES)
    unknown = only - set(SYNC_DIR_NAMES)
    if unknown:
        print(f"  ERROR: --only에 알 수 없는 이름: {sorted(unknown)} (허용: {SYNC_DIR_NAMES})")
        sys.exit(1)

    project_filter = None
    if args.project:
        project_filter = [p.strip() for p in args.project.split(",")]

    if subprocess.run(["which", "rsync"], capture_output=True).returncode != 0:
        print("  ERROR: rsync가 설치되어 있지 않습니다 (apt-get install -y rsync openssh-client).")
        sys.exit(1)

    ssh_cmd = build_ssh_cmd(args.port, args.identity)

    candidates = discover_candidates(ROOT, project_filter)
    candidates = [c for c in candidates if c.name in only]

    jobs = []  # (local_path, remote_path, relpath)
    for local_dir in candidates:
        relpath = local_dir.relative_to(ROOT).as_posix()
        remote_dir = f"{remote_root}/{relpath}"
        if args.direction == "push":
            if not local_dir.is_dir():
                continue
        else:  # pull
            if not remote_dir_exists(ssh_cmd, host, remote_dir):
                continue
        jobs.append((local_dir, remote_dir, relpath))

    if not jobs:
        print("  동기화할 디렉토리가 없습니다 (양쪽 모두 비었거나 필터가 너무 좁습니다).")
        return

    print(f"\n  Sync plan: {args.direction} ({'local → remote' if args.direction == 'push' else 'remote → local'})")
    print(f"    remote: {args.remote}")
    for _, _, relpath in jobs:
        print(f"    - {relpath}/")
    if args.delete:
        print("    ⚠ --delete: 목적지에만 있는 파일을 삭제합니다.")
    if args.dry_run:
        print("    (--dry-run: 실제 전송 없음)")
    confirm("  진행할까요?", args.yes)

    rsync_base = ["rsync", "-avh", "--progress", "-e", ssh_cmd]
    if args.dry_run:
        rsync_base.append("-n")
    if args.delete:
        rsync_base.append("--delete")

    failures = []
    for local_dir, remote_dir, relpath in jobs:
        if args.direction == "push":
            if not args.dry_run:
                ensure_remote_dir(ssh_cmd, host, remote_dir)
            src, dst = f"{local_dir}/", f"{host}:{remote_dir}/"
        else:
            local_dir.mkdir(parents=True, exist_ok=True)
            src, dst = f"{host}:{remote_dir}/", f"{local_dir}/"

        print(f"\n  → {relpath}/")
        r = subprocess.run([*rsync_base, src, dst])
        if r.returncode != 0:
            failures.append(relpath)

    if failures:
        print(f"\n  ✗ 실패: {failures}")
        sys.exit(1)
    print("\n  ✓ 완료.")


if __name__ == "__main__":
    main()
