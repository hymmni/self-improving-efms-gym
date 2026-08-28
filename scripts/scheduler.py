#!/usr/bin/env python3
"""
예약 시간에 Claude 세션을 자동으로 시작하는 스케줄러.

사용법 A — 스크립트 직접 편집:
    아래 CONFIG 섹션을 수정한 뒤 실행한다.
    python3 scheduler.py

사용법 B — 커맨드라인 인자:
    python3 scheduler.py --time 07:00 --resume <session-id> --prompt "프롬프트"
    python3 scheduler.py --time 23:30 --cmd "claude -p" --prompt "작업 내용"
    python3 scheduler.py --time 07:00 --resume <id> --model opus --permission-mode auto --prompt "작업"
"""

import argparse
import datetime
import re
import shlex
import subprocess
import sys
import time

# ──────────────────────────────────────────────
# CONFIG (사용법 A: 여기만 수정)
# ──────────────────────────────────────────────
TARGET_TIME = "15:00"  # 24시간제 HH:MM

# 세션 ID: 비워두면 새 세션으로 시작
SESSION_ID = "SI-EFMs"

# 모델:   "fable" | "opus" | "sonnet" |"haiku"
# 비워두면 Claude Code 설정값 사용 (세션에 지정된 모델 또는 CC 기본값)
MODEL = "opus"

# 권한 모드: "auto"(자동 승인) | "plan"(계획만) | "acceptEdits"(편집 자동 승인) | "dontAsk"(전체 자동 승인)
# 비워두면 Claude Code 설정값 사용
PERMISSION_MODE = "auto"

PROMPT = """이제 확인했는데 step 2에서 'd의 정의를 바꿔 데이터 수집 효율 개선'했다는게 무슨 의미야? d를 계산하는 방식을 바꿨다는거야?"""
# ──────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="예약 시간에 Claude를 실행한다.")
    p.add_argument("--time", metavar="HH:MM", help="실행 시각 (24시간제, 절대시간)")
    p.add_argument("--in", dest="in_after", metavar="DURATION",
                   help="지금부터 상대시간 (예: 2h30m, 90m, 45m). --time 대신 사용")
    p.add_argument("--resume", metavar="SESSION_ID", help="이어서 실행할 세션 ID")
    p.add_argument("--cmd", metavar="COMMAND", help="실행할 CLI 명령어 (--resume 대신)")
    p.add_argument("--model", metavar="MODEL",
                   help="모델 (sonnet | opus | haiku)")
    p.add_argument("--permission-mode", metavar="MODE",
                   choices=["auto", "plan", "acceptEdits", "dontAsk"],
                   help="권한 모드 (auto=자동승인 | plan=계획만 | acceptEdits=편집승인 | dontAsk=전체승인)")
    p.add_argument("--prompt", metavar="TEXT", help="세션에 전달할 프롬프트")
    return p.parse_args()


def next_target(time_str: str) -> datetime.datetime:
    """오늘 또는 내일의 목표 datetime을 반환한다."""
    t = datetime.datetime.strptime(time_str, "%H:%M").time()
    now = datetime.datetime.now()
    candidate = datetime.datetime.combine(now.date(), t)
    if candidate <= now:
        candidate += datetime.timedelta(days=1)
    return candidate


def parse_duration(text: str) -> datetime.timedelta:
    """'2h30m', '90m', '45m', '1h' 형식의 상대시간을 timedelta로 변환한다."""
    m = re.fullmatch(r"\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*", text, re.IGNORECASE)
    if not m or (m.group(1) is None and m.group(2) is None):
        raise ValueError(f"잘못된 시간 형식: '{text}' (예: 2h30m, 90m, 1h)")
    hours = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    return datetime.timedelta(hours=hours, minutes=mins)


def target_after(text: str) -> datetime.datetime:
    """지금부터 DURATION 만큼 뒤의 datetime을 반환한다."""
    return datetime.datetime.now() + parse_duration(text)


def run(command: str, prompt: str):
    args = shlex.split(command)
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )
    proc.communicate(input=prompt)
    return proc.returncode


def main():
    cli = parse_args()

    time_str = cli.time or TARGET_TIME
    prompt = cli.prompt or PROMPT
    model = cli.model or MODEL
    permission_mode = getattr(cli, "permission_mode", None) or PERMISSION_MODE
    session_id = cli.resume or SESSION_ID

    if cli.cmd:
        command = cli.cmd
    elif session_id:
        command = f"claude --resume {session_id}"
    else:
        command = "claude"

    if model:
        command += f" --model {model}"
    if permission_mode:
        command += f" --permission-mode {permission_mode}"

    # ANSI 색상
    C  = "\033[96m"   # cyan   — 라벨
    W  = "\033[97m"   # white  — 값
    Y  = "\033[93m"   # yellow — 카운트다운
    G  = "\033[92m"   # green  — 예약 라벨
    D  = "\033[2m"    # dim    — 보조 텍스트
    B  = "\033[1m"    # bold
    R  = "\033[0m"    # reset

    # --in(상대시간)이 주어지면 우선, 아니면 --time(절대시간) 사용
    if cli.in_after:
        try:
            target = target_after(cli.in_after)
        except ValueError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)
    else:
        target = next_target(time_str)
    print(f"  {C}{B}명령:{R}     {W}{command}{R}")
    print(f"  {C}{B}프롬프트:{R} {W}{prompt[:60].strip()}{'...' if len(prompt) > 60 else ''}{R}")
    print(f"  {D}(Ctrl+C 로 취소){R}")
    print()

    try:
        while True:
            now = datetime.datetime.now()
            remaining = target - now
            if remaining.total_seconds() <= 0:
                sys.stdout.write("\n")
                break
            total_secs = int(remaining.total_seconds())
            h, rem = divmod(total_secs, 3600)
            m, s = divmod(rem, 60)
            sys.stdout.write(
                f"\r  {G}{B}예약:{R} {W}{target.strftime('%Y-%m-%d %H:%M')}{R}"
                f"  {Y}({h:02d}:{m:02d}:{s:02d} 남음){R}"
            )
            sys.stdout.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        sys.stdout.write(f"\n  {D}취소됨.{R}\n")
        sys.exit(0)

    print(f"\n  {G}{B}[{datetime.datetime.now().strftime('%H:%M:%S')}] 실행합니다...{R}")
    code = run(command, prompt)
    if code == 0:
        print("\n  완료.")
    else:
        print(f"\n  종료 코드: {code}")
    sys.exit(code)


if __name__ == "__main__":
    main()
