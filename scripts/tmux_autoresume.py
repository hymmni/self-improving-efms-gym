#!/usr/bin/env python3
"""
tmux 안에서 claude를 돌리다가 세션 리밋에 걸리면, 리셋 시각에 자동으로 "continue"를
같은 세션에 입력해 이어가게 하는 감시기.

핵심 아이디어 (커뮤니티 정착 패턴):
  - claude를 tmux pane 안에서 실행한다 (창을 닫아도 안 죽고 멈춘 채 살아있다).
  - 감시 루프가 `tmux capture-pane`으로 화면을 주기적으로 읽어 리밋 메시지를 감지한다.
  - 리셋 시각을 파싱해 그때까지 기다린 뒤, `tmux send-keys`로 살아있는 그 세션에
    "continue"를 그대로 입력한다 → 새 프로세스(resume)가 아니라 같은 세션이 이어지므로
    대화가 갈라지지(fork) 않는다.

사용법:
    python3 scripts/tmux_autoresume.py            # 세션+claude+감시기 띄우고 attach
    python3 scripts/tmux_autoresume.py --session mywork --poll 5 --margin 60

세션이 이미 있으면 그 세션에 붙고 감시기만 (없으면) 새로 띄운다.
"""

import argparse
import datetime
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SESSION = "claude-harness"
MONITOR_WINDOW = "autoresume"   # 감시기를 띄울 tmux window 이름
CLAUDE_WINDOW = "claude"        # claude를 띄울 tmux window 이름

# 리밋 메시지 감지 패턴. Claude Code 버전에 따라 문구가 다를 수 있으니
# "limit"과 "reset"이 같은 줄/근처에 있으면 잡도록 느슨하게 둔다.
LIMIT_HINT = re.compile(r"limit\s*(reached|reset)", re.IGNORECASE)
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


# ── tmux 헬퍼 ───────────────────────────────────────────────

def tmux(*args, capture=False) -> subprocess.CompletedProcess:
    return subprocess.run(["tmux", *args], capture_output=capture, text=True)


def tmux_available() -> bool:
    try:
        return subprocess.run(["tmux", "-V"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def session_exists(name: str) -> bool:
    return tmux("has-session", "-t", name, capture=True).returncode == 0


def capture_pane(target: str) -> str:
    r = tmux("capture-pane", "-p", "-t", target, capture=True)
    return r.stdout if r.returncode == 0 else ""


def send_continue(target: str):
    # "continue" + Enter 만 보낸다. (Escape를 앞에 붙이면 더미/일부 터미널에서
    # ESC가 다음 글자와 합쳐져 'ESC c' = 터미널 리셋으로 먹히는 문제가 있어 제외.)
    # 리밋 시점의 입력창은 비어 있으므로 이대로 충분하다.
    tmux("send-keys", "-t", target, "continue", "Enter")


# ── 파싱 ────────────────────────────────────────────────────

def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_reset(text: str) -> datetime.datetime | None:
    """리밋 메시지에서 리셋 시각을 파싱한다. 실패 시 None.

    지원 형식 예:
      "resets 3pm"                  → 오늘(지났으면 내일) 15:00
      "resets 3:30pm"               → 15:30
      "resets Jun 10, 7:10pm"       → 6월 10일 19:10
      "resets at 3 pm"              → 15:00
    """
    clean = strip_ansi(text)
    m = re.search(
        r"reset[s]?\s*(?:at\s+)?"
        r"(?:([A-Za-z]{3})\w*\s+(\d{1,2}),?\s+)?"   # 선택: 월 일
        r"(\d{1,2})(?::(\d{2}))?\s*([ap]m)",          # 시[:분] am/pm
        clean, re.IGNORECASE)
    if not m:
        return None
    mon_s, day_s, hour_s, min_s, ampm = m.groups()
    hour = int(hour_s)
    minute = int(min_s or 0)
    if ampm.lower() == "pm" and hour != 12:
        hour += 12
    if ampm.lower() == "am" and hour == 12:
        hour = 0

    now = datetime.datetime.now()
    if mon_s:  # 월/일이 명시된 경우
        month = _MONTHS.get(mon_s.lower())
        if month is None:
            return None
        reset = datetime.datetime(now.year, month, int(day_s), hour, minute)
        # 연도는 메시지에 없다. 파싱 결과가 "크게(약 반년 이상) 과거"면 Dec→Jan 같은
        # 연말 경계로 보고 내년으로 올린다. 단지 몇 시간 과거인 경우는 그대로 둔다
        # (실제 리밋 메시지의 리셋 시각은 미래이므로 이런 케이스는 드물다).
        if reset < now - datetime.timedelta(days=180):
            reset = reset.replace(year=now.year + 1)
    else:      # 시각만 있는 경우 → 오늘, 지났으면 내일
        reset = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if reset <= now:
            reset += datetime.timedelta(days=1)
    return reset


# ── 감시 루프 ───────────────────────────────────────────────

def monitor_loop(session: str, poll: int, margin: int):
    """claude pane을 주기적으로 읽어 리밋을 감지하면 리셋 후 continue를 보낸다."""
    target = f"{session}:{CLAUDE_WINDOW}"
    handled_reset: datetime.datetime | None = None   # 같은 리밋 창 중복 처리 방지
    log(f"감시 시작: target={target}, poll={poll}s, margin={margin}s")

    while True:
        if not session_exists(session):
            log("세션이 사라졌습니다. 감시 종료.")
            return

        screen = capture_pane(target)
        if LIMIT_HINT.search(strip_ansi(screen)):
            reset = parse_reset(screen)
            if reset is None:
                log("리밋 메시지는 감지했으나 리셋 시각 파싱 실패. 다음 폴링에서 재시도.")
            elif reset == handled_reset:
                pass  # 이미 이 리셋 창을 처리함 — 대기/전송 중복 방지
            else:
                wait_s = (reset - datetime.datetime.now()).total_seconds() + margin
                log(f"리밋 감지 — 리셋 {reset:%Y-%m-%d %H:%M} 까지 {int(max(0, wait_s))}s 대기")
                if wait_s > 0:
                    time.sleep(wait_s)
                send_continue(target)
                handled_reset = reset
                log("continue 전송 완료. 계속 감시합니다.")

        time.sleep(poll)


def log(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── 시작 (세션 구성 + attach) ───────────────────────────────

def start(session: str, poll: int, margin: int):
    if not tmux_available():
        print("  ERROR: tmux가 설치돼 있지 않습니다. `sudo apt install tmux` 후 다시 실행하세요.")
        sys.exit(1)

    fresh = not session_exists(session)
    if fresh:
        # window 0: claude (인터랙티브)
        tmux("new-session", "-d", "-s", session, "-n", CLAUDE_WINDOW, "claude")
        # window 1: 이 스크립트를 감시 모드로 (같은 tmux 서버에 묶여 함께 생존)
        self_path = str(Path(__file__).resolve())
        monitor_cmd = (
            f"{sys.executable} {self_path} --monitor-only "
            f"--session {session} --poll {poll} --margin {margin}"
        )
        tmux("new-window", "-d", "-t", session, "-n", MONITOR_WINDOW, monitor_cmd)
        print(f"  세션 '{session}' 생성: window[{CLAUDE_WINDOW}]=claude, window[{MONITOR_WINDOW}]=감시기")
    else:
        print(f"  세션 '{session}' 이미 존재 — attach 합니다. (감시기는 기존 것 사용)")

    print(f"  attach: tmux attach -t {session}   |  분리: Ctrl+b d   |  창 전환: Ctrl+b 0/1")
    tmux("attach", "-t", session)


def main():
    p = argparse.ArgumentParser(description="tmux 기반 claude 리밋 자동 재개")
    p.add_argument("--session", default=DEFAULT_SESSION, help="tmux 세션 이름")
    p.add_argument("--poll", type=int, default=5, help="화면 폴링 간격(초)")
    p.add_argument("--margin", type=int, default=60, help="리셋 시각 후 추가 대기(초)")
    p.add_argument("--monitor-only", action="store_true",
                   help="(내부용) 감시 루프만 실행 — start가 tmux window에서 호출")
    args = p.parse_args()

    if args.monitor_only:
        monitor_loop(args.session, args.poll, args.margin)
    else:
        start(args.session, args.poll, args.margin)


if __name__ == "__main__":
    main()
