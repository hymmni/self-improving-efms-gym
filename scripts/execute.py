#!/usr/bin/env python3
"""
Harness Plan Executor — writing-plans 체크박스 플랜을 순차 실행하고 자가 교정한다.

Usage:
    python3 scripts/execute.py <plan.md> [--push]
"""

import argparse
import contextlib
import json
import re
import subprocess
import sys
import threading
import time
import types
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

_TASK_HEADER_RE = re.compile(r"^###\s+Task\s+(\d+):\s*(.+?)\s*$", re.MULTILINE)


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip().lower())
    return slug.strip("-")


def parse_plan(text: str) -> dict:
    """writing-plans 체크박스 플랜을 헤더 + task 목록으로 파싱한다.

    Returns:
        {
            "header": str,   # 첫 '### Task' 줄 이전 전체 (Goal/Architecture/
                              # Global Constraints 등) — 모든 task 프롬프트에
                              # 공통으로 포함된다.
            "tasks": [
                {"task": int, "title": str, "name": str, "raw": str},
                ...  # task 번호 오름차순
            ],
        }
    raw: 그 task의 '### Task N: <Title>' 줄부터 다음 '### Task' 줄 직전(또는
    파일 끝)까지 원문 그대로.
    name: <Title>을 kebab-case로 슬러그화한 것 (커밋 제목/브랜치 폴백에 사용).
    """
    matches = list(_TASK_HEADER_RE.finditer(text))
    header = text[: matches[0].start()] if matches else text
    tasks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        tasks.append({
            "task": int(m.group(1)),
            "title": m.group(2),
            "name": _slugify(m.group(2)),
            "raw": text[start:end].rstrip() + "\n",
        })
    tasks.sort(key=lambda t: t["task"])
    return {"header": header, "tasks": tasks}


def now_kst() -> str:
    return datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%dT%H:%M:%S%z")


def state_path_for(plan_path: Path) -> Path:
    return plan_path.with_suffix("").with_suffix(".state.json")


def load_or_create_state(plan_path: Path, parsed: dict) -> dict:
    sp = state_path_for(plan_path)
    if sp.exists():
        return json.loads(sp.read_text(encoding="utf-8"))
    state = {
        "plan_file": plan_path.name,
        "created_at": now_kst(),
        "completed_at": None,
        "tasks": [
            {
                "task": t["task"], "name": t["name"], "status": "pending",
                "attempts": [], "started_at": None, "completed_at": None,
                "summary": None, "commit_subject": None,
                "error_message": None, "blocked_reason": None,
            }
            for t in parsed["tasks"]
        ],
    }
    save_state(plan_path, state)
    return state


def save_state(plan_path: Path, state: dict) -> None:
    sp = state_path_for(plan_path)
    sp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


@contextlib.contextmanager
def progress_indicator(label: str):
    """터미널 진행 표시기. with 문으로 사용하며 .elapsed 로 경과 시간을 읽는다."""
    frames = "◐◓◑◒"
    stop = threading.Event()
    t0 = time.monotonic()

    def _animate():
        idx = 0
        while not stop.wait(0.12):
            sec = int(time.monotonic() - t0)
            sys.stderr.write(f"\r{frames[idx % len(frames)]} {label} [{sec}s]")
            sys.stderr.flush()
            idx += 1
        sys.stderr.write("\r" + " " * (len(label) + 20) + "\r")
        sys.stderr.flush()

    th = threading.Thread(target=_animate, daemon=True)
    th.start()
    info = types.SimpleNamespace(elapsed=0.0)
    try:
        yield info
    finally:
        stop.set()
        th.join()
        info.elapsed = time.monotonic() - t0


class StepExecutor:
    """플랜 안의 task들을 순차 실행하는 하네스."""

    MAX_RETRIES = 3
    FEAT_MSG = "feat({phase}): step {num} — {name}"
    FAIL_MSG = "wip({phase}): step {num} — {name} (FAILED)"
    # 매 step 프롬프트에 주입할 가드레일 문서 (docs/ 하위). glob이 아닌 명시 목록이다.
    # 코드 구현에 직접 필요한 기술 가드레일만 둔다. ROBOT_GUIDE.md(사람용 워크플로우 안내)는
    # 제외 — 3-PC/CPU-fallback 등 step에 필요한 규칙은 이미 CLAUDE.md로 주입된다.
    # 새 가드레일 문서를 추가하려면 여기에 파일명을 등록하라.
    GUARDRAIL_DOCS = ("ARCHITECTURE.md", "ADR.md")
    TZ = timezone(timedelta(hours=9))

    def __init__(self, plan_path_arg: str, *, auto_push: bool = False, model: str = ""):
        self._root = str(ROOT)
        self._auto_push = auto_push
        self._model = model
        # plan/state 관련 필드는 Task 3에서 채운다 — 지금은 자리만 잡아둔다.

    # --- timestamps ---

    def _stamp(self) -> str:
        return now_kst()

    # --- JSON I/O ---

    @staticmethod
    def _read_json(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(p: Path, data: dict):
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- git ---

    def _run_git(self, *args) -> subprocess.CompletedProcess:
        cmd = ["git"] + list(args)
        return subprocess.run(cmd, cwd=self._root, capture_output=True, text=True)

    def _checkout_branch(self):
        # Task 5에서 플랜 파일명 기반 feature-name 브랜치 로직으로 다시 만든다.
        raise NotImplementedError

    def _current_head(self) -> Optional[str]:
        """현재 HEAD의 커밋 SHA. 커밋이 하나도 없으면 None."""
        r = self._run_git("rev-parse", "HEAD")
        return r.stdout.strip() if r.returncode == 0 else None

    # --- guardrails & context ---

    def _load_guardrails(self) -> str:
        # docs/*.md 전체를 glob하지 않고 GUARDRAIL_DOCS 명시 목록만 주입한다.
        # 이유: 기존 코드가 있는 repo에 harness를 덧붙일 때, 그 repo의 무관한 docs/*.md가
        # 매 step 프롬프트에 섞여 노이즈/토큰 낭비가 되는 것을 방지한다(silent footgun).
        sections = []
        claude_md = ROOT / "CLAUDE.md"
        if claude_md.exists():
            sections.append(f"## 프로젝트 규칙 (CLAUDE.md)\n\n{claude_md.read_text()}")
        docs_dir = ROOT / "docs"
        for name in self.GUARDRAIL_DOCS:
            doc = docs_dir / name
            if doc.exists():
                sections.append(f"## {doc.stem}\n\n{doc.read_text()}")
        return "\n\n---\n\n".join(sections)


if __name__ == "__main__":
    pass
