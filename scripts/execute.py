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


_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-")


def _feature_name_from_plan(plan_path: Path) -> str:
    stem = plan_path.stem
    return _DATE_PREFIX_RE.sub("", stem) or stem


class PlanExecutor:
    """플랜 안의 task들을 순차 실행하는 하네스."""

    MAX_RETRIES = 3
    FEAT_MSG = "feat({feature}): task {num} — {name}"
    FAIL_MSG = "wip({feature}): task {num} — {name} (FAILED)"
    # 매 task 프롬프트에 주입할 가드레일 문서 (docs/ 하위). glob이 아닌 명시 목록이다.
    # 코드 구현에 직접 필요한 기술 가드레일만 둔다. ROBOT_GUIDE.md(사람용 워크플로우 안내)는
    # 제외 — 3-PC/CPU-fallback 등 task에 필요한 규칙은 이미 CLAUDE.md로 주입된다.
    # 새 가드레일 문서를 추가하려면 여기에 파일명을 등록하라.
    GUARDRAIL_DOCS = ("ARCHITECTURE.md", "ADR.md")
    TZ = timezone(timedelta(hours=9))

    def __init__(self, plan_path: Path, *, auto_push: bool = False, model: str = ""):
        self._root = str(ROOT)
        self._plan_path = plan_path
        if not self._plan_path.is_file():
            print(f"ERROR: {self._plan_path} not found")
            sys.exit(1)
        self._parsed = parse_plan(self._plan_path.read_text(encoding="utf-8"))
        self._feature = _feature_name_from_plan(plan_path)
        self._auto_push = auto_push
        self._model = model
        self._total = len(self._parsed["tasks"])

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
        # 매 task 프롬프트에 섞여 노이즈/토큰 낭비가 되는 것을 방지한다(silent footgun).
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

    @staticmethod
    def _build_task_context(state: dict) -> str:
        lines = [
            f"- Task {t['task']} ({t['name']}): {t['summary']}"
            for t in state["tasks"]
            if t["status"] == "completed" and t.get("summary")
        ]
        if not lines:
            return ""
        return "## 이전 Task 산출물\n\n" + "\n".join(lines) + "\n\n"

    def _build_preamble(self, header: str, guardrails: str, task_context: str,
                        prev_error: Optional[str] = None) -> str:
        state_rel = state_path_for(self._plan_path).relative_to(Path(self._root))
        commit_example = self.FEAT_MSG.format(feature=self._feature, num="N", name="<task-name>")
        guardrail_section = f"## 가드레일 (반드시 준수)\n\n{guardrails}\n\n---\n\n" if guardrails else ""
        plan_header_section = f"## 플랜 공통 정보\n\n{header}\n\n---\n\n" if header.strip() else ""
        retry_section = ""
        if prev_error:
            retry_section = (
                f"\n## ⚠ 이전 시도 실패 — 아래 에러를 반드시 참고하여 수정하라\n\n"
                f"{prev_error}\n\n"
            )
        return (
            f"당신은 이 프로젝트의 개발자입니다. 아래 task를 수행하세요.\n\n"
            f"{guardrail_section}{plan_header_section}"
            f"{task_context}{retry_section}"
            f"## 작업 규칙\n\n"
            f"1. 이전 task에서 작성된 코드를 확인하고 일관성을 유지하라.\n"
            f"2. 이 task에 명시된 작업만 수행하라. 추가 기능이나 파일을 만들지 마라.\n"
            f"3. 기존 테스트를 깨뜨리지 마라.\n"
            f"4. AC(Acceptance Criteria)/각 checkbox step을 직접 실행해 검증하라.\n"
            f"5. `{state_rel}`에서 이 task 항목의 status를 업데이트하라:\n"
            f"   - 통과 → \"completed\" + \"summary\"(산출물 한 줄 요약) + \"commit_subject\"\n"
            f"     (CLAUDE.md Scoped 커밋 규칙: 모듈 스코프 + 서술형 영어 문장,\n"
            f"     예: `feat(env): add observable obstacle fields`)\n"
            f"   - {self.MAX_RETRIES}회 시도 후 실패 → \"error\" + \"error_message\"\n"
            f"   - 사용자 개입 필요 → \"blocked\" + \"blocked_reason\" 후 즉시 중단\n"
            f"6. 모든 변경사항을 커밋하라(CLAUDE.md Scoped 커밋 규칙, 여러 번 커밋해도 좋다 —\n"
            f"   하네스가 task 종료 시 하나로 squash하며 제목은 5번의 commit_subject를 쓴다,\n"
            f"   누락 시 폴백: `{commit_example}`).\n\n"
            f"---\n\n"
        )

    # --- Claude 호출 ---

    def _invoke_claude(self, task: dict, preamble: str) -> dict:
        task_num = task["task"]
        prompt = preamble + task["raw"]
        try:
            cmd = ["claude", "-p", "--dangerously-skip-permissions", "--output-format", "json"]
            step_model = self._model
            if step_model:
                cmd += ["--model", step_model]
            cmd.append(prompt)
            result = subprocess.run(cmd, cwd=self._root, capture_output=True, text=True, timeout=1800)
            exit_code, out_text, err_text = result.returncode, result.stdout, result.stderr
            if exit_code != 0:
                print(f"\n  WARN: Claude가 비정상 종료됨 (code {exit_code})")
                if err_text:
                    print(f"  stderr: {err_text[:500]}")
        except subprocess.TimeoutExpired as e:
            print(f"\n  WARN: Claude 실행 타임아웃 ({e.timeout}초 지속됨)")
            exit_code, out_text, err_text = 124, "", f"TimeoutExpired: executed longer than {e.timeout}s"
        except Exception as e:
            print(f"\n  WARN: Claude 실행 중 예외 발생: {e}")
            exit_code, out_text, err_text = 1, "", str(e)

        output = {"task": task_num, "name": task["name"], "exitCode": exit_code,
                  "stdout": out_text, "stderr": err_text}
        out_path = self._plan_path.parent / f"{self._plan_path.stem}.task{task_num}.output.json"
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        return output

    # --- squash 커밋 ---

    def _commit_task(self, task_num: int, task_name: str,
                     task_start_sha: Optional[str], attempts: list, *, failed: bool = False) -> None:
        original_subjects = []
        if task_start_sha:
            r = self._run_git("log", "--reverse", "--format=%s", f"{task_start_sha}..HEAD")
            if r.returncode == 0:
                original_subjects = [ln for ln in r.stdout.splitlines() if ln.strip()]
            self._run_git("reset", "--soft", task_start_sha)

        self._run_git("add", "-A")
        if self._run_git("diff", "--cached", "--quiet").returncode == 0:
            return

        subject = "" if failed else self._task_commit_subject(task_num)
        if not subject:
            tmpl = self.FAIL_MSG if failed else self.FEAT_MSG
            subject = tmpl.format(feature=self._feature, num=task_num, name=task_name)
        body = self._compose_squash_body(task_num, task_name, original_subjects, attempts, failed=failed)
        r = self._run_git("commit", "-m", subject, "-m", body)
        if r.returncode == 0:
            note = f" ({len(original_subjects)} commits squashed)" if original_subjects else ""
            print(f"  Commit: {subject}{note}")
        else:
            print(f"  WARN: task 커밋 실패: {r.stderr.strip()}")

    def _task_commit_subject(self, task_num: int) -> str:
        state = load_or_create_state(self._plan_path, self._parsed)
        for t in state["tasks"]:
            if t["task"] == task_num:
                subj = str(t.get("commit_subject") or "").strip()
                return subj.splitlines()[0].strip() if subj else ""
        return ""

    def _compose_squash_body(self, task_num: int, task_name: str,
                             original_subjects: list, attempts: list, *, failed: bool = False) -> str:
        lines = [f"Squash all working commits from task {task_num} ({task_name}) into one."]
        if failed:
            state_rel = state_path_for(self._plan_path).relative_to(Path(self._root))
            lines += ["", f"Task did NOT pass: status=error after {self.MAX_RETRIES} retries.",
                      f"Work preserved for debugging; see {state_rel} for error_message."]
        lines.append("")
        if original_subjects:
            lines.append(f"Original commits ({len(original_subjects)}, oldest first):")
            lines += [f"- {s}" for s in original_subjects]
        else:
            lines.append("No intermediate commits; staged working-tree changes only.")
        if len(attempts) > 1:
            lines += ["", f"Retries: {len(attempts)} attempts."]
            for a in attempts:
                if a["status"] == "completed":
                    lines.append(f"- attempt {a['attempt']}: completed")
                else:
                    lines.append(f"- attempt {a['attempt']}: {a['status']} — {a.get('error_message', '')}")
        return "\n".join(lines)


if __name__ == "__main__":
    pass
