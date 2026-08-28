"""
execute.py 리팩터링 안전망 테스트.
리팩터링 전후 동작이 동일한지 검증한다.
"""

import json
import os
import subprocess
import sys
import textwrap
import types
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import execute as ex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path):
    """phases/, CLAUDE.md, docs/ 를 갖춘 임시 프로젝트 구조."""
    phases_dir = tmp_path / "phases"
    phases_dir.mkdir()

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Rules\n- rule one\n- rule two")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    # GUARDRAIL_DOCS 에 등록된 문서 (주입 대상)
    (docs_dir / "ARCHITECTURE.md").write_text("# Architecture\nSome content")
    (docs_dir / "ADR.md").write_text("# ADR\nDecisions")
    # 등록되지 않은 무관한 문서 (주입되면 안 됨 — 기존 repo 오염 시나리오)
    (docs_dir / "UNRELATED.md").write_text("# Unrelated\nproject's own doc")

    return tmp_path


@pytest.fixture
def executor(tmp_project):
    """테스트용 PlanExecutor 인스턴스. git 호출은 별도 mock 필요."""
    with patch.object(ex, "ROOT", tmp_project):
        inst = ex.PlanExecutor.__new__(ex.PlanExecutor)
    inst._root = str(tmp_project)
    return inst


# ---------------------------------------------------------------------------
# _stamp (= 이전 now_iso)
# ---------------------------------------------------------------------------

class TestStamp:
    def test_returns_kst_timestamp(self, executor):
        result = executor._stamp()
        assert "+0900" in result

    def test_format_is_iso(self, executor):
        result = executor._stamp()
        dt = datetime.strptime(result, "%Y-%m-%dT%H:%M:%S%z")
        assert dt.tzinfo is not None

    def test_is_current_time(self, executor):
        before = datetime.now(ex.PlanExecutor.TZ).replace(microsecond=0)
        result = executor._stamp()
        after = datetime.now(ex.PlanExecutor.TZ).replace(microsecond=0) + timedelta(seconds=1)
        parsed = datetime.strptime(result, "%Y-%m-%dT%H:%M:%S%z")
        assert before <= parsed <= after


# ---------------------------------------------------------------------------
# _read_json / _write_json
# ---------------------------------------------------------------------------

class TestJsonHelpers:
    def test_roundtrip(self, tmp_path):
        data = {"key": "값", "nested": [1, 2, 3]}
        p = tmp_path / "test.json"
        ex.PlanExecutor._write_json(p, data)
        loaded = ex.PlanExecutor._read_json(p)
        assert loaded == data

    def test_save_ensures_ascii_false(self, tmp_path):
        p = tmp_path / "test.json"
        ex.PlanExecutor._write_json(p, {"한글": "테스트"})
        raw = p.read_text()
        assert "한글" in raw
        assert "\\u" not in raw

    def test_save_indented(self, tmp_path):
        p = tmp_path / "test.json"
        ex.PlanExecutor._write_json(p, {"a": 1})
        raw = p.read_text()
        assert "\n" in raw

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ex.PlanExecutor._read_json(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# _load_guardrails
# ---------------------------------------------------------------------------

class TestLoadGuardrails:
    def test_loads_claude_md_and_allowlisted_docs(self, executor, tmp_project):
        with patch.object(ex, "ROOT", tmp_project):
            result = executor._load_guardrails()
        assert "# Rules" in result
        assert "rule one" in result
        assert "# Architecture" in result
        assert "# ADR" in result

    def test_unlisted_doc_is_not_injected(self, executor, tmp_project):
        # 기존 repo의 무관한 docs/*.md 가 가드레일에 섞이지 않아야 한다 (footgun 방지).
        with patch.object(ex, "ROOT", tmp_project):
            result = executor._load_guardrails()
        assert "Unrelated" not in result
        assert "project's own doc" not in result

    def test_sections_separated_by_divider(self, executor, tmp_project):
        with patch.object(ex, "ROOT", tmp_project):
            result = executor._load_guardrails()
        assert "---" in result

    def test_docs_in_allowlist_order(self, executor, tmp_project):
        # GUARDRAIL_DOCS 순서(ARCHITECTURE → ADR)대로 주입된다.
        with patch.object(ex, "ROOT", tmp_project):
            result = executor._load_guardrails()
        assert result.index("Architecture") < result.index("ADR")

    def test_no_claude_md(self, executor, tmp_project):
        (tmp_project / "CLAUDE.md").unlink()
        with patch.object(ex, "ROOT", tmp_project):
            result = executor._load_guardrails()
        assert "CLAUDE.md" not in result
        assert "Architecture" in result

    def test_no_docs_dir(self, executor, tmp_project):
        import shutil
        shutil.rmtree(tmp_project / "docs")
        with patch.object(ex, "ROOT", tmp_project):
            result = executor._load_guardrails()
        assert "Rules" in result
        assert "Architecture" not in result

    def test_empty_project(self, tmp_path):
        with patch.object(ex, "ROOT", tmp_path):
            inst = ex.PlanExecutor.__new__(ex.PlanExecutor)
            result = inst._load_guardrails()
        assert result == ""


# ---------------------------------------------------------------------------
# progress_indicator (= 이전 Spinner)
# ---------------------------------------------------------------------------

class TestProgressIndicator:
    def test_context_manager(self):
        import time
        with ex.progress_indicator("test") as pi:
            time.sleep(0.15)
        assert pi.elapsed >= 0.1

    def test_elapsed_increases(self):
        import time
        with ex.progress_indicator("test") as pi:
            time.sleep(0.2)
        assert pi.elapsed > 0


# ---------------------------------------------------------------------------
# parse_plan
# ---------------------------------------------------------------------------

class TestParsePlan:
    SAMPLE = """# Example Plan

**Goal:** build the thing

## Global Constraints

- constraint one

---

### Task 1: Core Policy Network

**Files:**
- Create: `src/policy.py`

- [ ] **Step 1: do it**

body text

### Task 2: Env Wrapper

- [ ] **Step 1: do it**

more body
"""

    def test_header_excludes_first_task(self):
        result = ex.parse_plan(self.SAMPLE)
        assert "### Task 1" not in result["header"]
        assert "Global Constraints" in result["header"]

    def test_two_tasks_parsed_in_order(self):
        result = ex.parse_plan(self.SAMPLE)
        assert [t["task"] for t in result["tasks"]] == [1, 2]

    def test_title_and_slug(self):
        result = ex.parse_plan(self.SAMPLE)
        assert result["tasks"][0]["title"] == "Core Policy Network"
        assert result["tasks"][0]["name"] == "core-policy-network"

    def test_raw_spans_to_next_task_header(self):
        result = ex.parse_plan(self.SAMPLE)
        assert "### Task 1: Core Policy Network" in result["tasks"][0]["raw"]
        assert "src/policy.py" in result["tasks"][0]["raw"]
        assert "### Task 2" not in result["tasks"][0]["raw"]

    def test_last_task_raw_runs_to_eof(self):
        result = ex.parse_plan(self.SAMPLE)
        assert "more body" in result["tasks"][1]["raw"]

    def test_no_tasks_returns_empty_list(self):
        result = ex.parse_plan("# Just a header\n\nno tasks here")
        assert result["tasks"] == []
        assert "Just a header" in result["header"]

    def test_slug_handles_punctuation(self):
        text = "### Task 1: Fix DB/Cache (v2)!\n\nbody\n"
        result = ex.parse_plan(text)
        assert result["tasks"][0]["name"] == "fix-db-cache-v2"


# ---------------------------------------------------------------------------
# plan state (create/load/save)
# ---------------------------------------------------------------------------

class TestPlanState:
    PARSED = {
        "header": "# P",
        "tasks": [
            {"task": 1, "title": "A", "name": "a", "raw": "..."},
            {"task": 2, "title": "B", "name": "b", "raw": "..."},
        ],
    }

    def test_state_path_swaps_extension(self, tmp_path):
        plan = tmp_path / "2026-08-27-foo.md"
        assert ex.state_path_for(plan) == tmp_path / "2026-08-27-foo.state.json"

    def test_creates_from_parsed_when_missing(self, tmp_path):
        plan = tmp_path / "foo.md"
        state = ex.load_or_create_state(plan, self.PARSED)
        assert [t["task"] for t in state["tasks"]] == [1, 2]
        assert state["tasks"][0]["status"] == "pending"
        assert state["tasks"][0]["attempts"] == []
        assert ex.state_path_for(plan).exists()

    def test_loads_existing_without_overwriting(self, tmp_path):
        plan = tmp_path / "foo.md"
        ex.load_or_create_state(plan, self.PARSED)
        state = ex.load_or_create_state(plan, self.PARSED)  # 두 번째 호출
        state["tasks"][0]["status"] = "completed"
        ex.save_state(plan, state)
        reloaded = ex.load_or_create_state(plan, self.PARSED)
        assert reloaded["tasks"][0]["status"] == "completed"

    def test_save_then_load_roundtrip(self, tmp_path):
        plan = tmp_path / "foo.md"
        state = ex.load_or_create_state(plan, self.PARSED)
        state["completed_at"] = "2026-08-27T00:00:00+0900"
        ex.save_state(plan, state)
        raw = json.loads(ex.state_path_for(plan).read_text())
        assert raw["completed_at"] == "2026-08-27T00:00:00+0900"


# ---------------------------------------------------------------------------
# _build_task_context
# ---------------------------------------------------------------------------

class TestBuildTaskContext:
    def test_includes_completed_with_summary(self):
        state = {"tasks": [
            {"task": 1, "name": "setup", "status": "completed", "summary": "did setup"},
            {"task": 2, "name": "core", "status": "pending", "summary": None},
        ]}
        result = ex.PlanExecutor._build_task_context(state)
        assert "Task 1 (setup): did setup" in result
        assert "core" not in result

    def test_empty_when_no_completed(self):
        state = {"tasks": [{"task": 1, "name": "a", "status": "pending", "summary": None}]}
        assert ex.PlanExecutor._build_task_context(state) == ""

    def test_has_header(self):
        state = {"tasks": [{"task": 1, "name": "a", "status": "completed", "summary": "s"}]}
        assert "이전 Task 산출물" in ex.PlanExecutor._build_task_context(state)


# ---------------------------------------------------------------------------
# PlanExecutor fixture (Task 4~)
# ---------------------------------------------------------------------------

@pytest.fixture
def plan_executor(tmp_project):
    plan_path = tmp_project / "docs" / "superpowers" / "plans" / "foo.md"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("# P\n\n### Task 1: Core\n\nbody\n")
    with patch.object(ex, "ROOT", tmp_project):
        inst = ex.PlanExecutor(plan_path)
    inst._root = str(tmp_project)
    return inst


# ---------------------------------------------------------------------------
# _build_preamble
# ---------------------------------------------------------------------------

class TestBuildPreamble:
    def test_includes_guardrails(self, plan_executor):
        result = plan_executor._build_preamble("# P", "GUARDRAIL_TEXT", "")
        assert "GUARDRAIL_TEXT" in result

    def test_includes_plan_header(self, plan_executor):
        result = plan_executor._build_preamble("PLAN_HEADER_TEXT", "", "")
        assert "PLAN_HEADER_TEXT" in result

    def test_includes_task_context(self, plan_executor):
        result = plan_executor._build_preamble("", "", "TASK_CTX_TEXT")
        assert "TASK_CTX_TEXT" in result

    def test_no_retry_section_by_default(self, plan_executor):
        result = plan_executor._build_preamble("", "", "")
        assert "이전 시도 실패" not in result

    def test_retry_section_with_prev_error(self, plan_executor):
        result = plan_executor._build_preamble("", "", "", prev_error="boom")
        assert "boom" in result
        assert "이전 시도 실패" in result

    def test_references_state_file_not_index_json(self, plan_executor):
        result = plan_executor._build_preamble("", "", "")
        assert "index.json" not in result
        assert ".state.json" in result


# ---------------------------------------------------------------------------
# _invoke_claude
# ---------------------------------------------------------------------------

class TestInvokeClaude:
    def test_uses_task_raw_not_a_file(self, plan_executor):
        task = {"task": 1, "name": "core", "raw": "### Task 1: Core\n\nUNIQUE_MARKER\n"}
        mock_result = MagicMock(returncode=0, stdout="{}", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            plan_executor._invoke_claude(task, preamble="PREAMBLE_TEXT")
        prompt = mock_run.call_args[0][0][-1]
        assert "UNIQUE_MARKER" in prompt
        assert "PREAMBLE_TEXT" in prompt

    def test_saves_output_json_named_by_task(self, plan_executor, tmp_project):
        task = {"task": 3, "name": "core", "raw": "x"}
        mock_result = MagicMock(returncode=0, stdout='{"ok": true}', stderr="")
        with patch("subprocess.run", return_value=mock_result):
            plan_executor._invoke_claude(task, preamble="p")
        out = plan_executor._plan_path.parent / f"{plan_executor._plan_path.stem}.task3.output.json"
        assert out.exists()

    def test_timeout_is_1800(self, plan_executor):
        mock_result = MagicMock(returncode=0, stdout="{}", stderr="")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            plan_executor._invoke_claude({"task": 1, "name": "a", "raw": "x"}, preamble="p")
        assert mock_run.call_args[1]["timeout"] == 1800


# ---------------------------------------------------------------------------
# _commit_task (squash + retry history)
# ---------------------------------------------------------------------------

class TestCommitTask:
    def test_squashes_into_single_commit(self, plan_executor):
        calls = []
        def fake_git(*args):
            calls.append(args)
            if args[:2] == ("log", "--reverse"):
                return MagicMock(returncode=0, stdout="wip: a\nfix: b\n", stderr="")
            if args[:2] == ("diff", "--cached"):
                return MagicMock(returncode=1)
            return MagicMock(returncode=0, stdout="", stderr="")
        plan_executor._run_git = fake_git
        plan_executor._commit_task(1, "core", "abc123", attempts=[
            {"attempt": 1, "status": "completed"},
        ])
        commit_call = next(c for c in calls if c[0] == "commit")
        assert "feat(" in commit_call[commit_call.index("-m") + 1]

    def test_no_changes_skips_commit(self, plan_executor):
        def fake_git(*args):
            if args[:2] == ("diff", "--cached"):
                return MagicMock(returncode=0)  # 변경 없음
            return MagicMock(returncode=0, stdout="", stderr="")
        plan_executor._run_git = fake_git
        plan_executor._commit_task(1, "core", "abc123", attempts=[{"attempt": 1, "status": "completed"}])

    def test_failed_task_uses_wip_not_feat(self, plan_executor):
        calls = []
        def fake_git(*args):
            calls.append(args)
            if args[:2] == ("log", "--reverse"):
                return MagicMock(returncode=0, stdout="wip: attempt\n", stderr="")
            if args[:2] == ("diff", "--cached"):
                return MagicMock(returncode=1)
            return MagicMock(returncode=0, stdout="", stderr="")
        plan_executor._run_git = fake_git
        plan_executor._commit_task(1, "core", "abc123", attempts=[
            {"attempt": 1, "status": "error", "error_message": "boom"},
            {"attempt": 2, "status": "error", "error_message": "boom2"},
            {"attempt": 3, "status": "error", "error_message": "boom3"},
        ], failed=True)
        commit_call = next(c for c in calls if c[0] == "commit")
        assert "wip(" in commit_call[commit_call.index("-m") + 1]
        assert "(FAILED)" in commit_call[commit_call.index("-m") + 1]

    def test_retry_history_in_body_when_multiple_attempts(self, plan_executor):
        calls = []
        def fake_git(*args):
            calls.append(args)
            if args[:2] == ("log", "--reverse"):
                return MagicMock(returncode=0, stdout="wip: a\n", stderr="")
            if args[:2] == ("diff", "--cached"):
                return MagicMock(returncode=1)
            return MagicMock(returncode=0, stdout="", stderr="")
        plan_executor._run_git = fake_git
        plan_executor._commit_task(1, "core", "abc123", attempts=[
            {"attempt": 1, "status": "error", "error_message": "ImportError: X"},
            {"attempt": 2, "status": "completed"},
        ])
        commit_call = next(c for c in calls if c[0] == "commit")
        body = commit_call[commit_call.index("-m") + 3]
        assert "ImportError: X" in body
        assert "attempt 1" in body and "attempt 2" in body

    def test_no_retry_section_for_single_attempt(self, plan_executor):
        calls = []
        def fake_git(*args):
            calls.append(args)
            if args[:2] == ("log", "--reverse"):
                return MagicMock(returncode=0, stdout="wip: a\n", stderr="")
            if args[:2] == ("diff", "--cached"):
                return MagicMock(returncode=1)
            return MagicMock(returncode=0, stdout="", stderr="")
        plan_executor._run_git = fake_git
        plan_executor._commit_task(1, "core", "abc123", attempts=[{"attempt": 1, "status": "completed"}])
        commit_call = next(c for c in calls if c[0] == "commit")
        body = commit_call[commit_call.index("-m") + 3]
        assert "Retries" not in body


# ---------------------------------------------------------------------------
# _check_blockers
# ---------------------------------------------------------------------------

class TestCheckBlockers:
    def test_error_task_exits_1(self, plan_executor):
        state = {"tasks": [{"task": 1, "status": "error", "error_message": "boom"}]}
        with pytest.raises(SystemExit) as e:
            plan_executor._check_blockers(state)
        assert e.value.code == 1

    def test_blocked_task_exits_2(self, plan_executor):
        state = {"tasks": [{"task": 1, "status": "blocked", "blocked_reason": "need key"}]}
        with pytest.raises(SystemExit) as e:
            plan_executor._check_blockers(state)
        assert e.value.code == 2

    def test_all_completed_is_noop(self, plan_executor):
        state = {"tasks": [{"task": 1, "status": "completed"}]}
        plan_executor._check_blockers(state)  # 예외 없이 통과


# ---------------------------------------------------------------------------
# _execute_single_task
# ---------------------------------------------------------------------------

class TestExecuteSingleTask:
    def _make_state(self):
        return {
            "plan_file": "foo.md", "created_at": "t", "completed_at": None,
            "tasks": [{"task": 1, "name": "core", "status": "pending", "attempts": [],
                       "started_at": None, "completed_at": None, "summary": None,
                       "commit_subject": None, "error_message": None, "blocked_reason": None}],
        }

    def test_success_on_first_attempt_records_one_attempt(self, plan_executor, tmp_project):
        state = self._make_state()
        ex.save_state(plan_executor._plan_path, state)

        def fake_invoke(task, preamble):
            s = ex.load_or_create_state(plan_executor._plan_path, plan_executor._parsed)
            s["tasks"][0]["status"] = "completed"
            s["tasks"][0]["summary"] = "done"
            ex.save_state(plan_executor._plan_path, s)
            return {"exitCode": 0}
        plan_executor._invoke_claude = fake_invoke
        plan_executor._commit_task = MagicMock()
        plan_executor._current_head = MagicMock(return_value="sha0")

        with patch.object(ex, "progress_indicator") as pi_cm:
            pi_cm.return_value.__enter__.return_value = types.SimpleNamespace(elapsed=1.0)
            result = plan_executor._execute_single_task({"task": 1, "name": "core", "raw": "x"}, "", state)

        assert result is True
        final = ex.load_or_create_state(plan_executor._plan_path, plan_executor._parsed)
        assert len(final["tasks"][0]["attempts"]) == 1
        assert final["tasks"][0]["attempts"][0]["status"] == "completed"

    def test_retries_up_to_max_then_marks_error(self, plan_executor):
        state = self._make_state()
        ex.save_state(plan_executor._plan_path, state)

        def fake_invoke(task, preamble):
            s = ex.load_or_create_state(plan_executor._plan_path, plan_executor._parsed)
            s["tasks"][0]["status"] = "error"
            s["tasks"][0]["error_message"] = "AC failed"
            ex.save_state(plan_executor._plan_path, s)
            return {"exitCode": 0}
        plan_executor._invoke_claude = fake_invoke
        plan_executor._commit_task = MagicMock()
        plan_executor._current_head = MagicMock(return_value="sha0")

        with patch.object(ex, "progress_indicator") as pi_cm:
            pi_cm.return_value.__enter__.return_value = types.SimpleNamespace(elapsed=1.0)
            with pytest.raises(SystemExit) as e:
                plan_executor._execute_single_task({"task": 1, "name": "core", "raw": "x"}, "", state)

        assert e.value.code == 1
        final = ex.load_or_create_state(plan_executor._plan_path, plan_executor._parsed)
        assert len(final["tasks"][0]["attempts"]) == 3
        assert final["tasks"][0]["status"] == "error"
        plan_executor._commit_task.assert_called_once()
        assert plan_executor._commit_task.call_args.kwargs.get("failed") is True


# ---------------------------------------------------------------------------
# _execute_all_tasks (checkpoint batching)
# ---------------------------------------------------------------------------

class TestExecuteAllTasks:
    def test_runs_until_no_checkpoint_limit(self, plan_executor):
        plan_executor._parsed = {"header": "", "tasks": [
            {"task": 1, "name": "a", "raw": "x"}, {"task": 2, "name": "b", "raw": "y"},
        ]}
        plan_executor._total = 2
        calls = []
        def fake_single(task, guardrails, state):
            calls.append(task["task"])
            s = ex.load_or_create_state(plan_executor._plan_path, plan_executor._parsed)
            for t in s["tasks"]:
                if t["task"] == task["task"]:
                    t["status"] = "completed"
            ex.save_state(plan_executor._plan_path, s)
            return True
        plan_executor._execute_single_task = fake_single
        done = plan_executor._execute_all_tasks(guardrails="", checkpoint_every=None)
        assert done is True
        assert calls == [1, 2]

    def test_stops_after_checkpoint_every(self, plan_executor):
        plan_executor._parsed = {"header": "", "tasks": [
            {"task": 1, "name": "a", "raw": "x"}, {"task": 2, "name": "b", "raw": "y"},
            {"task": 3, "name": "c", "raw": "z"},
        ]}
        plan_executor._total = 3
        calls = []
        def fake_single(task, guardrails, state):
            calls.append(task["task"])
            s = ex.load_or_create_state(plan_executor._plan_path, plan_executor._parsed)
            for t in s["tasks"]:
                if t["task"] == task["task"]:
                    t["status"] = "completed"
            ex.save_state(plan_executor._plan_path, s)
            return True
        plan_executor._execute_single_task = fake_single
        plan_executor._print_batch_summary = MagicMock()
        done = plan_executor._execute_all_tasks(guardrails="", checkpoint_every=2)
        assert done is False
        assert calls == [1, 2]
        plan_executor._print_batch_summary.assert_called_once()


# ---------------------------------------------------------------------------
# run() / _finalize() / main()
# ---------------------------------------------------------------------------

class TestRunAndFinalize:
    def test_finalize_writes_completed_at_and_commits(self, plan_executor):
        state = {"plan_file": "foo.md", "created_at": "t", "completed_at": None, "tasks": []}
        ex.save_state(plan_executor._plan_path, state)
        calls = []
        def fake_git(*args):
            calls.append(args)
            if args[:2] == ("diff", "--cached"):
                return MagicMock(returncode=1)
            return MagicMock(returncode=0, stdout="", stderr="")
        plan_executor._run_git = fake_git
        plan_executor._finalize()
        final = ex.load_or_create_state(plan_executor._plan_path, plan_executor._parsed)
        assert final["completed_at"] is not None
        assert any(c[0] == "commit" for c in calls)


class TestMainCli:
    def test_no_args_exits(self):
        with patch("sys.argv", ["execute.py"]):
            with pytest.raises(SystemExit):
                ex.main()

    def test_nonexistent_plan_exits(self, tmp_path):
        with patch("sys.argv", ["execute.py", str(tmp_path / "nope.md")]):
            with pytest.raises(SystemExit):
                ex.main()

    def test_checkpoint_every_parsed_as_int(self, tmp_project):
        plan = tmp_project / "docs" / "superpowers" / "plans" / "foo.md"
        plan.parent.mkdir(parents=True)
        plan.write_text("# P\n\n### Task 1: A\n\nbody\n")
        with patch.object(ex, "ROOT", tmp_project), \
             patch("sys.argv", ["execute.py", str(plan), "--checkpoint-every", "3"]), \
             patch.object(ex.PlanExecutor, "run") as mock_run:
            ex.main()
        mock_run.assert_called_once()
