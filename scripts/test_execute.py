"""
execute.py 리팩터링 안전망 테스트.
리팩터링 전후 동작이 동일한지 검증한다.
"""

import json
import os
import subprocess
import sys
import textwrap
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
    """테스트용 StepExecutor 인스턴스. git 호출은 별도 mock 필요."""
    with patch.object(ex, "ROOT", tmp_project):
        inst = ex.StepExecutor.__new__(ex.StepExecutor)
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
        before = datetime.now(ex.StepExecutor.TZ).replace(microsecond=0)
        result = executor._stamp()
        after = datetime.now(ex.StepExecutor.TZ).replace(microsecond=0) + timedelta(seconds=1)
        parsed = datetime.strptime(result, "%Y-%m-%dT%H:%M:%S%z")
        assert before <= parsed <= after


# ---------------------------------------------------------------------------
# _read_json / _write_json
# ---------------------------------------------------------------------------

class TestJsonHelpers:
    def test_roundtrip(self, tmp_path):
        data = {"key": "값", "nested": [1, 2, 3]}
        p = tmp_path / "test.json"
        ex.StepExecutor._write_json(p, data)
        loaded = ex.StepExecutor._read_json(p)
        assert loaded == data

    def test_save_ensures_ascii_false(self, tmp_path):
        p = tmp_path / "test.json"
        ex.StepExecutor._write_json(p, {"한글": "테스트"})
        raw = p.read_text()
        assert "한글" in raw
        assert "\\u" not in raw

    def test_save_indented(self, tmp_path):
        p = tmp_path / "test.json"
        ex.StepExecutor._write_json(p, {"a": 1})
        raw = p.read_text()
        assert "\n" in raw

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ex.StepExecutor._read_json(tmp_path / "nope.json")


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
            inst = ex.StepExecutor.__new__(ex.StepExecutor)
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
