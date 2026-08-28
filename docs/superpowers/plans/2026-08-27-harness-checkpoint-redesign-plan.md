# Harness Checkpoint Redesign Implementation Plan

> **실행 방식 — 표준 writing-plans 안내에서 의도적으로 벗어남:** 이 플랜은
> `superpowers:subagent-driven-development`나 `superpowers:executing-plans`로 실행하지
> **않는다.** 두 스킬 모두 "task 사이에 멈추지 않는다"는 전제라, 이 플랜이 구현하려는
> 바로 그 문제(설계 판단이 사용자 확인 없이 넘어감)를 재현한다. 대신 **현재 세션에서
> Task를 하나씩 구현 → 무엇을/왜 했는지 설명 → 사용자 승인 → 다음 Task** 순서로
> 진행한다 (스펙의 "평소" 티어). Task 사이 커밋은 각 Task 끝에서 하나로 만든다.

**Goal:** `scripts/execute.py`가 `phases/{name}/index.json` + `step{N}.md` 대신
writing-plans 형식의 checkbox 플랜(`docs/superpowers/plans/*.md`)을 직접 실행하도록
재작성하고, 자동화 정도를 고르는 3단계 티어(평소/바쁨/급함)와 재시도 이력 가시성을
갖춘다.

**Architecture:** `scripts/execute.py`는 단일 스크립트로 유지한다(레포의 `scripts/`
컨벤션 — 각 스크립트가 자기완결형 단일 파일). 플랜 마크다운을 파싱해 얻은 task 목록과,
그 옆에 자동 생성/갱신되는 `<plan>.state.json`(사람이 손으로 안 쓰는 파생 상태 파일)이
새로운 데이터 소스가 된다. git/가드레일/진행 표시기 등 포맷과 무관한 헬퍼는 그대로
재사용한다.

**Tech Stack:** Python 3(표준 라이브러리만 — `argparse`, `json`, `re`, `subprocess`,
`pathlib`), `pytest` + `unittest.mock`(기존 `scripts/test_execute.py`와 동일한 스타일:
`patch.object(ex, "ROOT", ...)`로 루트를 바꾸고, `executor._run_git = fake_git`으로 git
호출을 스텁하며, `subprocess.run`을 patch해 `claude -p` 호출을 모킹한다. 실제 git
저장소나 실제 `claude` 바이너리는 테스트에서 쓰지 않는다).

**Spec:** `docs/superpowers/specs/2026-08-27-harness-checkpoint-redesign-design.md`

## Global Constraints

- **레거시 포맷 지원 없음.** `phases/{name}/index.json` + `step{N}.md`를 읽는 코드는
  전부 제거한다. 되돌아가지 않는다 — 이 플랜은 `phases/4-diffusion-si`의 남은 step
  4(variant-eval)가 **별도 세션에서, 현재(수정 전) `execute.py`로 먼저 완료된 뒤**에만
  실행한다. (Task 1 착수 전 사용자에게 재확인할 것.)
- **squash 단위는 항상 task 1개.** 체크포인트 티어와 무관하게 git 히스토리 단위는
  바뀌지 않는다.
- **top-level `phases/index.json`은 더 이상 건드리지 않는다.** 과거 기록으로 그대로
  둔다(읽지도 쓰지도 않음).
- **재시도 정책은 유지.** AC 실패 시 최대 3회 자동 재시도(`MAX_RETRIES = 3`)는 그대로
  두되, 매 시도 결과를 `attempts` 배열에 기록하고 성공하더라도 squash 커밋 본문에
  재시도 이력을 남긴다.
- **커밋 메시지는 CLAUDE.md의 Scoped Conventional Commits 규칙**(영어, 서술형 제목,
  본문 글머리기호)을 따른다 — 이 규칙은 내가 이 플랜을 구현하며 남기는 커밋에 적용되는
  것이지, `execute.py`가 생성하는 squash 커밋 템플릿(`FEAT_MSG` 등)과는 별개다.
- **테스트는 기존 `scripts/test_execute.py`의 모킹 스타일을 따른다** — 실제 git
  저장소나 실제 `claude` CLI를 부르지 않는다.

---

### Task 1: 레거시 포맷 코드 제거, 포맷-무관 헬퍼만 남기기

**Files:**
- Modify: `scripts/execute.py`
- Modify: `scripts/test_execute.py`

**Interfaces:**
- Consumes: 없음 (첫 task)
- Produces: 이후 task들이 그대로 재사용할 헬퍼 —
  `progress_indicator(label: str)` (contextmanager, 미변경),
  `StepExecutor._stamp(self) -> str` (미변경, TZ=KST),
  `StepExecutor._read_json(p: Path) -> dict` / `_write_json(p: Path, data: dict)` (staticmethod, 미변경),
  `StepExecutor._run_git(self, *args) -> subprocess.CompletedProcess` (미변경),
  `StepExecutor._current_head(self) -> Optional[str]` (미변경),
  `StepExecutor._load_guardrails(self) -> str` (미변경, `CLAUDE.md` + `docs/ARCHITECTURE.md` + `docs/ADR.md` 주입).
  이 클래스는 이후 task에서 `PlanExecutor`로 이름이 바뀌지만, 이 task에서는 클래스명은
  그대로 두고 **내용만** 정리한다(이름 변경은 Task 4에서 한 번에).

이번 task에서 **삭제**하는 것: `__init__`의 `self._phases_dir/self._phase_dir/
self._top_index_file` 및 phase index 읽기, `_checkout_branch`의 `phase_name` 기반 브랜치
계산(다음 task에서 새로 만들 것이므로 일단 메서드 본문만 비우고 `NotImplementedError`),
`_step_commit_subject`, `_commit_step`, `_compose_squash_body`, `_update_top_index`,
`_build_step_context`, `_build_preamble`, `_invoke_claude`, `_print_header`,
`_check_blockers`, `_ensure_created_at`, `_execute_single_step`, `_print_step_eta`,
`_execute_all_steps`, `_finalize`, `main()`. 이 메서드들은 phase-index 포맷에 강하게
결합돼 있어 그대로 두면 다음 task들과 충돌한다 — 통째로 지우고 Task 3~5에서 새 이름·새
시그니처로 다시 만든다. `_print_step_eta`가 쓰던 ETA 추정 기능은 이번 재설계 범위 밖이라
되살리지 않는다 — 그 전용 상수/필드인 `MODEL_DEFAULT_SECS`와 `self._step_times`도 함께
삭제하고 이후 task 어디에서도 다시 만들지 않는다.

- [ ] **Step 1: 살아남을 테스트만 남기고 나머지 삭제**

`scripts/test_execute.py`에서 다음 클래스만 남긴다(내용은 그대로): `TestStamp`,
`TestJsonHelpers`, `TestLoadGuardrails`, `TestProgressIndicator`. 나머지
(`TestBuildStepContext`, `TestBuildPreamble`, `TestUpdateTopIndex`, `TestCheckoutBranch`,
`TestCommitStep`, `TestInvokeClaude`, `TestMainCli`, `TestCheckBlockers`)와 이들이 쓰던
`phase_dir`/`top_index` fixture는 삭제한다. `tmp_project`, `executor` fixture는 남기되
`executor` fixture 본문에서 `inst._phase_dir = phase_dir` 등 삭제된 필드 참조 줄을
지운다(`ex.StepExecutor("0-mvp")` 생성자 호출도 다음 스텝에서 `__init__`을 비우고 나면
깨지므로, 임시로 `ex.StepExecutor.__new__(ex.StepExecutor)`로 바꾸고 `inst._root =
str(tmp_project)`만 남긴다).

- [ ] **Step 2: 테스트 실행해 남은 것만 통과하는지 확인**

Run: `cd /home/hymm/Projects/self-improving-gym && python -m pytest scripts/test_execute.py -v`
Expected: `TestStamp`, `TestJsonHelpers`, `TestLoadGuardrails`, `TestProgressIndicator`의
테스트만 존재하고 전부 PASS. (이 시점엔 `execute.py`를 아직 안 건드렸으므로 자동으로
통과해야 한다 — 실패하면 Step 1에서 지운 fixture 참조가 남아있는 것이다.)

- [ ] **Step 3: `execute.py`에서 위에 나열한 메서드/블록 삭제**

`StepExecutor.__init__`을 다음으로 축소한다(phase 관련 필드 제거, root/model/retry
기록용 리스트만 남김):

```python
def __init__(self, plan_path_arg: str, *, auto_push: bool = False, model: str = ""):
    self._root = str(ROOT)
    self._auto_push = auto_push
    self._model = model
    # plan/state 관련 필드는 Task 3에서 채운다 — 지금은 자리만 잡아둔다.
```

나머지 나열된 메서드는 파일에서 완전히 삭제한다(빈 본문으로 남기지 않는다 — 다음
task들에서 다른 이름/시그니처로 새로 작성하므로 지금 자리를 차지하면 혼란만 준다).
`run()`도 삭제한다(Task 5에서 새로 작성). `MODEL_DEFAULT_SECS` 클래스 상수도 삭제한다.
`MAX_RETRIES`, `GUARDRAIL_DOCS`, `TZ` 클래스 상수는 내용 변경 없이 그대로 둔다.
`FEAT_MSG`/`FAIL_MSG` 상수는 이름은 남기되(삭제하지 않음), 문자열 내용은 Task 4에서
`{phase}`/`step` 표현을 `{feature}`/`task`로 바꾼다 — 지금 이 task에서는 건드리지 않는다.

- [ ] **Step 4: 테스트 재실행으로 회귀 없음 확인**

Run: `python -m pytest scripts/test_execute.py -v`
Expected: Step 2와 동일하게 전부 PASS (삭제한 메서드를 참조하는 테스트가 없어야 한다).

- [ ] **Step 5: Commit**

```bash
git add scripts/execute.py scripts/test_execute.py
git commit -m "$(cat <<'EOF'
refactor(harness): strip legacy phases-index execution path

- remove phase-directory/step-file specific methods (checkout,
  commit-squash, preamble, invoke, blockers, main loop) — they will
  be rebuilt against a writing-plans markdown plan instead
- keep format-agnostic helpers: progress_indicator, git wrapper,
  json helpers, guardrail loader
- trim scripts/test_execute.py to the surviving helper tests only
EOF
)"
```

---

### Task 2: 플랜 마크다운 파서

**Files:**
- Modify: `scripts/execute.py`
- Modify: `scripts/test_execute.py`

**Interfaces:**
- Consumes: 없음 (순수 함수, 파일 I/O 없음 — 텍스트만 받는다)
- Produces:
  ```python
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
  ```
  이후 task들이 `parse_plan`을 import해 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성**

`scripts/test_execute.py`에 추가:

```python
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
```

- [ ] **Step 2: 테스트 실행해 실패 확인**

Run: `python -m pytest scripts/test_execute.py::TestParsePlan -v`
Expected: FAIL — `AttributeError: module 'execute' has no attribute 'parse_plan'`

- [ ] **Step 3: 최소 구현 작성**

`scripts/execute.py`에 (클래스 밖, 모듈 레벨 함수로) 추가:

```python
import re

_TASK_HEADER_RE = re.compile(r"^###\s+Task\s+(\d+):\s*(.+?)\s*$", re.MULTILINE)


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip().lower())
    return slug.strip("-")


def parse_plan(text: str) -> dict:
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
```

- [ ] **Step 4: 테스트 재실행해 통과 확인**

Run: `python -m pytest scripts/test_execute.py::TestParsePlan -v`
Expected: 전부 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/execute.py scripts/test_execute.py
git commit -m "$(cat <<'EOF'
feat(harness): add writing-plans markdown parser

- parse_plan() splits a checkbox plan into a shared header (Goal/
  Architecture/Global Constraints) and ordered Task sections
- each task carries its raw markdown span plus a kebab-case name
  slug derived from its title, for use in commit subjects
EOF
)"
```

---

### Task 3: 플랜 상태 파일 (`.state.json`)

**Files:**
- Modify: `scripts/execute.py`
- Modify: `scripts/test_execute.py`

**Interfaces:**
- Consumes: `parse_plan(text) -> dict` (Task 2)
- Produces:
  ```python
  def state_path_for(plan_path: Path) -> Path:
      """<plan>.md -> <plan>.state.json (같은 디렉터리, 같은 stem)."""

  def load_or_create_state(plan_path: Path, parsed: dict) -> dict:
      """state 파일이 있으면 그대로 읽어 반환. 없으면 parsed["tasks"]로부터
      새로 만들어 저장하고 반환한다.

      각 task 항목의 형태:
      {
          "task": int, "name": str, "status": "pending",
          "attempts": [], "started_at": None, "completed_at": None,
          "summary": None, "commit_subject": None,
          "error_message": None, "blocked_reason": None,
      }
      최상위: {"plan_file": <plan_path.name>, "created_at": <ISO>,
               "completed_at": None, "tasks": [...]}
      """

  def save_state(plan_path: Path, state: dict) -> None:
      """state_path_for(plan_path)에 저장 (json, indent=2, ensure_ascii=False)."""
  ```
  이후 task들이 이 세 함수를 그대로 쓴다. `created_at`용 타임스탬프는
  `StepExecutor._stamp`를 재사용하지 않고(모듈 레벨 함수라 인스턴스 불필요) 아래처럼
  모듈 레벨 헬퍼로 뺀다:
  ```python
  def now_kst() -> str:
      return datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%dT%H:%M:%S%z")
  ```
  (`StepExecutor._stamp`는 Task 1에서 남겨둔 그대로 두되, 내부적으로
  `return now_kst()`를 호출하도록 한 줄만 바꿔 중복을 없앤다.)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
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
```

- [ ] **Step 2: 테스트 실행해 실패 확인**

Run: `python -m pytest scripts/test_execute.py::TestPlanState -v`
Expected: FAIL — `AttributeError: module 'execute' has no attribute 'state_path_for'`

- [ ] **Step 3: 구현 작성**

```python
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
```

`_stamp` 메서드 본문을 `return now_kst()`로 교체한다.

- [ ] **Step 4: 테스트 재실행**

Run: `python -m pytest scripts/test_execute.py -v`
Expected: 이전 task들 테스트 포함 전부 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/execute.py scripts/test_execute.py
git commit -m "$(cat <<'EOF'
feat(harness): add plan state file (create/load/save)

- <plan>.state.json is derived from parse_plan()'s task list on
  first run, never hand-authored, and carries per-task status/
  attempts/summary/commit_subject fields
- factor _stamp's KST timestamp into a module-level now_kst() shared
  by the new state helpers
EOF
)"
```

---

### Task 4: 단일 task 실행 엔진 (컨텍스트/재시도/squash)

이 task가 가장 크다 — 이전 `_build_preamble`/`_invoke_claude`/`_commit_step`/
`_compose_squash_body`/`_step_commit_subject`를 새 데이터 소스(플랜 텍스트 + state)에
맞게 재작성해 하나의 "task 1개 실행" 단위로 합친다. `PlanExecutor`로 클래스명을
바꾸는 것도 이 task에서 한다(다음 task부터는 이 이름을 쓴다).

**Files:**
- Modify: `scripts/execute.py`
- Modify: `scripts/test_execute.py`

**Interfaces:**
- Consumes: `parse_plan`, `state_path_for`/`load_or_create_state`/`save_state` (Task 2, 3)
- Produces (모두 `PlanExecutor`의 메서드, 이전 `StepExecutor`를 리네임):
  ```python
  class PlanExecutor:
      def __init__(self, plan_path: Path, *, auto_push: bool = False, model: str = ""): ...

      def _build_task_context(self, state: dict) -> str:
          """완료된 task들의 summary를 누적해 다음 task 프롬프트에 넣을 문자열을
          만든다. 형식: '## 이전 Task 산출물\\n\\n- Task 1 (a): <summary>\\n...'"""

      def _build_preamble(self, header: str, guardrails: str, task_context: str,
                          prev_error: Optional[str] = None) -> str: ...

      def _invoke_claude(self, task: dict, preamble: str) -> dict:
          """task['raw']를 프롬프트 본문으로 쓴다 (step{N}.md 파일을 더 이상 읽지
          않는다). 출력은 <plan_path.parent>/<plan_stem>.task{N}.output.json에 저장."""

      def _commit_task(self, task_num: int, task_name: str,
                       task_start_sha: Optional[str], attempts: list,
                       *, failed: bool = False) -> None:
          """기존 _commit_step과 동일한 squash 로직 + attempts가 2개 이상이면
          재시도 이력을 본문에 추가."""
  ```
  이 시그니처들은 Task 5가 그대로 호출한다.

- [ ] **Step 1: `_build_task_context` 실패 테스트**

```python
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
```

- [ ] **Step 2: 실행해 실패 확인**

Run: `python -m pytest scripts/test_execute.py::TestBuildTaskContext -v`
Expected: FAIL (클래스명 `StepExecutor`만 있고 `PlanExecutor`/`_build_task_context` 없음)

- [ ] **Step 3: 클래스 리네임 + `_build_task_context` 구현**

`class StepExecutor:` → `class PlanExecutor:`로 바꾸고 `__init__`을 아래로 교체:

```python
class PlanExecutor:
    MAX_RETRIES = 3
    FEAT_MSG = "feat({feature}): task {num} — {name}"
    FAIL_MSG = "wip({feature}): task {num} — {name} (FAILED)"
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
```

`_feature_name_from_plan`은 모듈 레벨 함수로 추가:

```python
_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-")


def _feature_name_from_plan(plan_path: Path) -> str:
    stem = plan_path.stem
    return _DATE_PREFIX_RE.sub("", stem) or stem
```

`_build_task_context`는 staticmethod로:

```python
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
```

- [ ] **Step 4: 실행해 통과 확인**

Run: `python -m pytest scripts/test_execute.py::TestBuildTaskContext -v`
Expected: PASS

- [ ] **Step 5: `_build_preamble` 실패 테스트**

```python
@pytest.fixture
def plan_executor(tmp_project):
    plan_path = tmp_project / "docs" / "superpowers" / "plans" / "foo.md"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("# P\n\n### Task 1: Core\n\nbody\n")
    with patch.object(ex, "ROOT", tmp_project):
        inst = ex.PlanExecutor(plan_path)
    inst._root = str(tmp_project)
    return inst


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
```

- [ ] **Step 6: 실행해 실패 확인**

Run: `python -m pytest scripts/test_execute.py::TestBuildPreamble -v`
Expected: FAIL — `_build_preamble` 없음

- [ ] **Step 7: `_build_preamble` 구현**

```python
def _build_preamble(self, header: str, guardrails: str, task_context: str,
                    prev_error: Optional[str] = None) -> str:
    state_rel = state_path_for(self._plan_path).relative_to(ROOT)
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
```

- [ ] **Step 8: 실행해 통과 확인**

Run: `python -m pytest scripts/test_execute.py::TestBuildPreamble -v`
Expected: PASS

- [ ] **Step 9: `_invoke_claude` 실패 테스트**

```python
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
```

- [ ] **Step 10: 실행해 실패 확인**

Run: `python -m pytest scripts/test_execute.py::TestInvokeClaude -v`
Expected: FAIL — `_invoke_claude` 없음

- [ ] **Step 11: `_invoke_claude` 구현**

```python
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
```

- [ ] **Step 12: 실행해 통과 확인**

Run: `python -m pytest scripts/test_execute.py::TestInvokeClaude -v`
Expected: PASS

- [ ] **Step 13: `_commit_task`(squash + 재시도 이력) 실패 테스트**

```python
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
        # 예외 없이 조용히 반환하면 통과 (commit 커맨드가 안 불림은 별도로 검증하지 않아도
        # diff --cached returncode 0 분기에서 즉시 return하는 게 기존 로직과 동일)

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
        body = commit_call[commit_call.index("-m") + 2]
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
        body = commit_call[commit_call.index("-m") + 2]
        assert "Retries" not in body
```

- [ ] **Step 14: 실행해 실패 확인**

Run: `python -m pytest scripts/test_execute.py::TestCommitTask -v`
Expected: FAIL — `_commit_task` 없음

- [ ] **Step 15: `_commit_task` 구현**

```python
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
        state_rel = state_path_for(self._plan_path).relative_to(ROOT)
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
```

- [ ] **Step 16: 실행해 통과 확인**

Run: `python -m pytest scripts/test_execute.py -v`
Expected: 전부 PASS

- [ ] **Step 17: Commit**

```bash
git add scripts/execute.py scripts/test_execute.py
git commit -m "$(cat <<'EOF'
feat(harness): add single-task execution engine (rename to PlanExecutor)

- rename StepExecutor -> PlanExecutor; it now owns a parsed plan
  (parse_plan) and a feature-name slug derived from the plan filename
- _build_task_context/_build_preamble read accumulated task summaries
  and the plan's shared header instead of a phase index
- _invoke_claude sends a task's raw markdown span as the prompt body
  instead of reading a separate step{N}.md file
- _commit_task/_compose_squash_body squash per task as before, and
  now append a condensed retry history to the commit body whenever a
  task needed more than one attempt
EOF
)"
```

---

### Task 5: 재시도 루프 + 체크포인트 티어 + CLI

**Files:**
- Modify: `scripts/execute.py`
- Modify: `scripts/test_execute.py`

**Interfaces:**
- Consumes: Task 2~4의 전부 (`parse_plan`, `load_or_create_state`/`save_state`,
  `PlanExecutor._build_task_context`/`_build_preamble`/`_invoke_claude`/`_commit_task`)
- Produces:
  ```python
  class PlanExecutor:
      def run(self) -> None: ...
      def _check_blockers(self, state: dict) -> None: ...
      def _execute_single_task(self, task: dict, guardrails: str, state: dict) -> bool: ...
      def _execute_all_tasks(self, guardrails: str, checkpoint_every: Optional[int]) -> bool:
          """모든 task가 끝났으면 True, checkpoint_every에 도달해 중간에 멈췄으면
          False를 반환한다."""
      def _print_batch_summary(self, state: dict, tasks_this_run: list) -> None: ...
      def _finalize(self) -> None: ...
  ```
  CLI: `python3 scripts/execute.py <plan.md> [--push] [--model MODEL] [--checkpoint-every N]`

- [ ] **Step 1: `_check_blockers` 실패 테스트**

```python
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
```

- [ ] **Step 2: 실행해 실패 확인**

Run: `python -m pytest scripts/test_execute.py::TestCheckBlockers -v`
Expected: FAIL

- [ ] **Step 3: `_check_blockers` 구현**

```python
def _check_blockers(self, state: dict) -> None:
    for t in reversed(state["tasks"]):
        if t["status"] == "error":
            print(f"\n  ✗ Task {t['task']} failed.")
            print(f"  Error: {t.get('error_message', 'unknown')}")
            print(f"  Fix and reset status to 'pending' to retry.")
            sys.exit(1)
        if t["status"] == "blocked":
            print(f"\n  ⏸ Task {t['task']} blocked.")
            print(f"  Reason: {t.get('blocked_reason', 'unknown')}")
            print(f"  Resolve and reset status to 'pending' to retry.")
            sys.exit(2)
        if t["status"] != "pending":
            break
```

- [ ] **Step 4: 실행해 통과 확인**

Run: `python -m pytest scripts/test_execute.py::TestCheckBlockers -v`
Expected: PASS

- [ ] **Step 5: `_execute_single_task` 재시도/attempts 기록 실패 테스트**

```python
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
```

- [ ] **Step 6: 실행해 실패 확인**

Run: `python -m pytest scripts/test_execute.py::TestExecuteSingleTask -v`
Expected: FAIL — `_execute_single_task` 없음

- [ ] **Step 7: `_execute_single_task` 구현**

```python
def _execute_single_task(self, task: dict, guardrails: str, state: dict) -> bool:
    task_num, task_name = task["task"], task["name"]
    prev_error = None
    task_start_sha = self._current_head()
    attempts: list = []

    for attempt in range(1, self.MAX_RETRIES + 1):
        current = load_or_create_state(self._plan_path, self._parsed)
        task_context = self._build_task_context(current)
        preamble = self._build_preamble(self._parsed["header"], guardrails, task_context, prev_error)

        tag = f"Task {task_num}/{self._total}: {task_name}"
        if attempt > 1:
            tag += f" [retry {attempt}/{self.MAX_RETRIES}]"

        with progress_indicator(tag) as pi:
            self._invoke_claude(task, preamble)
        elapsed = int(pi.elapsed)

        current = load_or_create_state(self._plan_path, self._parsed)
        entry = next(t for t in current["tasks"] if t["task"] == task_num)
        status = entry.get("status", "pending")
        ts = now_kst()

        if status == "completed":
            attempts.append({"attempt": attempt, "status": "completed"})
            entry["completed_at"] = ts
            entry["attempts"] = attempts
            save_state(self._plan_path, current)
            self._commit_task(task_num, task_name, task_start_sha, attempts)
            print(f"  ✓ Task {task_num}: {task_name} [{elapsed}s]")
            return True

        if status == "blocked":
            attempts.append({"attempt": attempt, "status": "blocked"})
            entry["blocked_at"] = ts
            entry["attempts"] = attempts
            save_state(self._plan_path, current)
            print(f"  ⏸ Task {task_num}: {task_name} blocked [{elapsed}s]")
            print(f"    Reason: {entry.get('blocked_reason', '')}")
            sys.exit(2)

        err_msg = entry.get("error_message", "Task did not update status")
        attempts.append({"attempt": attempt, "status": "error", "error_message": err_msg})

        if attempt < self.MAX_RETRIES:
            entry["status"] = "pending"
            entry["attempts"] = attempts
            save_state(self._plan_path, current)
            prev_error = err_msg
            print(f"  ↻ Task {task_num}: retry {attempt}/{self.MAX_RETRIES} — {err_msg}")
        else:
            entry["status"] = "error"
            entry["error_message"] = f"[{self.MAX_RETRIES}회 시도 후 실패] {err_msg}"
            entry["failed_at"] = ts
            entry["attempts"] = attempts
            save_state(self._plan_path, current)
            self._commit_task(task_num, task_name, task_start_sha, attempts, failed=True)
            print(f"  ✗ Task {task_num}: {task_name} failed after {self.MAX_RETRIES} attempts [{elapsed}s]")
            print(f"    Error: {err_msg}")
            sys.exit(1)

    return False  # unreachable
```

- [ ] **Step 8: 실행해 통과 확인**

Run: `python -m pytest scripts/test_execute.py::TestExecuteSingleTask -v`
Expected: PASS

- [ ] **Step 9: `_execute_all_tasks`(체크포인트 배치) 실패 테스트**

```python
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
```

- [ ] **Step 10: 실행해 실패 확인**

Run: `python -m pytest scripts/test_execute.py::TestExecuteAllTasks -v`
Expected: FAIL

- [ ] **Step 11: `_execute_all_tasks` + `_print_batch_summary` 구현**

```python
def _execute_all_tasks(self, guardrails: str, checkpoint_every: Optional[int]) -> bool:
    completed_this_run = []
    while True:
        state = load_or_create_state(self._plan_path, self._parsed)
        pending = next((t for t in state["tasks"] if t["status"] == "pending"), None)
        if pending is None:
            print("\n  All tasks completed!")
            return True

        task_num = pending["task"]
        task = next(t for t in self._parsed["tasks"] if t["task"] == task_num)
        if not pending.get("started_at"):
            pending["started_at"] = now_kst()
            save_state(self._plan_path, state)

        self._execute_single_task(task, guardrails, state)
        completed_this_run.append(task_num)

        if checkpoint_every is not None and len(completed_this_run) >= checkpoint_every:
            final_state = load_or_create_state(self._plan_path, self._parsed)
            self._print_batch_summary(final_state, completed_this_run)
            return False


def _print_batch_summary(self, state: dict, tasks_this_run: list) -> None:
    print(f"\n{'='*60}")
    print(f"  Checkpoint — {len(tasks_this_run)} task(s) completed this run")
    for t in state["tasks"]:
        if t["task"] not in tasks_this_run:
            continue
        print(f"  ✓ Task {t['task']} ({t['name']}): {t.get('summary', '')}")
        if len(t.get("attempts", [])) > 1:
            print(f"    (retried {len(t['attempts'])} attempts)")
        print(f"    commit: {t.get('commit_subject', '')}")
    print(f"  Review the diff/summary above, then re-run to continue:")
    print(f"    python3 scripts/execute.py {self._plan_path}")
    print(f"{'='*60}")
```

- [ ] **Step 12: 실행해 통과 확인**

Run: `python -m pytest scripts/test_execute.py::TestExecuteAllTasks -v`
Expected: PASS

- [ ] **Step 13: `run()` + `_finalize()` + `main()` 실패 테스트**

```python
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
```

- [ ] **Step 14: 실행해 실패 확인**

Run: `python -m pytest scripts/test_execute.py::TestRunAndFinalize scripts/test_execute.py::TestMainCli -v`
Expected: FAIL

- [ ] **Step 15: `run()`/`_finalize()`/`main()` 구현**

```python
def run(self, checkpoint_every: Optional[int] = None):
    self._print_header()
    state = load_or_create_state(self._plan_path, self._parsed)
    self._check_blockers(state)
    self._checkout_branch()
    guardrails = self._load_guardrails()
    plan_done = self._execute_all_tasks(guardrails, checkpoint_every)
    if plan_done:
        self._finalize()

def _print_header(self):
    print(f"\n{'='*60}")
    print(f"  Harness Plan Executor")
    print(f"  Plan: {self._plan_path.name} | Tasks: {self._total}")
    print(f"{'='*60}")

def _checkout_branch(self):
    branch = f"feat/{self._feature}"
    r = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
    if r.returncode != 0:
        print("  ERROR: git을 사용할 수 없거나 git repo가 아닙니다.")
        sys.exit(1)
    if r.stdout.strip() == branch:
        return
    r = self._run_git("rev-parse", "--verify", branch)
    r = self._run_git("checkout", branch) if r.returncode == 0 else self._run_git("checkout", "-b", branch)
    if r.returncode != 0:
        print(f"  ERROR: 브랜치 '{branch}' checkout 실패.")
        sys.exit(1)
    print(f"  Branch: {branch}")

def _finalize(self):
    state = load_or_create_state(self._plan_path, self._parsed)
    state["completed_at"] = now_kst()
    save_state(self._plan_path, state)

    self._run_git("add", "-A")
    if self._run_git("diff", "--cached", "--quiet").returncode != 0:
        msg = f"chore(harness): mark plan '{self._plan_path.name}' completed"
        body = f"- set completed_at in {state_path_for(self._plan_path).relative_to(ROOT)}"
        r = self._run_git("commit", "-m", msg, "-m", body)
        if r.returncode == 0:
            print(f"  ✓ {msg}")

    if self._auto_push:
        branch = f"feat/{self._feature}"
        r = self._run_git("push", "-u", "origin", branch)
        if r.returncode != 0:
            print(f"\n  ERROR: git push 실패: {r.stderr.strip()}")
            sys.exit(1)
        print(f"  ✓ Pushed to origin/{branch}")

    print(f"\n{'='*60}\n  Plan '{self._plan_path.name}' completed!\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Harness Plan Executor")
    parser.add_argument("plan", help="Path to a writing-plans markdown plan (docs/superpowers/plans/*.md)")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="N tasks per run before stopping for review (omit = run whole plan)")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = ROOT / plan_path
    if not plan_path.is_file():
        print(f"ERROR: {plan_path} not found")
        sys.exit(1)

    PlanExecutor(plan_path, auto_push=args.push, model=args.model).run(checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
```

- [ ] **Step 16: 전체 테스트 실행**

Run: `python -m pytest scripts/test_execute.py -v`
Expected: 전부 PASS

- [ ] **Step 17: Commit**

```bash
git add scripts/execute.py scripts/test_execute.py
git commit -m "$(cat <<'EOF'
feat(harness): add checkpoint-every tiers and wire up new CLI

- _execute_all_tasks stops after N completed tasks when
  --checkpoint-every is given, printing a consolidated batch summary
  instead of finalizing the plan
- _execute_single_task records every AC attempt (success or failure)
  so retry history survives even when a task eventually succeeds
- main() now takes a plan markdown path instead of a phases directory
  name; --checkpoint-every N selects the "바쁨" tier, omitting it
  keeps today's whole-plan "급함" behavior
EOF
)"
```

---

### Task 6: `harness.md` 재작성

**Files:**
- Modify: `.claude/commands/harness.md`

**Interfaces:**
- Consumes: Task 1~5에서 확정된 CLI/파일 계약 (`python3 scripts/execute.py <plan.md>
  [--checkpoint-every N] [--push] [--model M]`, `<plan>.state.json`)
- Produces: 없음 (문서만)

- [ ] **Step 1: "C. Step 설계"/"D. 파일 생성" 절을 writing-plans 안내로 교체**

`.claude/commands/harness.md`의 3~129번째 줄(현재 "B. 논의" 다음부터 "D-3.
step{N}.md" 끝까지)을 아래로 교체한다 — 브레인스토밍 이후 플랜은 `writing-plans`
스킬로 작성하며, 이 문서는 그 플랜을 실행하는 방법만 설명하도록 범위를 좁힌다:

```markdown
### B. 논의

구현을 위해 구체화하거나 기술적으로 결정해야 할 사항이 있으면 사용자에게 제시하고 논의한다.
설계 전반에 브레인스토밍이 필요하면 `superpowers:brainstorming`을 먼저 쓴다.

### C. 플랜 작성

사용자가 구현 계획 작성을 지시하면 `superpowers:writing-plans` 스킬로
`docs/superpowers/plans/YYYY-MM-DD-<feature>.md`에 checkbox task 플랜을 작성한다.
이 하네스는 그 플랜을 실행 단위로 그대로 쓴다 — 별도의 `phases/`, `step{N}.md` 포맷은
더 이상 쓰지 않는다.
```

- [ ] **Step 2: "E. 실행" 절을 3단계 티어로 교체**

기존 "E. 실행" 절(플래그 표, "커밋 전략" 앞부분)을 아래로 교체한다:

```markdown
### D. 실행 — 3단계 자율성 티어

플랜 문서는 티어와 무관하게 동일하다. 달라지는 건 몇 개 task를 실행한 뒤 멈춰서
보여주는가 뿐이다.

**평소 (기본값) — `execute.py` 미사용.** 현재 인터랙티브 세션에서 플랜을 읽고
Task를 하나씩 구현 → 설명 → 사용자 승인 → 다음 Task로 진행한다.

**바쁨 — task N개마다 멈춰서 검토:**

```bash
python3 scripts/execute.py docs/superpowers/plans/<file>.md --checkpoint-every 3
```

N개 task를 무인 실행한 뒤, 완료된 task들의 summary + 재시도 이력 + squash 커밋
제목을 종합해 보여주고 종료한다. 승인 = 같은 커맨드를 다시 실행하는 것 — 이미 끝난
task는 건너뛰고 다음 배치를 진행한다.

**급함 — 플랜 전체를 무인 실행, 끝에서만 검토:**

```bash
python3 scripts/execute.py docs/superpowers/plans/<file>.md
python3 scripts/execute.py docs/superpowers/plans/<file>.md --push
python3 scripts/execute.py docs/superpowers/plans/<file>.md --model opus
```

체크포인트 플래그 없이 실행하면 플랜 전체를 끝까지 무인으로 돌리고, error/blocked가
아닌 이상 끝에서만 멈춘다.

execute.py가 자동으로 처리하는 것:

- `feat/{feature-name}` 브랜치 생성/checkout (feature-name = 플랜 파일명에서 날짜
  접두어를 뺀 것)
- 가드레일 주입 — CLAUDE.md + docs/*.md 내용을 매 task 프롬프트에 포함
- 컨텍스트 누적 — 완료된 task의 summary를 다음 task 프롬프트에 전달
- 자가 교정 — AC 실패 시 최대 3회 재시도. **매 시도 결과가 `<plan>.state.json`의
  `attempts` 배열에 기록되고**, 최종 성공해도 재시도가 있었다면 squash 커밋 본문에
  그 이력이 남는다 (몇 번째 시도에서 무엇이 왜 실패했는지 최종 diff만 봐도 알 수 있다).
- **task별 squash 커밋** — 티어와 무관하게 항상 task 1개 = 커밋 1개.
- 상태 파일 — `<plan>.state.json`은 execute.py가 플랜에서 자동 생성/갱신한다.
  사람이 직접 쓰지 않는다.
```

- [ ] **Step 3: 커밋 전략/에러 복구 절의 "step"→"task" 용어 통일**

파일 나머지(현재 "#### 커밋 전략" 이후 ~ 끝)에서 `step`을 `task`로,
`phases/{task-name}/index.json`을 `<plan>.state.json`으로 바꾼다. "main 병합" 절
(`scripts/merge_to_main.py` 안내)은 내용 변경 없이 그대로 둔다.

- [ ] **Step 4: 검증**

Run: `grep -n "step{N}.md\|phases/{task-name}/index.json" .claude/commands/harness.md`
Expected: 결과 없음(모두 치환 완료)

- [ ] **Step 5: Commit**

```bash
git add .claude/commands/harness.md
git commit -m "$(cat <<'EOF'
docs(harness): rewrite workflow doc for plan-based 3-tier execution

- step design/file creation sections now point to superpowers:writing-plans
  instead of phases/{name}/step{N}.md
- execution section documents the 평소/바쁨/급함 tiers and the new
  execute.py <plan.md> [--checkpoint-every N] CLI
- unify step/task terminology and point at <plan>.state.json instead
  of phases/{name}/index.json throughout
EOF
)"
```

---

### Task 7: `CLAUDE.md` 개발 프로세스 절 갱신

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: Task 6에서 확정된 harness.md 서술(3단계 티어 명칭)
- Produces: 없음 (문서만)

- [ ] **Step 1: `## 📝 개발 프로세스`의 `scripts/execute.py` 문단에 티어 설명 추가**

`CLAUDE.md`의 "**`scripts/execute.py` (보조 도구, 예외적 사용)**" 문단 끝에 다음
문장을 추가한다: "실행 방식은 3단계 티어(평소=인터랙티브 task별 승인 / 바쁨=`--checkpoint-every N` / 급함=플랜 전체 무인 실행)로 나뉜다 — 상세는 `harness`
스킬(`.claude/commands/harness.md`) 참고." 기존 문장(설계 판단은 사용자와 먼저
논의 등)은 그대로 둔다 — 모순 없이 강화하는 방향이다.

- [ ] **Step 2: 검증**

Run: `grep -n "3단계 티어" CLAUDE.md`
Expected: 방금 추가한 줄이 출력됨

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(claude): note execute.py's 3-tier checkpoint model

- point to harness.md for the 평소/바쁨/급함 autonomy tiers, so the
  existing "execute.py is an exceptional tool" guidance links to the
  concrete mechanism that backs it
EOF
)"
```
