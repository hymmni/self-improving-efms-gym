# 하네스 실행 메커니즘 재설계: 신뢰/추적성 문제 해결

## 배경

이 프로젝트의 하네스(`scripts/execute.py` + `.claude/commands/harness.md`)는 phase를 여러
step으로 쪼개 `claude -p`로 헤드리스 실행하고, 각 step은 실행 가능한 AC(Acceptance
Criteria)로 검증하며, 실패하면 최대 3회까지 자동 재시도하고, step이 끝나면 중간 커밋들을
하나로 squash한다.

실제로 phase 1/3/5는 이 방식으로 잘 진행됐고 산출물(step summary 등)도 알차다. 하지만
반복적으로 발생한 문제가 있다:

> AI가 필요한 질문을 나에게 하지 않고 혼자서 개발을 진행해버려서, 내가 코드 구현 방식을
> 정확히 추적하지 못하고, 연구적으로 잘못된 구현이 뒤늦게(내가 결과물을 보고서야) 발견되는
> 일이 지속된다. 내 연구인데 내가 코드가 어떻게 돌아가는지 파악하기 어렵다. 하네스가
> 프로덕트 개발에 초점을 맞춘 느낌이라, 아무튼 AC만 통과하면 넘어가 버린다.

핵심 원인 분석:

1. `execute.py`는 `claude -p`(1회성, 비대화형) 호출로 step을 실행한다. Claude가 step 도중
   모호한 지점에서 사용자에게 되물을 방법이 구조적으로 없다 — 듣고 있는 사람이 없다. 즉
   "Claude가 질문하게 만들자"는 프롬프트 튜닝으로 풀리는 문제가 아니라, **멈추는 지점을
   어디에 두느냐**의 문제다.
2. AC는 "코드가 돌아가는가/테스트가 통과하는가"만 검증하며, "연구적으로 올바른 설계인가"는
   검증하지 않는다. step summary는 실제로 상당히 풍부한 연구적 판단·caveat를 담고 있지만
   (phase 5 step 0-3 실측 확인), 아무도 그걸 읽고 넘어가는 것을 강제하지 않는다 — squash되고
   다음 step으로 넘어가 버리면 그 시점의 판단을 되짚기 어려워진다.
3. AC 실패 시 3회 자동 재시도는 유지할 가치가 있지만(단순 문법 오류 등에는 유용), 성공하고
   나면 각 시도가 뭘 시도했다 실패했는지가 사실상 사라진다(최종 diff만 남는다).

이미 다른 세션에서 `CLAUDE.md` 레벨로 "기본 워크플로우 = 직접 세션 협업, execute.py = 설계가
이미 합의된 기계적 작업에만 쓰는 예외적 보조 도구"로 격하해 두었다 (이번 스펙에서 재논의하지
않음). 이 문서는 그 위에서, 실행 메커니즘 자체 — 플랜 포맷, 체크포인트 위치, 자동화 정도
선택, 재시도 가시성 — 를 재설계한다.

## 목표

1. 브레인스토밍 이후의 모든 다단계 작업이 **하나의 플랜 포맷**(writing-plans의 checkbox
   task 플랜)을 공유하게 한다 — 자동화 정도와 무관하게.
2. 평소 기본 워크플로우를 "task 단위로 구현→설명→승인"이 되도록 만들어, 연구 설계 판단이
   당신 모르게 넘어가는 지점을 없앤다.
3. 바쁠 때는 자동화 정도를 올릴 수 있는 명시적 선택지를 제공하되, 완전 자동(급함 티어)에서도
   최종적으로 무엇이 왜 그렇게 됐는지 재구성 가능하게 한다(재시도 가시성).
4. `execute.py`/`harness.md`의 기존 안전장치(브랜치 분리, step/task별 squash, 실패 시
   `wip(...)(FAILED)` 커밋 보존, `merge_to_main.py`를 통한 수동 병합)는 유지한다.

## 비목표

- `phases/1..5`의 기존 산출물 마이그레이션 (과거 기록으로 그대로 둔다 — 새 작업만 새 포맷)
- main 병합 정책 변경 (`scripts/merge_to_main.py` 워크플로우는 그대로)
- 모델 선택 가이드(`CLAUDE.md`의 fable/opus/sonnet/haiku 표) 변경
- `subagent-driven-development`/`executing-plans` superpowers 스킬 자체의 수정 — 이 프로젝트는
  그 두 스킬을 기본 워크플로우에 쓰지 않기로 한다(둘 다 "중간에 멈추지 않는다"는 전제라 이
  문제에 맞지 않음). 스킬 자체는 건드리지 않고, 이 프로젝트의 기본 워크플로우 문서에서 "쓰지
  않는다"고 명시하는 선에서 그친다.

## A. 플랜 포맷 통합

브레인스토밍 이후 구현 대상이 확정되면, `writing-plans` 스킬로 `docs/superpowers/plans/
YYYY-MM-DD-<feature>.md`에 checkbox task 플랜을 작성한다 — `### Task N: <이름>` 섹션마다
Files/Interfaces + bite-sized `- [ ]` step. 이것이 harness의 실행 단위가 된다: 하네스의
"step" = writing-plans의 "Task"로 1:1 대응시킨다. 별도 세분화 기준을 새로 만들지 않는다.

기존 `phases/{name}/index.json` + `phases/{name}/step{N}.md` 형식(사용자가 직접 시그니처
수준으로 설계하고 AC를 박아넣던 방식)은 폐기하고, 앞으로는 writing-plans 플랜이 그 역할을
대체한다. writing-plans 플랜에는 이미 파일 경로/인터페이스/테스트 커맨드가 들어가므로, 기존
harness.md의 "C. Step 설계" 절(시그니처 수준 지시, AC는 실행 가능한 커맨드 등)이 요구하던
정보와 대체로 겹친다 — 실질적 정보 손실 없이 포맷만 통합된다.

`phases/1..5`는 과거 기록이므로 마이그레이션하지 않고 그대로 둔다. `phases/index.json`(전체
현황 인덱스)도 과거 항목은 그대로 두되, 새 작업은 더 이상 이 인덱스에 추가하지 않는다(플랜
파일 목록 자체가 `docs/superpowers/plans/`에 있으므로 별도 top-level 인덱스가 필요 없음).

## B. 상태 추적

`execute.py`는 이제 `phases/*/index.json`을 읽는 대신 `docs/superpowers/plans/<file>.md`를
파싱해 `### Task N` 헤더로 task 경계를 찾는다. 사람이 손으로 관리하던 JSON을 대신해, 플랜과
같은 디렉터리에 **동반 상태 파일** `docs/superpowers/plans/<file>.state.json`을 첫 실행 시
플랜에서 자동 생성하고, 이후 실행마다 갱신한다:

```json
{
  "plan_file": "2026-08-27-example-plan.md",
  "created_at": "...",
  "tasks": [
    {
      "task": 1,
      "name": "core-policy",
      "status": "pending",
      "attempts": [],
      "started_at": null,
      "completed_at": null,
      "summary": null,
      "commit_subject": null
    }
  ]
}
```

필드 의미는 기존 `phases/*/index.json`의 step 레벨 필드와 동일하다(`status`:
pending/completed/error/blocked, `summary`, `commit_subject`, `error_message`,
`blocked_reason`, 타임스탬프). 차이는 다음 두 가지뿐이다:

- task 목록이 사람이 손으로 쓴 배열이 아니라 플랜 마크다운에서 파싱된다(plan이 source of
  truth, state 파일은 파생 캐시).
- `attempts` 배열이 추가된다 (아래 D절).

플랜 파일 자체(`.md`)는 실행 중 execute.py가 쓰지 않는다 — 상태는 전부 `.state.json`에만
쓴다. 이렇게 하면 사람이 쓴 플랜 문서와 기계가 갱신하는 상태를 분리해, 플랜 문서가 실행 중에
오염되지 않는다.

## C. 3단계 자율성 티어

플랜 문서는 티어와 무관하게 동일하다 — 달라지는 것은 **몇 개의 task를 실행한 뒤 멈춰서
보여주는가** 뿐이다.

### 평소 (기본값) — `execute.py` 미사용

현재 인터랙티브 세션에서 플랜을 직접 읽고, Task N을 구현하고, 무엇을 왜 했는지 설명한 뒤
당신의 승인을 기다렸다가 Task N+1로 넘어간다. 스크립트 개입 없음 — 오늘 이 대화와 동일한
직접 협업 모드. AC는 그 자리에서 같이 실행해서 확인한다.

### 바쁨 — `execute.py <plan> --checkpoint-every N` (기본 N=3)

N개 task를 무인으로 실행한 뒤, 그 배치의 결과(각 task의 summary + 재시도 이력 + squash
커밋 subject 목록)를 종합해 보여주고 프로세스가 종료된다. 실제 diff는 필요하면 조회
(`git log`/`git diff <배치 시작 SHA>..HEAD`)한다. 승인 방법은 별도 플래그 없이, **다음
`execute.py <plan> --checkpoint-every N` 재실행**이 곧 승인이다 — 이미 완료된 task는
건너뛰고 다음 배치를 진행한다(기존 execute.py의 "재진입 가능" 설계를 그대로 재사용). 수정이
필요하면 재실행 전에 직접 개입(코드 수정 후 커밋, 또는 해당 task를 pending으로 되돌려 재작업
지시)한다.

### 급함 — `execute.py <plan>` (체크포인트 플래그 없음)

플랜 전체를 무인으로 실행하고 끝에서만(또는 error/blocked 시) 멈춘다. 오늘의 execute.py와
동일한 동작이며, 플랜 포맷만 바뀐 것이다.

squash 단위는 티어와 무관하게 **task 단위**로 고정한다 — 체크포인트는 "언제 보여주는가"만
바꾸고 git 히스토리의 단위는 바꾸지 않는다. 즉 "바쁨" 티어에서도 N개 task는 N개의 개별
커밋으로 남는다(배치를 하나로 뭉개 커밋하지 않는다).

## D. 재시도 가시성

현재도 squash 커밋 본문에 중간 커밋 제목 목록(`original_subjects`)은 남지만, **시도별 AC
실패 사유**는 최종 성공 시 완전히 사라진다(단지 다음 시도 프롬프트에 `prev_error`로
전달됐다가 버려짐).

state 파일의 각 task에 `attempts` 배열을 추가한다:

```json
"attempts": [
  { "attempt": 1, "status": "error", "error_message": "...", "timestamp": "..." },
  { "attempt": 2, "status": "completed", "timestamp": "..." }
]
```

`execute.py`가 매 시도 종료 시(성공이든 실패든) 이 배열에 기록한다. squash 커밋 본문
작성 시(`_compose_squash_body` 상당 로직), 최종 성공이어도 실패한 시도가 있었다면 그
사유를 요약해 포함한다. 예:

```
Retries: 2 attempts.
- attempt 1: AC failed — ImportError: cannot import name 'DstgReward'
- attempt 2: succeeded
```

이렇게 하면 최종 diff만 봐서는 "한 번에 됐다"처럼 보이는 task도, 실제로 몇 번 헤맸고
무엇이 원인이었는지가 커밋 로그에 남는다. AC 실패 시 자동 재시도(최대 3회) 자체의 정책은
바꾸지 않는다.

## E. 문서/코드 변경 범위 (구현 단계에서 다룰 것, writing-plans에서 task로 분해)

- `scripts/execute.py`: `phases/*/index.json` 파싱 → 플랜 마크다운 파싱 + `.state.json`
  관리로 교체. `--checkpoint-every N` 플래그 추가. `attempts` 기록 및 squash 본문 반영.
- `.claude/commands/harness.md`: "C. Step 설계"/"D. 파일 생성" 절을 "writing-plans로 플랜
  작성" 절로 교체. "E. 실행" 절에 3단계 티어와 각 커맨드 설명 추가. 커밋 전략 절의 "step"
  표현을 "task"로 통일.
- `CLAUDE.md`: 개발 프로세스 절에 "기본 = 평소 티어(직접 협업), execute.py = 바쁨/급함
  티어에서 의도적으로 선택" 명시. 기존 "execute.py는 보조 도구, 예외적 사용" 문구와 정합성
  확인(모순 없음 — 강화하는 방향).
- 기존 `phases/` 디렉터리와 `scripts/execute.py`의 구 파싱 경로: 완전 제거할지, 과거 phase
  재실행 호환을 위해 당분간 남겨둘지는 구현 단계(writing-plans)에서 실제 코드를 보며 결정한다
  (이 스펙의 범위 밖 — 비목표 절 참고: 과거 phase는 재실행 대상이 아니므로 구 경로 처리
  코드는 삭제해도 무방해 보이지만, 최종 판단은 구현 시점에 코드베이스를 보고 내린다).

## 미해결/구현 시 확인할 점

- `--checkpoint-every N`의 기본값 3은 이 스펙 작성 중 제안값이다. 실사용 후 조정 가능.
- "바쁨" 티어에서 배치 중 하나의 task가 `blocked`(사용자 개입 필요)로 끝나면, 남은 배치는
  진행하지 않고 즉시 멈춘다 (기존 `_check_blockers` 동작을 그대로 승계).
- 플랜 마크다운 파싱 방식(정규식 vs 간단한 라인 파서)은 구현 단계에서 결정한다 — 이 스펙은
  "Task N 헤더로 경계를 나눈다"는 계약만 정의한다.
