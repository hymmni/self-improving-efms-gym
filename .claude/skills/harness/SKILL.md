---
name: harness
description: 이 프로젝트(self-improving-gym)의 개발 워크플로우. 새 구현 작업을 설계·계획·실행할 때, 또는 scripts/execute.py로 플랜을 실행할 때 사용한다.
---

이 프로젝트는 Harness 프레임워크를 사용한다. 아래 워크플로우에 따라 작업을 진행하라.

---

## 워크플로우

### A. 탐색

`/docs/` 하위 문서(ARCHITECTURE, ADR 등)를 읽고 프로젝트의 아키텍처·설계 의도를 파악한다. 필요시 Explore 에이전트를 병렬로 사용한다.

### B. 논의

구현을 위해 구체화하거나 기술적으로 결정해야 할 사항이 있으면 사용자에게 제시하고 논의한다.
설계 전반에 브레인스토밍이 필요하면 `superpowers:brainstorming`을 먼저 쓴다.

### C. 플랜 작성

사용자가 구현 계획 작성을 지시하면 `superpowers:writing-plans` 스킬로
`docs/superpowers/plans/YYYY-MM-DD-<feature>.md`에 checkbox task 플랜을 작성한다.
이 하네스는 그 플랜을 실행 단위로 그대로 쓴다 — 별도의 `phases/`, `step{N}.md` 포맷은
더 이상 쓰지 않는다.

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

#### 커밋 전략 (task별 squash)

브랜치/커밋 구조는 다음과 같다:

```
main
 └─ feat/{feature-name}
      ├─ feat({feature}): task 1 — core-policy   ← task 1의 모든 중간 커밋이 1개로 압축
      ├─ feat({feature}): task 2 — env-wrapper   ← task 2의 모든 중간 커밋이 1개로 압축
      └─ ...
```

- task 진행 중 Claude는 자유롭게 여러 번 커밋한다(재시도 노이즈 포함).
- task가 **completed**(또는 최종 **error**)에 도달하면, execute.py가 task 시작 시점의 HEAD로 `git reset --soft`한 뒤, 그 사이의 모든 변경을 **단일 커밋**으로 압축한다.
- 압축 커밋의 제목은 `feat({feature}): task {N} — {name}` (CLAUDE.md의 Scoped Conventional Commits 규칙). **본문에는 원본 중간 커밋 제목들이 오래된 순으로 요약·기록**되고, 시도가 2회 이상이었다면 재시도 이력도 함께 기록되어 추적성을 유지한다.
- 결과적으로 `feat/{feature-name}` 브랜치 히스토리는 "task = 커밋 1개"로 깔끔하게 정렬되며, 상세 맥락은 커밋 본문 + `experiments/` 기록에 남는다.

> **압축 범위는 task 단위뿐이다.** main 병합 시에는 압축하지 않는다 (각 task 커밋을 그대로 보존한다). execute.py는 main을 절대 건드리지 않는다.

#### main 병합 (수동 / opt-in 헬퍼)

main 병합은 **하네스가 자동으로 하지 않는다.** 보통 사용자가 직접 수행하며, 권장 시퀀스는 다음과 같다:

1. local `main` 으로 checkout 후 `origin/main` 을 fast-forward pull
2. `feat/{feature-name}` 에서 `main` 을 rebase (히스토리 선형 유지)
3. `main` 으로 돌아와 `--no-ff` 로 병합 (task 커밋들을 보존하면서 병합 지점 명시)

이 시퀀스를 그대로 인코딩한 opt-in 헬퍼가 있다. **명시적으로 실행할 때만** 동작하며, rebase 충돌 시 자동 abort 후 안내하고, force push는 하지 않는다:

```bash
python3 scripts/merge_to_main.py [feat-branch]          # 생략 시 현재 브랜치
python3 scripts/merge_to_main.py feat/0-mvp --push      # 병합 후 origin/main push
python3 scripts/merge_to_main.py feat/0-mvp --yes       # 확인 프롬프트 생략
```

> **주의**: rebase는 feature 브랜치 히스토리를 재작성한다. 이미 `--push`로 공유된(다른 PC가 pull한) feature 브랜치라면, 병합 후 그 브랜치는 종료된 것으로 간주하고 재사용·재push하지 마라 (origin과 발산하므로 force push가 필요해진다).

에러 복구:

- **error 발생 시**: `<plan>.state.json`에서 해당 task의 `status`를 `"pending"`으로 바꾸고 `error_message`를 삭제한 뒤 재실행한다. 실패한 task는 `wip({feature}): task N — name (FAILED)` 커밋으로 보존돼 있다 — 재실행 전 그 작업이 불필요하면 `git reset --hard HEAD~1`로 정리한 뒤 재실행하면 히스토리가 깔끔하다 (그대로 둬도 다음 성공 커밋이 그 위에 쌓일 뿐 동작엔 문제없다).
- **blocked 발생 시**: `blocked_reason`에 적힌 사유를 해결한 뒤, `status`를 `"pending"`으로 바꾸고 `blocked_reason`을 삭제한 뒤 재실행한다.
