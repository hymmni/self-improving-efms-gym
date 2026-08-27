# 명령어 총정리

모든 명령은 레포 루트(`~/Projects/self-improving-gym`)에서, `self-improving-gym` conda
환경으로 실행한다. GUI 뷰어는 X 디스플레이가 필요하다 — 원격 데스크톱/서버
모니터에서는 명령 앞에 `DISPLAY=:1` 을 붙인다.

```bash
conda activate self-improving-gym
cd ~/Projects/self-improving-gym
```

---

## 1. 실시간 뷰어 (GUI)

### 신맵 — 장애물 회피 환경 자동 재생 (`watch_obstacle.py`)
왼쪽 에이전트 움직임 + 오른쪽 STG 분포/E·σ² 추이. 에피소드 자동 반복.
```bash
DISPLAY=:1 python watch_obstacle.py                        # 학습된 BC 정책 (기본)
DISPLAY=:1 python watch_obstacle.py --controller pf        # 데모 컨트롤러(PF+노이즈 1.5e-4)
DISPLAY=:1 python watch_obstacle.py --controller tangent   # 접선점 조준 (매끈한 비교용)
```
| 옵션 | 기본 | 설명 |
|---|---|---|
| `--noise` | 1.5e-4 | pf 컨트롤러 노이즈 std (데이터 생성 채택값) |
| `--fps` | 20 | 재생 속도 |
| `--seed` | 0 | 시작 시드 (에피소드마다 +1) |
| `--max-steps` | 500 | 에피소드 상한 |
| `--dist-xmax` | 300 | 분포 패널 x축 상한 |

### 신맵 — 스킬 체이닝 경계 검출 mp4 녹화 (`record_skill_chaining.py`)
```bash
python record_skill_chaining.py                      # 학습된 정책 6 에피소드
python record_skill_chaining.py --episodes 5 --seed0 20
```
출력: `results/videos/skill_chaining.mp4`. 궤적이 검출된 스킬 경계마다 색이
바뀌고, σ² 패널에 임계값(점선)·경계(빨강 세로선)가 도달 시점에 나타남.

### 신맵 — 마우스 수동 조종 (`drive_obstacle.py`)
마우스 커서가 "당근"이 되어 에이전트를 끌고 다닌다. 시간 제한 없음.
분포 패널은 항상 **진짜 골** 기준 — 골 반대편으로 끌고 가며 분포 반응 실험 가능.
```bash
DISPLAY=:1 python drive_obstacle.py
```
| 키 | 동작 |
|---|---|
| 마우스 이동 | 에이전트가 커서를 향해 PD 제어로 끌려옴 (보라 X = 커서) |
| `p` | 학습된 정책에게 조종권 넘김/뺏음 (토글) |
| `space` | 일시정지/재개 |
| `n` | 새 에피소드 (장애물·골·시작 재배치) |
| `s` | 세션 저장 → `results/manual_sessions/` (png + pkl, 분포 전체 이력 포함) |
| `q` | 종료 |

### 구맵 — 웨이포인트 환경 + 구/신 예측기 비교 (`watch_stdmap.py`)
```bash
DISPLAY=:1 python watch_stdmap.py                                  # 구맵 예측기, 직선 주행
DISPLAY=:1 python watch_stdmap.py --controller learned             # 구맵 BC 정책 주행
DISPLAY=:1 python watch_stdmap.py --probe both --controller learned # 구+신 예측기 겹쳐 보기
DISPLAY=:1 python watch_stdmap.py --probe obstacle                  # 신맵 예측기만 교차 적용
```
- `--controller`: `straight`(PD가 골 직접 조준) / `learned`(구맵 BC 정책)
- `--probe`: `std`(구맵 예측기, 파랑) / `obstacle`(신맵 예측기 교차 적용, 빨강) /
  `both`(겹침). 신맵 예측기는 구석의 **가상 장애물**(회색 점선, 관측 전용)로
  장애물 필드를 공급받는다.
- 분포 패널의 회색 세로선 = 14 step 눈금 (웨이포인트 사다리 주기 참조)
- `--dist-xmax`(기본 200), `--fps`, `--seed`, `--max-steps`(기본 300)

### 신맵 — 에피소드 mp4 녹화 (`record_obstacle.py`)
디스플레이 없이 실행 가능. 저장본을 일시정지/되감기 하며 볼 때.
```bash
python record_obstacle.py                          # 학습된 정책 10 에피소드 mp4
python record_obstacle.py --controller pf --episodes 5 --fps 15
```
출력: `results/videos/obstacle_episodes.mp4` (`--out`으로 변경). 화면 구성은
watch_obstacle.py와 동일, 에피소드 사이 1초 정지 프레임 삽입.

### 구맵 — 구/신 예측기 겹침 mp4 녹화 (`record_stdmap.py`)
```bash
python record_stdmap.py                             # learned 주행 10 에피소드
python record_stdmap.py --controller straight --episodes 5
```
출력: `results/videos/stdmap_both_predictors.mp4`. 파랑=구맵 예측기, 빨강=신맵
예측기(가상 장애물 공급), 회색 세로선=14 step 눈금.

### 구맵 — 키보드 GUI / 리플레이 (`interactive_control.py`, phase 1)
```bash
DISPLAY=:1 python interactive_control.py --checkpoint checkpoints/sft_state.pkl   # GUI
python interactive_control.py --checkpoint checkpoints/sft_state.pkl \
    --replay scripts_replay/demo_session.json --out outputs/interactive/replay.mp4 # 헤드리스 mp4
```
GUI 키: 방향키=이동, `p`=정책 스텝, `b`=외력 토글, `x`=랜덤액션 토글,
`t`/`o`+클릭=순간이동/장애물, `g`=새 골, `n`=리셋, `s`=세션 저장.

---

## 2. 학습 / 데이터 생성

### 신맵 예측기 + BC 정책 (`src/train_obstacle_predictor.py`)
```bash
python -m src.train_obstacle_predictor                     # 전체 (생성→학습→평가)
python -m src.train_obstacle_predictor --episodes 10000 --steps 32768 --eval-episodes 100
```
- 데모가 `data/obstacle_demos.pkl`에 있으면 재사용(골 ablation 등에 활용),
  다시 만들려면 그 파일을 지우고 실행.
- 산출: `checkpoints/obstacle/predictor.pkl` (5필드 관측, bin 500, obs_fields 자기술)

### 구맵 SFT (phase 1, `train_sft.py`)
```bash
python train_sft.py        # -> checkpoints/sft_state.pkl (~15분, GPU)
```

### 구맵/멀티모달맵 예측기 스냅샷 (phase 2, `src/train_predictor.py`)
```bash
python -m src.train_predictor --map standard    # -> checkpoints/std/predictor_f*.pkl
python -m src.train_predictor                   # -> checkpoints/mm/predictor_f*.pkl
```

---

## 3. 분석 / 진단 스크립트 (그림은 `results/` 하위에 저장)

```bash
# phase 2 실험 배터리 (E0/E1/E1b/E2/sweeps/E3)
python -m src.run_experiment --exp e0     # e1, e1b, e2, sweeps, e3 동일

# phase 1 개입 시나리오 배터리 (teleport/bias/obstacle/random, 7종)
python run_scenarios.py --episodes 20 --out outputs/scenarios/

# STG 분포 프로브 (에피소드 플롯 몇 장 뽑기)
python stg_probe.py --checkpoint checkpoints/sft_state.pkl --episodes 3

# 다봉성 원인 진단: PD 진동 vs 웨이포인트 모호성 (구맵)
python -m src.diagnose_multimodal --episodes 60

# 방향전환 시 분산 증감의 부호 요인
python -m src.diagnose_switch_sign --episodes 120

# 분산 스파이크 화이트박스 분해 (between/within, 입력귀속, 지형, 선형화)
python -m src.diagnose_variance_spikes --episodes 60

# μ-σ² 전역 관계 (상관, 캘리브레이션, 진행단계별)
python -m src.analyze_mu_sigma_relationship --episodes 150

# 신맵 사다리 제거 검증 (구맵 vs 신맵 봉우리 간격 비교)
python -m src.diagnose_obstacle_multimodal --episodes 40
```

주의: 신맵 예측기(bin 500)의 **피크 기반 분석은 w=3~5 이동평균 평활 후** 수행
(bin 잔물결이 가짜 다봉으로 잡힘). σ²는 적분 기반이라 그대로 사용 가능.

---

## 4. 테스트

```bash
pytest tests/          # 전체 (core 동등성, 개입 훅, 멀티모달 env, 보상함수)
pytest tests/test_env_enhanced.py -v
```

---

## 5. 하네스/워크플로우 유틸 (상세는 CLAUDE.md)

```bash
python scripts/execute.py <phase_dir> [--model MODEL]   # phase step 자동 실행
python scripts/merge_to_main.py <feat-branch> [--push]  # main 병합 (직접 git merge 금지)
python scripts/tmux_autoresume.py                       # 리밋 자동 재개 tmux 세션
```

---

## 부록: 주요 체크포인트/데이터 경로

| 경로 | 내용 |
|---|---|
| `checkpoints/sft_state.pkl` | 구맵 SFT (50 bins, phase 1) |
| `checkpoints/std/predictor_f*.pkl` | 구맵 예측기 스냅샷 (140 bins, bin폭 1) |
| `checkpoints/mm/predictor_f*.pkl` | 멀티모달맵 예측기 (E0/E3용) |
| `checkpoints/obstacle/predictor.pkl` | 신맵 5필드 예측기 (500 bins) + BC 정책 |
| `data/obstacle_demos.pkl` | 신맵 데모 10k eps (1.19M transitions, 재사용) |
| `results/manual_sessions/` | 수동 조종 세션 저장 (png+pkl) |
