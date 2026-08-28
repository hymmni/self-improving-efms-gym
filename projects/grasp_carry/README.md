# grasp_carry

GraspCarry2D — 은닉 물성(마찰·질량 등) 기반 2D 파지·운반 환경과 그 위의 JAX/Haiku 학습 스택.

## 구조

- `src/grasp_carry/` — 환경(`env.py`, `gripper.py`, `config.py`), 스크립트 정책(`policy.py`),
  학습 라이브러리 코드(`ddpo.py`, `diffusion_act.py`, `conditional_unet1d.py`, `reward.py`,
  `carry_stg_reward.py`, `train_carry_dstg.py`, `train_carry_predictor.py`, `train_carry_qstg.py`,
  `eval_carry.py`, `networks.py`)
- `src/grasp_carry/scripts/` — 실행 진입점. `train/`(정책 학습), `collect/`(데모/롤아웃 수집),
  `record/`(mp4 영상 녹화), `analyze/`(체크포인트 평가·비교·진단)로 기능별 분류
- `tests/` — pytest 스위트
- `data/`, `checkpoints/`, `results/`, `outputs/` — 대용량/산출물 (git 미추적)

## 실행

의존성은 레포 루트의 `requirements.txt`(JAX 스택 공유). Docker `jax` 서비스가
`PYTHONPATH=/workspace/projects/grasp_carry/src`를 설정해두므로 별도 설치 없이 바로 import된다.

```bash
python -m grasp_carry.scripts.train.train_carry_actor --help
python -m grasp_carry.scripts.record.record_carry --help
```

로컬(비-Docker)에서 실행할 때는 `PYTHONPATH=projects/grasp_carry/src`를 직접 설정한다.
