# pointmass

SI-EFM 논문의 pointmass 데모 재현 트랙. 연구가 GraspCarry2D/square_assembly로 옮겨가며 한동안
`archive/`에 있었으나, 다시 들여다볼 수 있는 연구 트랙이라 최소 골격만 남기고 독립 프로젝트
폴더로 분리했다.

## 구조

- `pointmass_core.py`, `pointmass_notebook.ipynb` — 원본 노트북에서 그대로 추출한 클린 버전
  (ADR-004, copy-as-is)
- `archive/` — 과거 진단·실험 스크립트 이력. 전부 `pointmass_core`(또는 그 파생인
  `envs_enhanced`, `obstacle_env`, `multimodal_env` 등)에 의존하는 pointmass 계열이며,
  현재 실행되는 코드에서 import하지 않는 죽은 코드다. 부활 작업 시 참고용.

## 실행

의존성은 레포 루트의 `requirements.txt`(JAX 스택 공유).

```bash
python -m jupyter notebook pointmass_notebook.ipynb
```
