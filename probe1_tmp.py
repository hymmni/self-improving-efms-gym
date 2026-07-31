import numpy as np
from src.grasp_carry.config import CarryConfig
from src.grasp_carry.env import GraspCarry2D

env = GraspCarry2D()
cfg = env.cfg
print('gow', cfg.gripper_outer_width, 'src range', cfg.src_box_width_range,
      'tgt', cfg.tgt_box_width, 'finger_len', cfg.finger_length)
rows=[]
for seed in range(30):
    env.reset(seed=seed)
    s, t = env.src_box, env.tgt_box
    bx = float(env.block_body.position.x)
    # can gripper center on block and fit?
    lo_ok = s.left_inner + 2 + cfg.gripper_outer_width/2
    hi_ok = s.right_inner - 2 - cfg.gripper_outer_width/2
    fits = s.inner_width >= cfg.gripper_outer_width + 4
    rows.append((seed, round(s.inner_width,1), round(bx,1), fits,
                 round(lo_ok,1), round(hi_ok,1),
                 round(env.max_descend_y(bx),1),
                 round(env.max_descend_y(float(np.clip(bx, lo_ok, hi_ok))),1),
                 round(env.block_h,1),
                 round(s.right_outer,1), round(t.left_outer,1)))
for r in rows: print(r)
print('deep-capable frac', np.mean([r[3] for r in rows]))
