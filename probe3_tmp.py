import numpy as np
from src.grasp_carry.env import GraspCarry2D
from src.grasp_carry.policy import ScriptedCarryPolicy

env = GraspCarry2D(); pol = ScriptedCarryPolicy()
env.reset(seed=0); pol.reset()
for t in range(60):
    a = pol(env)
    obs,r,term,trunc,info = env.step(a)
    if pol.phase=='transport':
        print(t, pol._sub, 'ey=%.1f'%env.gripper.pose[1], 'ty_hold=%.1f'%pol._travel_y_hold(env),
              'bbot=%.1f'%pol.block_bottom, 'act=(%.1f,%.1f)'%(a[0],a[1]), 'held', info['is_held'], 's=%.2f'%pol._speed_hold)
    if term or trunc: break
