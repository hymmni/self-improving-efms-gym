import numpy as np, sys
from src.grasp_carry.env import GraspCarry2D
from src.grasp_carry.policy import ScriptedCarryPolicy
seed=int(sys.argv[1]); t0=int(sys.argv[2]); t1=int(sys.argv[3])
env=GraspCarry2D(); pol=ScriptedCarryPolicy()
env.reset(seed=seed); pol.reset()
for t in range(env.cfg.max_steps):
    a=pol(env)
    row='%3d %-9s %-8s ex=%6.1f ey=%6.1f gap=%5.1f bx=%6.1f by=%6.1f ang=%6.1f tyf=%6.1f wait=%d retry=%d act=(%.0f,%.0f,%.0f)'%(
        t,pol.phase,pol._sub,pol.ex,pol.ey,env.gripper.gap,pol.bx,pol.by,np.degrees(env.block_body.angle),
        pol._travel_y_free(env),pol._wait,pol._retry,a[0],a[1],a[3])
    _,_,term,trunc,info=env.step(a)
    if t0<=t<=t1: print(row,'held=%d L=%.1f'%(info['is_held'],info['contact_length']))
    if term or trunc: print('END',t,info['outcome']); break
