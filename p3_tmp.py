import numpy as np, sys
from src.grasp_carry.env import GraspCarry2D
from src.grasp_carry.policy import ScriptedCarryPolicy
seed,t0,t1=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
env=GraspCarry2D(); pol=ScriptedCarryPolicy()
env.reset(seed=seed); pol.reset()
print('src.ro=%.1f tgt.lo=%.1f floor=%.1f'%(env.src_box.right_outer,env.tgt_box.left_outer,env.cfg.floor_y))
for t in range(env.cfg.max_steps):
    a=pol(env)
    row='%3d %-9s %-8s ex=%6.1f ey=%6.1f bx=%6.1f bbot=%6.1f ang=%6.1f dest=(%.1f,%.1f) stall=%d wait=%d s=%.2f lim=%.1f act=(%.0f,%.0f,%.0f)'%(
        t,pol.phase,pol._sub,pol.ex,pol.ey,pol.bx,pol.block_bottom,np.degrees(env.block_body.angle),
        pol._dest_x,pol._dest_floor,pol._stall,pol._wait,pol._speed_hold,env.max_descend_y(pol.ex),a[0],a[1],a[3])
    _,_,term,trunc,info=env.step(a)
    if t0<=t<=t1: print(row,'held=%d'%info['is_held'])
    if term or trunc: print('END',t,info['outcome']); break
