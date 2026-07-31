import numpy as np
from src.grasp_carry.env import GraspCarry2D
from src.grasp_carry.policy import ScriptedCarryPolicy
import sys
seed=int(sys.argv[1])
env=GraspCarry2D(); pol=ScriptedCarryPolicy()
env.reset(seed=seed); pol.reset()
print('src_w=%.1f h=%.1f w=%.1f bx=%.1f'%(env.src_box.inner_width, env.block_h, env.block_w, env.block_body.position.x))
last=None
for t in range(env.cfg.max_steps):
    a=pol(env)
    key=(pol.phase,pol._sub)
    _,_,term,trunc,info=env.step(a)
    if key!=last or t<3:
        print('%3d %-9s %-8s ex=%6.1f ey=%6.1f gap=%5.1f bx=%6.1f by=%6.1f ang=%5.1f held=%d L=%4.1f act=(%.0f,%.0f,%.0f) gx=%.1f'%(
            t,pol.phase,pol._sub,pol.ex,pol.ey,env.gripper.gap,pol.bx,pol.by,np.degrees(env.block_body.angle),
            info['is_held'],info['contact_length'],a[0],a[1],a[3], pol._grasp_x_cache if pol._grasp_x_cache is not None else -1))
        last=key
    if term or trunc:
        print('END',t,info['outcome']); break
