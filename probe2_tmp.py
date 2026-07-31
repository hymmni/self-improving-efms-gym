import numpy as np
from src.grasp_carry.env import GraspCarry2D
from src.grasp_carry.policy import ScriptedCarryPolicy

env = GraspCarry2D()
pol = ScriptedCarryPolicy()
for seed in range(6):
    env.reset(seed=seed); pol.reset()
    trace=[]
    for t in range(env.cfg.max_steps):
        a = pol(env)
        obs,r,term,trunc,info = env.step(a)
        trace.append((t, pol.phase, pol._sub, round(float(env.gripper.pose[0]),1), round(float(env.gripper.pose[1]),1), round(info['contact_length'],1), info['is_held']))
        if term or trunc: break
    print('seed',seed,'src_w',round(env.src_box.inner_width,1),'h',round(env.block_h,1),'bx0',round(pol.grasp_arms[0],1) if pol.grasp_arms else None,
          'outcome',info['outcome'],'steps',info['steps'],'regrasp',pol.n_regrasps,'drops',pol.n_drops,
          'contacts',[round(c,1) for c in pol.grasp_contacts],'speeds',[round(s,2) for s in pol.grasp_speeds])
    # print phase transitions
    last=None
    for row in trace:
        key=(row[1],row[2])
        if key!=last:
            print('   ',row); last=key
