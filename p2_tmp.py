import numpy as np, sys
from src.grasp_carry.env import GraspCarry2D
from src.grasp_carry.policy import ScriptedCarryPolicy
seed, n = int(sys.argv[1]), int(sys.argv[2])
env=GraspCarry2D(); pol=ScriptedCarryPolicy()
env.reset(seed=seed); pol.reset()
for t in range(n):
    a=pol(env); env.step(a)
b=env.gripper.base; g=env.gripper
print('phase',pol.phase,pol._sub,'wait',pol._wait)
print('base',b.position,'vel',b.velocity,'gap %.1f'%g.gap)
print('span [%.1f, %.1f]  src_inner [%.1f, %.1f] rim %.1f'%(
    b.position.x-68, b.position.x+68, env.src_box.left_inner, env.src_box.right_inner, env.src_box.rim_y))
print('fingers',[ (round(f.position.x,2), round(f.position.y,2)) for f in g.fingers])
print('block',env.block_body.position, np.degrees(env.block_body.angle))
names={}
for s in env.space.shapes:
    names[s]=('static' if s.body.body_type==2 else ('block' if s is env.block_shape else ('finger' if s in g.finger_shapes else 'base')))
def visit(arb):
    a,b2=arb.shapes
    print('   ',names.get(a),names.get(b2),'n',arb.contact_point_set.normal, 'tot_impulse', arb.total_impulse)
print('base arbs:'); b.each_arbiter(visit)
for i,f in enumerate(g.fingers):
    print('finger',i,'arbs:'); f.each_arbiter(visit)
