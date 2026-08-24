"""Single entry point for the variance-reward experiments (spec section 5.3).

  python -m src.run_experiment --exp e0       # bimodal collapse
  python -m src.run_experiment --exp e1       # 3 situations
  python -m src.run_experiment --exp e1b      # Delta-mu vs Delta-sigma2 independence
  python -m src.run_experiment --exp e2       # baseline vs ours (main)
  python -m src.run_experiment --exp sweeps   # decision items 2.1/2.2/2.3
  python -m src.run_experiment --exp e3       # predictor-quality ablation

Thin dispatcher over the per-experiment modules; extra args after `--` are passed
through to the underlying module.
"""

import argparse
import runpy
import sys


_MODULES = {
    'e0': 'src.experiments_observe',
    'e1': 'src.experiments_observe',
    'e1b': 'src.experiments_observe',
    'e2': 'src.run_e2',
    'sweeps': 'src.run_sweeps',
    'e3': 'src.run_e3',
}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--exp', required=True, choices=list(_MODULES))
  ap.add_argument('passthrough', nargs=argparse.REMAINDER,
                  help='args forwarded to the module (after --)')
  args = ap.parse_args()

  mod = _MODULES[args.exp]
  fwd = [a for a in args.passthrough if a != '--']
  # e0/e1/e1b share one module which runs all three; note the selection.
  if args.exp in ('e0', 'e1', 'e1b'):
    print(f'[run_experiment] {args.exp}: running {mod} (produces E0+E1+E1b)')
  sys.argv = [mod] + fwd
  runpy.run_module(mod, run_name='__main__')


if __name__ == '__main__':
  main()
