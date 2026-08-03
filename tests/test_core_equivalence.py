"""Equivalence tests: pointmass_core vs. the baseline notebook.

Loads cell 4 (environment) and cell 6 (PD controller) of the repo-root
`pointmass_notebook.ipynb` via exec() into an isolated namespace and checks
that `pointmass_core.Point2D` / `pointmass_core.pd_controller` produce
bit-identical results under the same seeds and action sequences.
"""

import json
import pathlib
from typing import Optional, Any, NamedTuple, Callable, Mapping, Sequence, Dict, Tuple
import dataclasses

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as patches
import dm_env
from dm_env import specs
import pytest

import pointmass_core

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / 'pointmass_notebook.ipynb'

ENV_CELL_INDEX = 4
PD_CELL_INDEX = 6


def _load_notebook_namespace():
    """exec() the notebook's env and PD-controller cells in a fresh namespace."""
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)

    ns = {
        'Optional': Optional, 'Any': Any, 'NamedTuple': NamedTuple,
        'Callable': Callable, 'Mapping': Mapping, 'Sequence': Sequence,
        'Dict': Dict, 'Tuple': Tuple, 'dataclasses': dataclasses,
        'np': np, 'plt': plt, 'FigureCanvas': FigureCanvas,
        'patches': patches, 'dm_env': dm_env, 'specs': specs,
    }
    for idx in (ENV_CELL_INDEX, PD_CELL_INDEX):
        cell = nb['cells'][idx]
        assert cell['cell_type'] == 'code'
        exec(''.join(cell['source']), ns)

    assert 'Point2D' in ns and 'pd_controller' in ns
    return ns


@pytest.fixture(scope='module')
def notebook_ns():
    return _load_notebook_namespace()


def _rollout(env_cls, seed, actions):
    """Reset an env under a fixed global-numpy seed and step a fixed action
    sequence, returning the full observation history.

    Point2D draws its goal/start from the global numpy RNG in reset(); step()
    is deterministic, so re-seeding before construction makes runs comparable.
    """
    np.random.seed(seed)
    env = env_cls()
    ts = env.reset()
    observations = [ts.observation]
    rewards = []
    for act in actions:
        ts = env.step(act.copy())
        observations.append(ts.observation)
        rewards.append(ts.reward)
    return observations, rewards


def _make_actions(seed, num_steps=50):
    rng = np.random.RandomState(seed)
    return rng.uniform(-1e-3, 1e-3, size=(num_steps, 2)).astype(np.float32)


@pytest.mark.parametrize('seed', [0, 1, 42])
def test_env_equivalence(notebook_ns, seed):
    actions = _make_actions(seed + 1000)

    core_obs, core_rews = _rollout(pointmass_core.Point2D, seed, actions)
    nb_obs, nb_rews = _rollout(notebook_ns['Point2D'], seed, actions)

    assert len(core_obs) == len(nb_obs)
    for core_o, nb_o in zip(core_obs, nb_obs):
        assert sorted(core_o.keys()) == sorted(nb_o.keys())
        for k in core_o:
            np.testing.assert_array_equal(core_o[k], nb_o[k])
    np.testing.assert_array_equal(np.array(core_rews), np.array(nb_rews))


@pytest.mark.parametrize('seed', [0, 1, 42])
def test_env_equivalence_pd_driven(notebook_ns, seed):
    """Roll both envs with the shared (module-level) PD controller in the
    loop, so the action at each step depends on the trajectory so far."""

    def pd_rollout(env_cls, max_steps=200):
        np.random.seed(seed)
        env = env_cls()
        ts = env.reset()
        observations = [ts.observation]
        steps = 0
        while (not env.success()) and steps < max_steps:
            obs = ts.observation
            act = pointmass_core.pd_controller(
                obs['cur_pos'], obs['cur_vel'], obs['goal_pos'])
            ts = env.step(act)
            observations.append(ts.observation)
            steps += 1
        return observations

    core_obs = pd_rollout(pointmass_core.Point2D)
    nb_obs = pd_rollout(notebook_ns['Point2D'])

    assert len(core_obs) == len(nb_obs)
    for core_o, nb_o in zip(core_obs, nb_obs):
        for k in core_o:
            np.testing.assert_array_equal(core_o[k], nb_o[k])


@pytest.mark.parametrize('seed', [0, 1, 42])
def test_pd_controller_equivalence(notebook_ns, seed):
    rng = np.random.RandomState(seed)
    for _ in range(100):
        cur_pos = rng.uniform(-1., 1., size=2).astype(np.float32)
        cur_vel = rng.uniform(-0.1, 0.1, size=2).astype(np.float32)
        goal_pos = rng.uniform(-1., 1., size=2).astype(np.float32)

        core_act = pointmass_core.pd_controller(cur_pos, cur_vel, goal_pos)
        nb_act = notebook_ns['pd_controller'](cur_pos, cur_vel, goal_pos)
        np.testing.assert_array_equal(core_act, nb_act)
