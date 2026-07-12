"""Core components of the SI-EFM pointmass demo, extracted as a module.

Every block below is copied as-is (verbatim) from the original notebook,
per ADR-004 (copy-as-is). The only permitted changes (ADR-004 relaxation,
marked with `# MODIFIED:` comments) are:
  - removal of IPython/colab-only code (display calls, `#@param` cell magic,
    sanity-check/execution code at the bottom of cells),
  - conversion of notebook-global dependencies into function arguments,
  - trimming imports to what the extracted code needs.
Numerical logic (physics stepping, losses, distribution parameterizations,
bin conversion formulas) is untouched.

The notebook itself (`pointmass_notebook.ipynb` at the repo root) remains the
reproduction baseline and is not modified.
"""

# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 2)
# MODIFIED: trimmed to the imports needed by the extracted code; removed
# notebook-only imports (mediapy, moviepy, IPython.display, PIL, pickle, ...).
from typing import Optional, Any, NamedTuple, Callable, Mapping, Sequence, Dict, Tuple
import dataclasses

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as patches

import jax
import jax.numpy as jnp
import optax
import haiku as hk

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import dm_env
from dm_env import specs


################################################################################
# Environment
################################################################################

# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 4)
BOUNDS_X = np.array([-1., 1.], dtype=np.float32)
BOUNDS_Y = np.array([-1., 1.], dtype=np.float32)

DPI = 200
RENDER_HEIGHT_INCHES = 5

class Point2D(dm_env.Environment):
  def __init__(self):
    self._cur_pos = np.zeros(2, dtype=np.float32)
    self._goal_pos = np.zeros(2, dtype=np.float32)
    self._cur_vel = np.zeros(2, dtype=np.float32)
    self._cur_episode_traj = []
    self._physics_substeps = 10
    self._success_radius = 0.15

  def sample_goal(self):
    border_x = (BOUNDS_X[1] - BOUNDS_X[0]) * 0.05
    border_y = (BOUNDS_Y[1] - BOUNDS_Y[0]) * 0.05
    goal_x = np.random.uniform(
        BOUNDS_X[0] + border_x, BOUNDS_X[1] - border_x)
    goal_y = np.random.uniform(
        BOUNDS_Y[0] + border_y, BOUNDS_Y[1] - border_y)
    return np.array([goal_x, goal_y], dtype=np.float32)

  def set_goal(self, goal_pos):
    self._goal_pos = goal_pos

  def reset(self):
    self._goal_pos = self.sample_goal()
    cur_x = np.random.uniform(BOUNDS_X[0], BOUNDS_X[1])
    cur_y = np.random.uniform(BOUNDS_Y[0], BOUNDS_Y[1])
    self._cur_pos = np.array([cur_x, cur_y], dtype=np.float32)
    cur_pos_copy = self._cur_pos.copy()

    self._cur_vel = np.zeros(2, dtype=np.float32)
    cur_vel_copy = self._cur_vel.copy()

    obs = {
        'cur_pos': cur_pos_copy,
        'cur_vel': cur_vel_copy,
        'goal_pos': self._goal_pos.copy()}
    ts = dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=None,
        discount=None,
        observation=obs,)

    self._cur_episode_traj = [cur_pos_copy]
    return ts

  def step(self, action):
    for i in range(self._physics_substeps):
      self._cur_vel += action
      self._cur_pos += self._cur_vel

    cur_pos_copy = self._cur_pos.copy()
    cur_vel_copy = self._cur_vel.copy()
    obs = {
        'cur_pos': cur_pos_copy,
        'cur_vel': cur_vel_copy,
        'goal_pos': self._goal_pos.copy()}

    if self.success():
      step_type = dm_env.StepType.LAST
    else:
      step_type = dm_env.StepType.MID
    ts = dm_env.TimeStep(
        step_type=step_type,
        reward=-1. * np.linalg.norm(self._cur_pos - self._goal_pos),
        discount=1.,
        observation=obs,)

    self._cur_episode_traj.append(cur_pos_copy)
    return ts

  def success(self, waypoint: Optional[np.ndarray] = None):
    if waypoint is not None:
      goal_pos = waypoint
    else:
      goal_pos = self._goal_pos
    return np.linalg.norm(self._cur_pos - goal_pos) < self._success_radius

  def observation_spec(self):
    return {
        'cur_pos': specs.Array((2,), dtype=np.float32),
        'cur_vel': specs.Array((2,), dtype=np.float32),
        'goal_pos': specs.Array((2,), dtype=np.float32),}

  def action_spec(self):
    return specs.Array((2,), dtype=np.float32)

  def render(
      self,
      title: str = '',
      points: Optional[np.ndarray] = None,
      goal_pos: Optional[np.ndarray] = None):
    fig, ax = plt.subplots(
        figsize=(RENDER_HEIGHT_INCHES, RENDER_HEIGHT_INCHES), dpi=DPI)
    ax.set_xlim(BOUNDS_X[0], BOUNDS_X[1])
    ax.set_ylim(BOUNDS_Y[0], BOUNDS_Y[1])
    ax.set_aspect('equal')

    if points is None:
      points = np.array(self._cur_episode_traj)
      cur_pos = self._cur_pos
    else:
      cur_pos = points[-1]

    if goal_pos is None:
      goal_pos = self._goal_pos

    ax.plot(points[:, 0], points[:, 1], marker='.', color='blue', markersize=16, linewidth=4)
    ax.scatter(
        goal_pos[0], goal_pos[1], marker='*', s=200, color='orange', linewidths=8)
    ax.scatter(
        cur_pos[0], cur_pos[1], marker='o', s=100, color='red', linewidths=8)

    # Add a dashed circle around the star
    circle = patches.Circle(
        (goal_pos[0], goal_pos[1]),  # Center of the circle
        self._success_radius,  # Radius of the circle
        edgecolor='green',  # Color of the circle
        linestyle='--',  # Dashed line
        linewidth=4,  # Thickness of the circle line
        fill=False  # Ensure it's just an outline
    )
    ax.add_patch(circle)  # Add the circle to the plot

    # Make the axes lines thicker
    for spine in ax.spines.values():
        spine.set_linewidth(4)  # Adjust the thickness here

    if title != '':
      ax.set_title(title, fontsize=18, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    # Render the plot using FigureCanvasAgg
    canvas = FigureCanvas(fig)
    canvas.draw()

    # Convert the rendered image to a numpy array
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(int(height), int(width), 3)

    plt.close(fig)
    return image


################################################################################
# PD controller
################################################################################

# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 6)
def pd_controller(cur_pos, cur_vel, goal_pos):
  Kp = 0.0002
  Kd = 0.0125
  act = Kp * (goal_pos - cur_pos) + Kd * (-1. * cur_vel)
  return act


################################################################################
# Dataset generation
################################################################################

# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 10)
# MODIFIED: the notebook runs this at cell scope with `#@param` globals; here it
# is wrapped in `generate_dataset(...)` taking those globals as arguments, and
# the stats-printing code at the end of the cell is removed. The generation
# loop body is verbatim.
NestedArray = Any

# This dataset has the format (obs, act, time_to_success, next_obs)
# It's inefficent to store next obs but makes life easier
# Will save two versions: 1) trajectories kept separate,
# 2) just tuples

class DataTuple(NamedTuple):
  observation: NestedArray
  action: NestedArray
  time_to_success: NestedArray
  reward: NestedArray
  discount: NestedArray
  next_observation: NestedArray


def generate_dataset(
    env,
    pd_controller,
    num_episodes=10000,
    num_waypoints_per_episode=5,
    episode_len_discard_thresh=10):
  episodes = []
  while len(episodes) < num_episodes:
    ep_num = len(episodes)
    traj = []

    ts = env.reset()
    cur_obs = ts.observation
    succ = env.success()

    waypoint_idx = 0
    if num_waypoints_per_episode == 0:
      cur_waypoint = cur_obs['goal_pos']
    else:
      cur_waypoint = env.sample_goal()
    waypoint_succ = env.success(waypoint=cur_waypoint)

    while not succ:
      if waypoint_succ:
        waypoint_idx += 1
        waypoint_idx = min(waypoint_idx, num_waypoints_per_episode)
        if waypoint_idx == num_waypoints_per_episode:
          cur_waypoint = cur_obs['goal_pos']
        else:
          cur_waypoint = env.sample_goal()

      act = pd_controller(
          cur_obs['cur_pos'], cur_obs['cur_vel'], cur_waypoint)
      ts = env.step(act)
      next_obs = ts.observation

      data_tuple = DataTuple(
          observation=cur_obs,
          action=act,
          time_to_success=0.,
          reward=ts.reward if ts.reward is not None else 0.,
          discount=1.,
          next_observation=next_obs,)
      traj.append(data_tuple)

      cur_obs = next_obs
      succ = env.success()
      waypoint_succ = env.success(waypoint=cur_waypoint)

    # add the last timestep
    act = pd_controller(
        cur_obs['cur_pos'], cur_obs['cur_vel'], cur_waypoint)
    data_tuple = DataTuple(
        observation=cur_obs,
        action=act,
        time_to_success=0.,
        reward=ts.reward if ts.reward is not None else 0.,
        discount=0.,
        next_observation=cur_obs,)
    traj.append(data_tuple)

    # discard if episode is too short
    traj_len = len(traj)
    if traj_len < episode_len_discard_thresh:
      continue

    # stack the traj arrays
    new_traj = jax.tree.map(
        lambda *xs: np.stack(xs, dtype=np.float32), *traj)

    # label traj arrays with time to success
    new_traj = new_traj._replace(
        time_to_success=np.arange(
            traj_len - 1, -1, -1, dtype=np.float32))
    episodes.append(new_traj)

  # concat all the trajs
  all_tuples = jax.tree.map(
      lambda *xs: np.concatenate(xs, dtype=np.float32), *episodes)

  return episodes, all_tuples


################################################################################
# Networks
################################################################################

# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 15)
Params = Any
PRNGKey = Any
NetworkOutput = Any
Entropy = Any
ActDistParams = Params
FeedForwardPolicyWithExtra = Any
LogProbFn = Any
SampleFn = Any
Observation = Any
Action = Any
DistanceToSuccessDistParams = Params
EntropyFn = Callable[
    [Params, PRNGKey], Entropy]


@dataclasses.dataclass
class FeedForwardNetwork:
  """Holds a pair of pure functions defining a feed-forward network.

  Attributes:
    init: A Jax pure function: ``params = init(rng, *a, **k)``
    apply: A Jax pure function: ``out = apply(params, rng, *a, **k)``
  """
  # Initializes and returns the networks parameters.
  init: Callable[..., Params]
  # Computes and returns the outputs of a forward pass.
  apply: Callable[..., NetworkOutput]


# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 16)
MIN_ACT_SCALE = 1e-2  # MODIFIED: removed `#@param` colab annotation


class MVNDiagParams(NamedTuple):
  """Parameters for a diagonal multi-variate normal distribution."""
  loc: jnp.ndarray
  scale_diag: jnp.ndarray


class CategoricalParams(NamedTuple):
  """Parameters for a categorical distribution."""
  logits: jnp.ndarray


class TIMERNetworkOutput(NamedTuple):
  act_dist_params: ActDistParams
  dist_to_succ_dist_params: DistanceToSuccessDistParams


@dataclasses.dataclass
class TIMERNetworks:
  """Network and pure functions for the TIMER agent.

  network: outputs TIMERNetworkOutputs
  act_log_prob: log probability of an action
  act_entropy: optional method for entropy of an action distribution
  sample_act: samples an action given [ActDistParams, PRNGKey]
  sample_act_mode: optional separate action sampling procedure
  dist_log_prob: log probability of a distance
  dist_entropy: optional method for entropy of a distance distribution
  sample_dist: samples a distance given [DistanceToSuccessDistParams, PRNGKey]
  sample_dist_mode: optional separate distance sampling procedure
  """
  network: FeedForwardNetwork
  act_log_prob: LogProbFn
  sample_act: SampleFn
  dist_log_prob: LogProbFn
  sample_dist: SampleFn
  act_entropy: Optional[EntropyFn] = None
  sample_act_mode: Optional[SampleFn] = None
  dist_entropy: Optional[EntropyFn] = None
  sample_dist_mode: Optional[SampleFn] = None


def make_policy_fn(
    timer_networks: TIMERNetworks,
    evaluation: bool) -> FeedForwardPolicyWithExtra:
  """Returns a policy function for the TIMER agent."""

  def _policy_fn(
      params: Params,
      key: PRNGKey,
      observations: Observation,
  ):
    timer_network_output: TIMERNetworkOutput = timer_networks.network.apply(
        params, observations)
    if evaluation:
      actions = timer_networks.sample_act_eval(
          timer_network_output.act_dist_params, key)
    else:
      actions = timer_networks.sample_act(
          timer_network_output.act_dist_params, key)
    return actions, {}

  return _policy_fn


def build_continuous_act_discrete_dist_v0(
    layer_sizes: Sequence[int],
    act_dim: int,
    num_dist_bins: int,
    dummy_input,
) -> TIMERNetworks:
  """"Builds TIMERNetworks for continuous action and discrete distance."""

  def _network(
      x: Observation) -> TIMERNetworkOutput:
    #### Build the action part
    h_act = hk.nets.MLP(
        output_sizes=layer_sizes,
        activation=jax.nn.relu,
        activate_final=True,)(x)
    act_loc = hk.Linear(
        act_dim,
        w_init=hk.initializers.VarianceScaling(1e-4),
        b_init=hk.initializers.Constant(0.))(h_act)
    act_scale = hk.Linear(
        act_dim,
        w_init=hk.initializers.VarianceScaling(1e-4),
        b_init=hk.initializers.Constant(0.))(h_act)
    act_scale = jax.nn.softplus(act_scale) + MIN_ACT_SCALE
    act_dist = MVNDiagParams(loc=act_loc, scale_diag=act_scale)

    #### Build the distance part
    h_dist = hk.nets.MLP(
        output_sizes=layer_sizes,
        activation=jax.nn.relu,
        activate_final=True,)(x)
    dist_logits = hk.Linear(num_dist_bins, with_bias=False)(h_dist)
    distance_dist = CategoricalParams(logits=dist_logits)

    return TIMERNetworkOutput(
        act_dist_params=act_dist,
        dist_to_succ_dist_params=distance_dist,)

  transformed_network = hk.without_apply_rng(hk.transform(_network))
  def init_closure(rng: PRNGKey):
    return transformed_network.init(rng, dummy_input)
  network = FeedForwardNetwork(
      init=init_closure,
      apply=transformed_network.apply,)

  def act_log_prob(params: MVNDiagParams, action):
    return tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=params.scale_diag).log_prob(action)

  def act_entropy(
      params: MVNDiagParams, key: PRNGKey
  ) -> Entropy:
    del key
    return tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=params.scale_diag).entropy()

  def sample_act(params: MVNDiagParams, key: PRNGKey):
    return tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=params.scale_diag).sample(seed=key)

  def sample_act_mode(params: MVNDiagParams, key: PRNGKey):
    del key
    return tfd.MultivariateNormalDiag(
        loc=params.loc, scale_diag=params.scale_diag).mode()

  def dist_log_prob(params: CategoricalParams, action):
    return tfd.Categorical(logits=params.logits).log_prob(action)

  def dist_entropy(
      params: CategoricalParams, key: PRNGKey
  ) -> Entropy:
    del key
    return tfd.Categorical(logits=params.logits).entropy()

  def sample_dist(params: CategoricalParams, key: PRNGKey):
    return tfd.Categorical(logits=params.logits).sample(seed=key)

  def sample_dist_mode(params: CategoricalParams, key: PRNGKey):
    del key
    return tfd.Categorical(logits=params.logits).mode()

  return TIMERNetworks(
      network=network,
      act_log_prob=act_log_prob,
      sample_act=sample_act,
      dist_log_prob=dist_log_prob,
      sample_dist=sample_dist,
      act_entropy=act_entropy,
      sample_act_mode=sample_act_mode,
      dist_entropy=dist_entropy,
      sample_dist_mode=sample_dist_mode,
  )


################################################################################
# Timestep prediction converters
################################################################################

# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 19)
class DistanceConverters(NamedTuple):
  distance_to_network_format: Callable[
      [jnp.ndarray], NetworkOutput]
  network_format_to_distance: Callable[
      [NetworkOutput], jnp.ndarray]


def build_discrete_distance_converter(
    min_distance: float,
    max_distance: float,
    num_bins: int = 100) -> DistanceConverters:

  bin_size = (max_distance - min_distance) / num_bins

  def _distance_to_network_format(d: float):
    d = jnp.clip(d, min_distance, max_distance - bin_size / 2.)
    bin_index = jnp.floor_divide(d - min_distance, bin_size)
    return bin_index

  distance_to_network_format = jax.vmap(_distance_to_network_format)

  dist_vals = jnp.linspace(
      min_distance,
      max_distance,
      num_bins + 1,
      endpoint=True, dtype=jnp.float32)
  dist_vals = dist_vals[:-1]

  def _network_format_to_distance(logits: NetworkOutput):
    dist = jnp.sum(dist_vals * jax.nn.softmax(logits))
    return dist

  network_format_to_distance = jax.vmap(_network_format_to_distance)

  return DistanceConverters(
      distance_to_network_format,
      network_format_to_distance,)


################################################################################
# Stage 1 (SFT) learner utilities
################################################################################

# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 22)
def get_from_first_device(x, as_numpy=False):
  if as_numpy:
    return jax.device_get(jax.tree.map(lambda x: x[0], x))
  else:
    return jax.tree.map(lambda x: x[0], x)


# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 23)
# MODIFIED: removed the sanity-check execution code at the bottom of the cell;
# inside `pretrain_loss`, the notebook-global `dc` is replaced by the
# `distance_converters` constructor argument (in the notebook both names bind
# the same DistanceConverters object).
TIMERParams = Params


class TrainingState(NamedTuple):
  """Training state for the TIMER learner."""
  params: TIMERParams
  opt_state: optax.OptState
  random_key: PRNGKey


class PretrainLearner():
  def __init__(
      self,
      timer_networks: TIMERNetworks,
      distance_converters: DistanceConverters,
      optimizer: optax.GradientTransformation,
      random_key: PRNGKey,
      global_minibatch_size: int,
      num_minibatches: int,):

    self.local_learner_devices = jax.local_devices()
    self.num_local_learner_devices = jax.local_device_count()
    self.learner_devices = jax.devices()
    per_device_minibatch_size = (
        global_minibatch_size // len(self.learner_devices))

    self._num_full_update_steps = 0
    self.global_minibatch_size = global_minibatch_size
    self.num_minibatches = num_minibatches

    def pretrain_loss(params, minibatch: DataTuple):
      obs = minibatch.observation
      obs = jnp.concatenate(
          [obs['cur_pos'], obs['cur_vel'], obs['goal_pos']], axis=-1)
      acts = minibatch.action
      dist_idx = distance_converters.distance_to_network_format(
          minibatch.time_to_success)  # MODIFIED: was notebook-global `dc`

      preds = timer_networks.network.apply(params, obs)
      act_dist_params = preds.act_dist_params
      dist_to_succ_dist_params = preds.dist_to_succ_dist_params

      # bc loss
      act_log_prob = jnp.mean(
          timer_networks.act_log_prob(act_dist_params, acts))
      bc_loss = -1.0 * act_log_prob

      # Distance to success loss
      dist_log_prob = jnp.mean(timer_networks.dist_log_prob(
          dist_to_succ_dist_params, minibatch.time_to_success))
      dist_loss = -1.0 * dist_log_prob

      total_loss = bc_loss + dist_loss

      return total_loss, {
          'pretrain_loss': total_loss,
          'act_log_prob': act_log_prob,
          'dist_log_prob': dist_log_prob,}

    pretrain_grad = jax.grad(pretrain_loss, has_aux=True)

    def per_device_pretrain_step(
        state: TrainingState,
        minibatch: DataTuple,
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
      pretrain_loss_grad, metrics = pretrain_grad(state.params, minibatch)
      pretrain_loss_grad = jax.lax.pmean(
          pretrain_loss_grad, axis_name='devices'
      )
      updates, new_opt_state = optimizer.update(
          pretrain_loss_grad, state.opt_state
      )
      new_params = optax.apply_updates(state.params, updates)
      state = state._replace(params=new_params, opt_state=new_opt_state)
      return state, metrics

    def scanned_per_device_pretrain_step(
        state: TrainingState, batch: DataTuple):
      def reshape_for_scan(x):
        new_shape = [
            num_minibatches,
            per_device_minibatch_size,
        ] + list(x.shape[1:])
        return jnp.reshape(x, new_shape)

      minibatches = jax.tree.map(reshape_for_scan, batch)
      state, metrics = jax.lax.scan(
          per_device_pretrain_step, state, minibatches, length=num_minibatches)
      metrics = jax.tree.map(jnp.mean, metrics)

      return state, metrics

    self._pmapped_scanned_pretrain_step = jax.pmap(
        scanned_per_device_pretrain_step,
        axis_name='devices',
        devices=self.learner_devices)

    def per_device_compute_loss(
        state: TrainingState,
        minibatch: DataTuple,
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
      _, metrics = pretrain_loss(state.params, minibatch)
      return state, metrics

    def scanned_per_device_compute_loss(
        state: TrainingState, batch: DataTuple):
      def reshape_for_scan(x):
        new_shape = [
            num_minibatches,
            per_device_minibatch_size,
        ] + list(x.shape[1:])
        return jnp.reshape(x, new_shape)

      minibatches = jax.tree.map(reshape_for_scan, batch)
      state, metrics = jax.lax.scan(
          per_device_compute_loss, state, minibatches, length=num_minibatches)
      metrics = jax.tree.map(jnp.mean, metrics)

      return state, metrics

    self._pmapped_scanned_compute_loss = jax.pmap(
        scanned_per_device_compute_loss,
        axis_name='devices',
        devices=self.learner_devices)

    def make_initial_state(random_key: PRNGKey) -> TrainingState:
      all_keys = jax.random.split(
          random_key, num=self.num_local_learner_devices + 1)
      key_init, key_state = all_keys[0], all_keys[1:]
      key_state = [key_state[i] for i in range(self.num_local_learner_devices)]
      key_state = jax.device_put_sharded(key_state, self.local_learner_devices)

      initial_params = timer_networks.network.init(key_init)
      initial_opt_state = optimizer.init(initial_params)

      initial_params = jax.device_put_replicated(initial_params,
                                                 self.local_learner_devices)
      initial_opt_state = jax.device_put_replicated(initial_opt_state,
                                                    self.local_learner_devices)

      return TrainingState(
          params=initial_params,
          opt_state=initial_opt_state,
          random_key=key_state,)

    # Initialise training state (parameters and optimizer state).
    self._state = make_initial_state(random_key)

  def step(self, batch: DataTuple):
    self._state, results = self._pmapped_scanned_pretrain_step(
        self._state, batch)

    self._num_full_update_steps += self.num_minibatches

    results = jax.tree.map(jnp.mean, results)
    return results

  def compute_loss(self, batch: DataTuple):
    _, results = self._pmapped_scanned_pretrain_step(
        self._state, batch)
    results = jax.tree.map(jnp.mean, results)
    return results

  def get_state(self):
    return get_from_first_device(self._state, as_numpy=True)

  def restore(self, state: TrainingState):
    random_key = state.random_key
    random_key = jax.random.split(
        random_key, num=self.num_local_learner_devices)
    random_key = jax.device_put_sharded(
        [random_key[i] for i in range(self.num_local_learner_devices)],
        self.local_learner_devices)

    state = jax.device_put_replicated(state, self.local_learner_devices)
    state = state._replace(random_key=random_key)
    self._state = state


################################################################################
# Normalization
################################################################################

# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 25)
# MODIFIED: the notebook computes these statistics at cell scope from the
# global `all_tuples` and the normalizer functions capture them as globals;
# here the statistics computation is `compute_normalization_stats(all_tuples)`
# and the normalizers are built by `make_normalizers(stats)` which closes over
# the given stats. The formulas are verbatim.
def compute_normalization_stats(all_tuples):
  cur_pos_mean = jnp.mean(all_tuples.observation['cur_pos'], axis=0, keepdims=True)
  cur_pos_std = jnp.std(all_tuples.observation['cur_pos'], axis=0, keepdims=True)
  cur_vel_mean = jnp.mean(all_tuples.observation['cur_vel'], axis=0, keepdims=True)
  cur_vel_std = jnp.std(all_tuples.observation['cur_vel'], axis=0, keepdims=True)
  act_mean = jnp.mean(all_tuples.action, axis=0, keepdims=True)
  act_std = jnp.std(all_tuples.action, axis=0, keepdims=True)
  return {
      'cur_pos_mean': cur_pos_mean,
      'cur_pos_std': cur_pos_std,
      'cur_vel_mean': cur_vel_mean,
      'cur_vel_std': cur_vel_std,
      'act_mean': act_mean,
      'act_std': act_std,}


def make_normalizers(stats):
  cur_pos_mean = stats['cur_pos_mean']
  cur_pos_std = stats['cur_pos_std']
  cur_vel_mean = stats['cur_vel_mean']
  cur_vel_std = stats['cur_vel_std']
  act_mean = stats['act_mean']
  act_std = stats['act_std']

  def normalize_obs(obs):
    normalized_obs = {
        'cur_pos': (obs['cur_pos'] - cur_pos_mean) / cur_pos_std,
        'cur_vel': (obs['cur_vel'] - cur_vel_mean) / cur_vel_std,
        'goal_pos': (obs['goal_pos'] - cur_pos_mean) / cur_pos_std,}
    return normalized_obs

  def normalize_action(action):
    return (action - act_mean) / act_std

  def unnormalize_action(action):
    return action * act_std + act_mean

  return normalize_obs, normalize_action, unnormalize_action


################################################################################
# RL utilities (rollout + REINFORCE dataset)
################################################################################

# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 39)
# MODIFIED: `_timer_rollout_policy` closes over the notebook globals
# `timer_networks` and `distance_converter`; here it is wrapped in a factory
# taking them as arguments (the notebook then jits it with backend='cpu',
# reproduced here). `evaluate_timer_rollout_policy` takes the notebook globals
# `timer_rollout_policy`, `normalize_obs`, `unnormalize_action`,
# `max_distance` as arguments. Bodies are verbatim.
def make_timer_rollout_policy(timer_networks, distance_converter):
  def _timer_rollout_policy(params, normalized_obs, rng):
    normalized_obs = jnp.concatenate(
        [normalized_obs['cur_pos'], normalized_obs['cur_vel'], normalized_obs['goal_pos']], axis=-1)
    preds = timer_networks.network.apply(params, normalized_obs)
    normalized_act = timer_networks.sample_act(preds.act_dist_params, rng)
    dist_pred = distance_converter.network_format_to_distance(
        preds.dist_to_succ_dist_params.logits)
    return normalized_act, {'pred_dist_to_succ': dist_pred}

  timer_rollout_policy = jax.jit(_timer_rollout_policy, backend='cpu')
  return timer_rollout_policy


def evaluate_timer_rollout_policy(
    env, params, num_episodes,
    timer_rollout_policy, normalize_obs, unnormalize_action, max_distance):
  stats = []
  for eps_num in range(num_episodes):
    timesteps = []
    ts = env.reset()
    ts = ts._replace(reward=0.)
    timesteps.append(ts)

    key = jax.random.PRNGKey(42)

    while (not env.success()) and len(timesteps) < max_distance:
      key, sub_key = jax.random.split(key)
      cur_obs = ts.observation
      cur_obs = jax.tree.map(lambda x: x[None], cur_obs)
      cur_obs = normalize_obs(cur_obs)  # 1 x dims
      norm_act, _ = timer_rollout_policy(params, cur_obs, sub_key)
      unnorm_act = unnormalize_action(norm_act)
      ts = env.step(unnorm_act[0])
      timesteps.append(ts)

    episode_stats = {}
    episode_stats['success'] = env.success()
    episode_stats['return'] = sum(x.reward for x in timesteps)
    episode_stats['len'] = len(timesteps)
    stats.append(episode_stats)

  stats = jax.tree.map(lambda *xs: np.stack(xs), *stats)
  print(f'Success Rate: {np.mean(stats["success"])}')
  print(f'Returns: {np.mean(stats["return"]):.2f} +/- {np.std(stats["return"]):.2f}')
  print(f'Episode Lengths: {np.mean(stats["len"]):.2f} +/- {np.std(stats["len"]):.2f}')
  print(f'Max Episode Length: {np.max(stats["len"])}')
  print(f'Min Episode Length: {np.min(stats["len"])}')


# From: references/self-improving-efms.github.io/pointmass_notebook.ipynb (cell 40)
# MODIFIED: takes the notebook globals `timer_rollout_policy`,
# `normalize_obs`, `unnormalize_action`, `max_distance` as arguments; the
# sanity-check execution code at the bottom of the cell is removed. Body is
# verbatim.
class REINFORCETuple(NamedTuple):
  observation: NestedArray
  action: NestedArray
  weight: NestedArray

def generate_timer_reinforce_dataset(
    env, params, num_steps, gamma,
    timer_rollout_policy, normalize_obs, unnormalize_action, max_distance):
  total_steps = 0
  data_tuples = []
  key = jax.random.PRNGKey(42)

  data_stats = []

  while total_steps < num_steps:
    traj_obs = []
    traj_acts = []
    traj_dist_preds = []
    episode_stats = {}

    episode_steps = 0
    episode_return = 0.
    ts = env.reset()
    cur_obs = ts.observation
    cur_obs = jax.tree.map(lambda x: x[None], cur_obs)
    cur_obs = normalize_obs(cur_obs)  # 1 x dims
    traj_obs.append(cur_obs)

    while (not env.success()) and episode_steps < max_distance:
      sub_key, key = jax.random.split(key)
      norm_act, extras = timer_rollout_policy(params, cur_obs, sub_key)
      unnorm_act = unnormalize_action(norm_act)
      traj_acts.append(norm_act)
      traj_dist_preds.append(extras['pred_dist_to_succ'])

      ts = env.step(unnorm_act[0])
      episode_return += ts.reward
      cur_obs = ts.observation
      cur_obs = jax.tree.map(lambda x: x[None], cur_obs)
      cur_obs = normalize_obs(cur_obs)  # 1 x dims
      traj_obs.append(cur_obs)

      episode_steps += 1

    if episode_steps < 1:
      continue

    episode_stats['success'] = env.success()
    episode_stats['return'] = episode_return
    episode_stats['len'] = episode_steps

    total_steps += len(traj_acts)

    sub_key, key = jax.random.split(key)
    norm_act, extras = timer_rollout_policy(params, cur_obs, sub_key)
    traj_dist_preds.append(extras['pred_dist_to_succ'])

    traj_obs = jax.tree.map(lambda *xs: np.concatenate(xs), *traj_obs)
    traj_acts = jax.tree.map(
        lambda *xs: np.concatenate(xs), *traj_acts)
    traj_dist_preds = jax.tree.map(
        lambda *xs: np.concatenate(xs), *traj_dist_preds)

    rews = -1. * (traj_dist_preds[1:] - traj_dist_preds[:-1])
    weights = []
    temp = 0.
    for i in range(rews.shape[0] - 1, -1, -1):
      weights.append(rews[i] + gamma * temp)
      temp = weights[-1]
    weights = np.array(weights[::-1], dtype=np.float32)


    traj_tuples = REINFORCETuple(
        observation=jax.tree.map(lambda x: x[:-1], traj_obs),
        action=traj_acts,
        weight=weights,
    )
    data_tuples.append(traj_tuples)

    data_stats.append(episode_stats)

  data_tuples = jax.tree.map(
      lambda *xs: np.concatenate(xs), *data_tuples)
  data_stats = jax.tree.map(lambda *xs: np.stack(xs), *data_stats)
  return data_tuples, data_stats
