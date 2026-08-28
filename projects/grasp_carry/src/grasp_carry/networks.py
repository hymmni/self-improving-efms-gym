"""TIMER network builder, ported from pointmass_core.py (SI-EFM paper reproduction).

From: pointmass/pointmass_core.py (build_continuous_act_discrete_dist_v0 and its
supporting types) -- copied rather than imported so grasp_carry has no runtime
dependency on the pointmass/ subproject.
"""

import dataclasses
from typing import Any, Callable, NamedTuple, Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

Params = Any
PRNGKey = Any
NetworkOutput = Any
Entropy = Any
ActDistParams = Params
LogProbFn = Any
SampleFn = Any
Observation = Any
DistanceToSuccessDistParams = Params
EntropyFn = Callable[[Params, PRNGKey], Entropy]

MIN_ACT_SCALE = 1e-2


@dataclasses.dataclass
class FeedForwardNetwork:
  """Holds a pair of pure functions defining a feed-forward network."""
  init: Callable[..., Params]
  apply: Callable[..., NetworkOutput]


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
  """Network and pure functions for the TIMER agent."""
  network: FeedForwardNetwork
  act_log_prob: LogProbFn
  sample_act: SampleFn
  dist_log_prob: LogProbFn
  sample_dist: SampleFn
  act_entropy: Optional[EntropyFn] = None
  sample_act_mode: Optional[SampleFn] = None
  dist_entropy: Optional[EntropyFn] = None
  sample_dist_mode: Optional[SampleFn] = None


def build_continuous_act_discrete_dist_v0(
    layer_sizes: Sequence[int],
    act_dim: int,
    num_dist_bins: int,
    dummy_input,
) -> TIMERNetworks:
  """Builds TIMERNetworks for continuous action and discrete distance."""

  def _network(x: Observation) -> TIMERNetworkOutput:
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

  def act_entropy(params: MVNDiagParams, key: PRNGKey) -> Entropy:
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

  def dist_entropy(params: CategoricalParams, key: PRNGKey) -> Entropy:
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
