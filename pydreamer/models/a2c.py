import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor

from .functions import *
from .common import *


class ActorCritic(nn.Module):

    def __init__(self,
                 in_dim,
                 out_actions,
                 hidden_dim=400,
                 hidden_layers=4,
                 layer_norm=True,
                 gamma=0.999,
                 lambda_gae=0.95,
                 entropy_weight=1e-3,
                 target_interval=100,
                 actor_grad='reinforce',
                 actor_dist='onehot',
                 use_vtrace=False,
                 vtrace_rho_clip=1.0,
                 vtrace_c_clip=1.0,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_actions = out_actions
        self.gamma = gamma
        self.lambda_ = lambda_gae
        self.entropy_weight = entropy_weight
        self.target_interval = target_interval
        self.actor_grad = actor_grad
        self.actor_dist = actor_dist
        self.use_vtrace = use_vtrace
        self.rho_clip = vtrace_rho_clip
        self.c_clip = vtrace_c_clip

        actor_out_dim = out_actions if actor_dist == 'onehot' else 2 * out_actions
        self.actor = MLP(in_dim, actor_out_dim, hidden_dim, hidden_layers, layer_norm)
        self.critic = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)
        self.critic_target = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)
        self.critic_target.requires_grad_(False)
        self.train_steps = 0

    def forward_actor(self, features: Tensor) -> D.Distribution:
        y = self.actor.forward(features).float()  # .float() to force float32 on AMP

        if self.actor_dist == 'onehot':
            return D.OneHotCategorical(logits=y)
        
        if self.actor_dist == 'normal_tanh':
            return normal_tanh(y)

        if self.actor_dist == 'tanh_normal':
            return tanh_normal(y)

        assert False, self.actor_dist

    def forward_value(self, features: Tensor) -> Tensor:
        y = self.critic.forward(features)
        return y

    def compute_vtrace(self,
                       policy_logprobs: TensorHM,
                       behavior_logprobs: TensorHM,
                       rewards: TensorHM,
                       values: TensorHM,
                       next_values: TensorHM,
                       terminals: TensorHM,
                       ) -> Tuple[Tensor, Tensor]:
        
        # Importance sampling ratios
        log_rhos = policy_logprobs - behavior_logprobs
        rhos = torch.exp(log_rhos)
        
        # Clipped importance sampling for value function
        cs = torch.clamp(rhos, max=self.c_clip)
        
        # Clipped importance sampling for policy gradient
        clipped_rhos = torch.clamp(rhos, max=self.rho_clip)
        
        # Temporal difference errors
        deltas = clipped_rhos * (rewards + self.gamma * (1.0 - terminals) * next_values - values)
        
        # V-trace targets computed backward
        vs = []
        vs_next = next_values[-1]  # Bootstrap from last value
        
        for i in reversed(range(len(deltas))):
            vs_t = values[i] + deltas[i] + self.gamma * (1.0 - terminals[i]) * cs[i] * (vs_next - next_values[i])
            vs.append(vs_t)
            vs_next = vs_t
            
        vs.reverse()
        vs = torch.stack(vs)
        
        # V-trace advantage: A_t = ρ̄_t * (r_t + γ * vs_{t+1} - V(s_t))
        vs_next_shifted = torch.cat([vs[1:], next_values[-1:]])  # vs_{t+1}
        advantages = clipped_rhos * (rewards + self.gamma * (1.0 - terminals) * vs_next_shifted - values)
        
        return vs, advantages

    def training_step(self,
                      features: TensorJMF,
                      actions: TensorHMA,
                      rewards: TensorJM,
                      terminals: TensorJM,
                      log_only=False,
                      behavior_logprobs: Optional[TensorHM] = None
                      ):
        """
        The ordering is as follows:
            features[0] 
            -> actions[0] -> rewards[1], terminals[1], features[1]
            -> actions[1] -> ...
            ...
            -> actions[H-1] -> rewards[H], terminals[H], features[H]
            
        Args:
            behavior_logprobs: Log probs of actions under behavior policy (for V-trace)
                              If None and use_vtrace=True, assumes on-policy (behavior=current policy)
        """
        if not log_only:
            if self.train_steps % self.target_interval == 0:
                self.update_critic_target()
            self.train_steps += 1

        reward1: TensorHM = rewards[1:]
        terminal0: TensorHM = terminals[:-1]
        terminal1: TensorHM = terminals[1:]

        value_t: TensorJM = self.critic_target.forward(features)
        value0t: TensorHM = value_t[:-1]
        value1t: TensorHM = value_t[1:]
        
        # Compute policy distribution and log probs
        policy_distr = self.forward_actor(features[:-1])
        policy_logprobs = policy_distr.log_prob(actions)
        
        # Choose between V-trace and GAE
        if self.use_vtrace:
            # V-trace off-policy correction
            if behavior_logprobs is None:
                # If no behavior policy provided, assume on-policy (μ = π)
                behavior_logprobs = policy_logprobs.detach()
            
            value_target, advantage_vtrace = self.compute_vtrace(
                policy_logprobs=policy_logprobs.detach(),
                behavior_logprobs=behavior_logprobs,
                rewards=reward1,
                values=value0t,
                next_values=value1t,
                terminals=terminal1
            )
            
            # Apply GAE on top of V-trace for even smoother gradients
            if self.lambda_ < 1.0:
                # Compute residual TD errors after V-trace
                td_errors = value_target - value0t
                advantage_gae = []
                agae = None
                for td_err, term in zip(reversed(td_errors.unbind()), reversed(terminal1.unbind())):
                    if agae is None:
                        agae = td_err
                    else:
                        agae = td_err + self.lambda_ * self.gamma * (1.0 - term) * agae
                    advantage_gae.append(agae)
                advantage_gae.reverse()
                advantage_gae = torch.stack(advantage_gae)
            else:
                advantage_gae = advantage_vtrace
                
        else:
            # Standard GAE
            # GAE from https://arxiv.org/abs/1506.02438 eq (16)
            #   advantage_gae[t] = advantage[t] + (gamma lambda) advantage[t+1] + (gamma lambda)^2 advantage[t+2] + ...
            advantage = - value0t + reward1 + self.gamma * (1.0 - terminal1) * value1t
            advantage_gae = []
            agae = None
            for adv, term in zip(reversed(advantage.unbind()), reversed(terminal1.unbind())):
                if agae is None:
                    agae = adv
                else:
                    agae = adv + self.lambda_ * self.gamma * (1.0 - term) * agae
                advantage_gae.append(agae)
            advantage_gae.reverse()
            advantage_gae = torch.stack(advantage_gae)
            # Note: if lambda=0, then advantage_gae=advantage, then value_target = advantage + value0t = reward + gamma * value1t
            value_target = advantage_gae + value0t

        # When calculating losses, should ignore terminal states, or anything after, so:
        #   reality_weight[i] = (1-terminal[0]) (1-terminal[1]) ... (1-terminal[i])
        # Note this takes care of the case when initial state features[0] is terminal - it will get weighted by (1-terminals[0]).
        reality_weight = (1 - terminal0).log().cumsum(dim=0).exp()

        # Critic loss

        value: TensorJM = self.critic.forward(features)
        value0: TensorHM = value[:-1]
        loss_critic = 0.5 * torch.square(value_target.detach() - value0)
        loss_critic = (loss_critic * reality_weight).mean()

        # Actor loss

        if self.actor_grad == 'reinforce':
            loss_policy = - policy_logprobs * advantage_gae.detach()
        elif self.actor_grad == 'dynamics':
            loss_policy = - value_target
        else:
            assert False, self.actor_grad

        policy_entropy = policy_distr.entropy()
        loss_actor = loss_policy - self.entropy_weight * policy_entropy
        loss_actor = (loss_actor * reality_weight).mean()
        assert (loss_policy.requires_grad and policy_entropy.requires_grad) or not loss_critic.requires_grad

        with torch.no_grad():
            # Compute standard advantage for logging
            advantage = - value0t + reward1 + self.gamma * (1.0 - terminal1) * value1t
            
            metrics = dict(loss_critic=loss_critic.detach(),
                           loss_actor=loss_actor.detach(),
                           policy_entropy=policy_entropy.mean(),
                           policy_value=value0[0].mean(),  # Value of real states
                           policy_value_im=value0.mean(),  # Value of imagined states
                           policy_reward=reward1.mean(),
                           policy_reward_std=reward1.std(),
                           )
            
            if self.use_vtrace and behavior_logprobs is not None:
                # Log importance sampling statistics
                log_rhos = policy_logprobs - behavior_logprobs
                rhos = torch.exp(log_rhos)
                metrics.update(
                    vtrace_rho_mean=rhos.mean(),
                    vtrace_rho_max=rhos.max(),
                    vtrace_rho_clipped=(rhos > self.rho_clip).float().mean(),
                )
            
            tensors = dict(value=value.detach(),
                           value_target=value_target.detach(),
                           value_advantage=advantage.detach(),
                           value_advantage_gae=advantage_gae.detach(),
                           value_weight=reality_weight.detach(),
                           )

        return (loss_actor, loss_critic), metrics, tensors

    def update_critic_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())  # type: ignore
