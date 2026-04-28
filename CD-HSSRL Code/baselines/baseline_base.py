"""
baseline_base.py — 所有基线方法的公共基类与工具函数

提供统一接口：
  BaseAgent.select_action(state) → action
  BaseAgent.store(...)
  BaseAgent.update() → metrics
  BaseAgent.save/load
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import math

# ── 环境常量 ─────────────────────────────────────────────────────────────────
STATE_DIM  = 25   # 五传感器状态向量维度
ACTION_DIM = 2    # [vx_norm, wz_norm]
LIDAR_IDX  = slice(11, 19)   # LiDAR 8扇区 [11:19]
US_IDX     = slice(19, 23)   # 超声波 4路  [19:23]
DEPTH_IDX  = 23               # 深度传感器  [23]
DOMAIN_IDX = 24               # 域标签      [24]


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def mlp(dims: List[int], activation=nn.ReLU,
        output_activation=None) -> nn.Sequential:
    """通用 MLP 构建。dims = [in, h1, h2, ..., out]"""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(activation())
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


def get_obstacle_min(state: np.ndarray) -> float:
    """从 25 维状态提取 LiDAR 最近障碍物距离。"""
    if len(state) >= 19:
        return float(np.min(state[LIDAR_IDX]))
    return 10.0


def get_water_depth(state: np.ndarray) -> float:
    """从 25 维状态提取深度传感器读数。"""
    return float(state[DEPTH_IDX]) if len(state) > DEPTH_IDX else 0.0


def get_domain(state: np.ndarray) -> int:
    return int(state[DOMAIN_IDX]) if len(state) > DOMAIN_IDX else 0


# ── 经验回放 ─────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """标准经验回放缓冲区。"""

    def __init__(self, capacity: int = 100_000):
        self._buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self._buf.append((
            np.array(s,  np.float32),
            np.array(a,  np.float32),
            float(r),
            np.array(s2, np.float32),
            float(done),
        ))

    def sample(self, batch: int, device='cpu') -> Dict[str, torch.Tensor]:
        b = random.sample(self._buf, batch)
        s, a, r, s2, d = zip(*b)
        def ft(x): return torch.FloatTensor(np.array(x)).to(device)
        return {'s': ft(s), 'a': ft(a), 'r': ft(r).unsqueeze(1),
                's2': ft(s2), 'd': ft(d).unsqueeze(1)}

    def __len__(self): return len(self._buf)


class PrioritizedReplayBuffer:
    """优先经验回放 (PER)，用于 APF-D3QNPER。"""

    def __init__(self, capacity=100_000, alpha=0.6):
        self.cap   = capacity
        self.alpha = alpha
        self.buf   = []
        self.pos   = 0
        self.prios = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, done, error=1.0):
        p = (abs(error) + 1e-6) ** self.alpha
        if len(self.buf) < self.cap:
            self.buf.append(None)
        self.buf[self.pos] = (np.array(s, np.float32), np.array(a, np.float32),
                               float(r), np.array(s2, np.float32), float(done))
        self.prios[self.pos] = p
        self.pos = (self.pos + 1) % self.cap

    def sample(self, batch, beta=0.4, device='cpu'):
        N    = len(self.buf)
        p    = self.prios[:N]
        prob = p / p.sum()
        idx  = np.random.choice(N, batch, replace=False, p=prob)
        b    = [self.buf[i] for i in idx]
        s, a, r, s2, d = zip(*b)
        weights = (N * prob[idx]) ** (-beta)
        weights /= weights.max()
        def ft(x): return torch.FloatTensor(np.array(x)).to(device)
        return {'s': ft(s), 'a': ft(a), 'r': ft(r),
                's2': ft(s2), 'd': ft(d),
                'weights': ft(weights), 'idx': idx}

    def update_priorities(self, idx, errors):
        for i, e in zip(idx, errors):
            self.prios[i] = (abs(e) + 1e-6) ** self.alpha

    def __len__(self): return len(self.buf)


# ── PPO Rollout Buffer ────────────────────────────────────────────────────────

class RolloutBuffer:
    """PPO 轨迹缓冲区。"""

    def __init__(self):
        self.states:    List[np.ndarray] = []
        self.actions:   List[np.ndarray] = []
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[bool]       = []
        self.advantages: np.ndarray      = np.array([])
        self.returns_:   np.ndarray      = np.array([])

    def add(self, s, a, lp, r, v, done):
        self.states.append(np.array(s, np.float32))
        self.actions.append(np.array(a, np.float32))
        self.log_probs.append(float(lp))
        self.rewards.append(float(r))
        self.values.append(float(v))
        self.dones.append(bool(done))

    def compute_gae(self, last_val=0., gamma=0.99, lam=0.95):
        T   = len(self.rewards)
        adv = np.zeros(T, np.float32)
        gae = 0.
        vals = self.values + [last_val]
        for t in reversed(range(T)):
            nv  = vals[t+1] * (1. - float(self.dones[t]))
            d   = self.rewards[t] + gamma * nv - vals[t]
            gae = d + gamma * lam * (1. - float(self.dones[t])) * gae
            adv[t] = gae
        self.advantages = adv
        self.returns_   = adv + np.array(self.values, np.float32)

    def get(self, device='cpu'):
        def ft(x): return torch.FloatTensor(np.array(x)).to(device)
        return {
            's':    ft(self.states),
            'a':    ft(self.actions),
            'lp':   ft(self.log_probs),
            'adv':  ft(self.advantages),
            'ret':  ft(self.returns_),
        }

    def reset(self): self.__init__()
    def __len__(self):  return len(self.rewards)


# ── Actor 网络基类 ────────────────────────────────────────────────────────────

class StochasticActor(nn.Module):
    """连续动作高斯随机 Actor（SAC/DDPG 风格）。"""
    LOG_STD_MIN, LOG_STD_MAX = -5., 2.

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden=256, n_hidden=2):
        super().__init__()
        dims = [state_dim] + [hidden]*n_hidden
        self.net     = mlp(dims)
        self.mu_out  = nn.Linear(hidden, action_dim)
        self.std_out = nn.Linear(hidden, action_dim)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu_out.weight,  gain=0.01)
        nn.init.orthogonal_(self.std_out.weight, gain=0.01)

    def forward(self, s):
        h = self.net(s)
        mu  = self.mu_out(h)
        lsd = self.std_out(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, lsd

    def sample(self, s) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, lsd = self.forward(s)
        std = lsd.exp()
        x   = mu + torch.randn_like(mu) * std
        y   = torch.tanh(x)
        lp  = (Normal(mu, std).log_prob(x)
               - torch.log(1 - y.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return y, lp, torch.tanh(mu)


class DeterministicActor(nn.Module):
    """确定性 Actor（DDPG 风格）。"""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden=256, n_hidden=2):
        super().__init__()
        dims = [state_dim] + [hidden]*n_hidden + [action_dim]
        self.net = mlp(dims, output_activation=nn.Tanh)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, s): return self.net(s)


class QNetwork(nn.Module):
    """Q(s, a) 网络。"""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden=256, n_hidden=2, twin=True):
        super().__init__()
        in_dim = state_dim + action_dim
        dims   = [in_dim] + [hidden]*n_hidden + [1]
        self.q1 = mlp(dims)
        self.q2 = mlp(dims) if twin else None
        self.twin = twin

    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        if self.twin:
            return self.q1(x), self.q2(x)
        return self.q1(x)


class ValueNet(nn.Module):
    """V(s) 网络，用于 PPO Critic。"""

    def __init__(self, state_dim=STATE_DIM, hidden=256, n_hidden=2):
        super().__init__()
        dims = [state_dim] + [hidden]*n_hidden + [1]
        self.net = mlp(dims)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, s): return self.net(s).squeeze(-1)


class PPOActorCritic(nn.Module):
    """PPO Actor-Critic（共享主干或分离）。"""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden=256, n_hidden=2, continuous=True):
        super().__init__()
        self.continuous = continuous
        dims_shared = [state_dim] + [hidden]*n_hidden
        self.trunk  = mlp(dims_shared)
        if continuous:
            self.mu_head  = nn.Linear(hidden, action_dim)
            self.log_std  = nn.Parameter(torch.zeros(action_dim))
        else:
            self.pi_head  = nn.Linear(hidden, action_dim)
        self.v_head = nn.Linear(hidden, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        if self.continuous:
            nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        else:
            nn.init.orthogonal_(self.pi_head.weight, gain=0.01)
        nn.init.orthogonal_(self.v_head.weight, gain=1.0)

    def forward(self, s):
        h = self.trunk(s)
        v = self.v_head(h).squeeze(-1)
        if self.continuous:
            mu  = torch.tanh(self.mu_head(h))
            std = self.log_std.clamp(-4, 2).exp().expand_as(mu)
            dist = Normal(mu, std)
        else:
            logits = self.pi_head(h)
            dist   = Categorical(logits=logits)
        return dist, v

    def get_action(self, s, deterministic=False):
        dist, v = self.forward(s)
        a = dist.mean if deterministic else dist.sample()
        if not self.continuous:
            a = a.unsqueeze(-1).float()
        lp = dist.log_prob(a if self.continuous else
                           a.squeeze(-1).long()).sum(-1) \
             if self.continuous else dist.log_prob(a.squeeze(-1).long())
        return a, lp, v


# ── 基类 ─────────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有基线方法的公共接口。"""

    def __init__(self, name: str, state_dim=STATE_DIM,
                 action_dim=ACTION_DIM, device='cpu'):
        self.name       = name
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = device
        self.total_steps = 0
        self.n_updates   = 0

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        raise NotImplementedError

    def store(self, *args, **kwargs): pass

    def update(self) -> Dict[str, float]: return {}

    def save(self, path: str): pass

    def load(self, path: str): pass

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.FloatTensor(x).unsqueeze(0).to(self.device)

    def __repr__(self):
        return f"{self.name}(state_dim={self.state_dim})"
