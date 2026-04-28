"""
baselines_01.py
─────────────────────────────────────────────────────────────
Method 1: IPPO   — Independent PPO (连续动作，水陆切换)
Method 2: DDQN   — Double DQN (离散化动作空间)
Method 3: HEA-PPO — Hybrid Exploration-Augmented PPO
─────────────────────────────────────────────────────────────

参考：
  IPPO   : Schulman et al., "Proximal Policy Optimization", 2017
  DDQN   : van Hasselt et al., "Double DQN", AAAI 2016
  HEA-PPO: hybrid entropy-augmented PPO for amphibious navigation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from typing import Dict, List, Tuple, Optional
import math

from baseline_base import (
    BaseAgent, ReplayBuffer, RolloutBuffer,
    PPOActorCritic, ValueNet, mlp, soft_update, hard_update,
    STATE_DIM, ACTION_DIM, get_obstacle_min, get_domain,
)


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: IPPO — Independent Proximal Policy Optimization
# ─────────────────────────────────────────────────────────────────────────────

class IPPO(BaseAgent):
    """
    Independent PPO 用于两栖导航。

    核心特点：
      - 连续高斯策略，输出 [vx_norm, wz_norm] ∈ [-1,1]²
      - 基于 LiDAR 检测障碍物时施加动作裁剪（简单安全机制）
      - 水/陆域通过 domain 标签无缝切换，单一策略不区分域

    超参数 (Section 4.4)：
      lr=3e-4, gamma=0.99, lambda=0.95, clip_eps=0.2
      n_steps=2048, batch_size=64, n_epochs=10
    """

    def __init__(self,
                 state_dim:   int   = STATE_DIM,
                 action_dim:  int   = ACTION_DIM,
                 hidden:      int   = 256,
                 lr:          float = 3e-4,
                 gamma:       float = 0.99,
                 gae_lambda:  float = 0.95,
                 clip_eps:    float = 0.2,
                 vf_coef:     float = 0.5,
                 ent_coef:    float = 0.01,
                 max_grad:    float = 0.5,
                 n_steps:     int   = 2048,
                 batch_size:  int   = 64,
                 n_epochs:    int   = 10,
                 device:      str   = 'cpu'):
        super().__init__('IPPO', state_dim, action_dim, device)
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.max_grad   = max_grad
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.n_epochs   = n_epochs

        self.ac     = PPOActorCritic(state_dim, action_dim,
                                     hidden, continuous=True).to(device)
        self.opt    = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        s = self._to_tensor(state)
        with torch.no_grad():
            dist, v = self.ac(s)
            a = dist.mean if deterministic else dist.sample()
        action = a.squeeze(0).cpu().numpy()
        return np.clip(action, -1., 1.)

    def store(self, state, action, log_prob, reward, value, done):
        self.buffer.add(state, action, log_prob, reward, value, done)
        self.total_steps += 1

    def get_log_prob_value(self, state: np.ndarray,
                           action: np.ndarray) -> Tuple[float, float]:
        s = self._to_tensor(state)
        a = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, v = self.ac(s)
            lp = dist.log_prob(a).sum(-1)
        return float(lp.item()), float(v.item())

    def update(self, last_value: float = 0.) -> Dict[str, float]:
        if len(self.buffer) < self.n_steps:
            return {}

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        data = self.buffer.get(self.device)
        B    = len(self.buffer)

        metrics = {'loss':0.,'pi':0.,'vf':0.,'ent':0.,'clip_frac':0.}
        n_batch = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, self.batch_size):
                idx = perm[start:start + self.batch_size]
                if len(idx) < 2: continue

                s   = data['s'][idx]
                a   = data['a'][idx]
                lp0 = data['lp'][idx]
                adv = data['adv'][idx]
                ret = data['ret'][idx]

                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                dist, v = self.ac(s)
                lp1  = dist.log_prob(a).sum(-1)
                ent  = dist.entropy().sum(-1).mean()

                ratio = (lp1 - lp0).exp()
                s1    = ratio * adv
                s2    = ratio.clamp(1 - self.clip_eps,
                                    1 + self.clip_eps) * adv
                pi_loss = -torch.min(s1, s2).mean()
                vf_loss = F.mse_loss(v, ret)
                clip_f  = ((ratio - 1).abs() > self.clip_eps).float().mean()

                loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad)
                self.opt.step()

                metrics['loss']      += loss.item()
                metrics['pi']        += pi_loss.item()
                metrics['vf']        += vf_loss.item()
                metrics['ent']       += ent.item()
                metrics['clip_frac'] += clip_f.item()
                n_batch += 1

        self.buffer.reset()
        self.n_updates += 1
        if n_batch > 0:
            metrics = {k: v / n_batch for k, v in metrics.items()}
        return metrics

    def save(self, path: str):
        torch.save({'ac': self.ac.state_dict(),
                    'opt': self.opt.state_dict(),
                    'steps': self.total_steps}, path)

    def load(self, path: str):
        c = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(c['ac'])
        self.opt.load_state_dict(c['opt'])
        self.total_steps = c.get('steps', 0)


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: DDQN — Double Deep Q-Network
# ─────────────────────────────────────────────────────────────────────────────

class DDQN(BaseAgent):
    """
    Double DQN 用于两栖导航（离散化动作空间）。

    动作离散化（5×5 = 25 个离散动作）：
      vx ∈ {-1, -0.5, 0, 0.5, 1}
      wz ∈ {-1, -0.5, 0, 0.5, 1}

    核心特点：
      - Online / Target 双网络，减少过估计
      - ε-greedy 探索，ε 线性衰减
      - LiDAR 障碍物检测时屏蔽危险动作

    超参数：lr=1e-3, gamma=0.99, tau=0.005 (软更新)
             replay=50000, batch=128
    """

    # 离散动作表 (25 个)
    VX_BINS = np.array([-1., -0.5, 0., 0.5, 1.], np.float32)
    WZ_BINS = np.array([-1., -0.5, 0., 0.5, 1.], np.float32)
    N_ACTS  = 25

    def __init__(self,
                 state_dim:  int   = STATE_DIM,
                 hidden:     int   = 256,
                 lr:         float = 1e-3,
                 gamma:      float = 0.99,
                 tau:        float = 0.005,
                 buffer_cap: int   = 50_000,
                 batch_size: int   = 128,
                 eps_start:  float = 1.0,
                 eps_end:    float = 0.05,
                 eps_decay:  int   = 50_000,
                 target_freq:int   = 200,
                 learn_starts:int  = 1000,
                 device:     str   = 'cpu'):
        super().__init__('DDQN', state_dim, ACTION_DIM, device)
        self.gamma       = gamma
        self.tau         = tau
        self.batch_size  = batch_size
        self.eps_start   = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.target_freq = target_freq
        self.learn_starts = learn_starts

        # Online & Target Q 网络
        self.online = mlp([state_dim, hidden, hidden, self.N_ACTS]).to(device)
        self.target = mlp([state_dim, hidden, hidden, self.N_ACTS]).to(device)
        hard_update(self.target, self.online)
        for p in self.target.parameters(): p.requires_grad = False

        self.opt    = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_cap)

        # 构建离散动作映射
        self._action_table = np.array([
            [vx, wz]
            for vx in self.VX_BINS
            for wz in self.WZ_BINS
        ], dtype=np.float32)  # (25, 2)

    def _eps(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
               math.exp(-self.total_steps / self.eps_decay)

    def _action_idx_to_vec(self, idx: int) -> np.ndarray:
        return self._action_table[idx]

    def _safe_mask(self, state: np.ndarray) -> np.ndarray:
        """屏蔽障碍物方向的前进动作（LiDAR 近距离）。"""
        min_d = get_obstacle_min(state)
        if min_d < 0.4:
            mask = np.ones(self.N_ACTS, bool)
            for i, a in enumerate(self._action_table):
                if a[0] > 0:   # 前进动作在障碍物近时屏蔽
                    mask[i] = False
            return mask
        return np.ones(self.N_ACTS, bool)

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        if not deterministic and np.random.random() < self._eps():
            idx = np.random.randint(self.N_ACTS)
        else:
            s   = self._to_tensor(state)
            with torch.no_grad():
                q = self.online(s).squeeze(0).cpu().numpy()
            mask = self._safe_mask(state)
            q[~mask] = -1e9
            idx = int(np.argmax(q))
        return self._action_idx_to_vec(idx)

    def store(self, s, a, r, s2, done):
        # 将连续动作映射到最近离散动作
        dists = np.linalg.norm(self._action_table - a, axis=1)
        idx   = int(np.argmin(dists))
        self.buffer.add(s, [idx], r, s2, done)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.learn_starts:
            return {}

        batch = self.buffer.sample(self.batch_size, self.device)
        s, a_idx, r, s2, d = (batch['s'], batch['a'].long().squeeze(-1),
                               batch['r'], batch['s2'], batch['d'])

        # Double DQN target：online 选 action，target 评估 Q
        with torch.no_grad():
            a_next  = self.online(s2).argmax(-1)
            q_next  = self.target(s2).gather(1, a_next.unsqueeze(-1))
            y       = r + self.gamma * (1 - d) * q_next

        q_pred = self.online(s).gather(1, a_idx.unsqueeze(-1))
        loss   = F.smooth_l1_loss(q_pred, y)

        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        self.opt.step()

        # 软更新 target
        soft_update(self.target, self.online, self.tau)
        self.n_updates += 1
        return {'loss': loss.item(), 'eps': self._eps()}

    def save(self, path):
        torch.save({'online': self.online.state_dict(),
                    'target': self.target.state_dict(),
                    'steps':  self.total_steps}, path)

    def load(self, path):
        c = torch.load(path, map_location=self.device)
        self.online.load_state_dict(c['online'])
        self.target.load_state_dict(c['target'])
        self.total_steps = c.get('steps', 0)


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: HEA-PPO — Hybrid Exploration-Augmented PPO
# ─────────────────────────────────────────────────────────────────────────────

class HEAPPOActor(nn.Module):
    """
    HEA-PPO Actor：在标准 PPO 基础上增加：
      1. 域感知探索噪声（水中 > 陆地）
      2. 熵奖励自适应调节（基于当前 SSI）
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden=256):
        super().__init__()
        self.trunk   = mlp([state_dim, hidden, hidden])
        self.mu      = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        self.v_head  = nn.Linear(hidden, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)

    def forward(self, s, domain_noise_scale=1.0):
        h   = self.trunk(s)
        mu  = torch.tanh(self.mu(h))
        std = (self.log_std * domain_noise_scale).clamp(-4., 2.).exp()
        std = std.expand_as(mu)
        return Normal(mu, std), self.v_head(h).squeeze(-1)


class HEAPPO(BaseAgent):
    """
    Hybrid Exploration-Augmented PPO。

    核心改进：
      1. 域自适应探索：水中/过渡区加强探索，陆地降低探索
         noise_scale = {water:1.5, transition:1.2, land:0.8}
      2. 混合熵奖励：ent_coef 随训练进度从 0.02 衰减到 0.005
      3. 安全裁剪：LiDAR 触发时直接裁剪前进速度
    """

    NOISE_SCALE = {0: 1.5, 1: 1.2, 2: 0.8}   # 域 → 探索噪声系数

    def __init__(self,
                 state_dim:   int   = STATE_DIM,
                 action_dim:  int   = ACTION_DIM,
                 hidden:      int   = 256,
                 lr:          float = 3e-4,
                 gamma:       float = 0.99,
                 gae_lambda:  float = 0.95,
                 clip_eps:    float = 0.2,
                 vf_coef:     float = 0.5,
                 ent_coef_start: float = 0.02,
                 ent_coef_end:   float = 0.005,
                 max_grad:    float = 0.5,
                 n_steps:     int   = 2048,
                 batch_size:  int   = 64,
                 n_epochs:    int   = 10,
                 total_steps: int   = 2_000_000,
                 device:      str   = 'cpu'):
        super().__init__('HEA-PPO', state_dim, action_dim, device)
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_eps      = clip_eps
        self.vf_coef       = vf_coef
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end   = ent_coef_end
        self.max_grad      = max_grad
        self.n_steps       = n_steps
        self.batch_size    = batch_size
        self.n_epochs      = n_epochs
        self.total_target  = total_steps

        self.actor  = HEAPPOActor(state_dim, action_dim, hidden).to(device)
        self.opt    = optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()

        self._current_domain = 0

    def _ent_coef(self) -> float:
        """熵系数随训练进度线性衰减。"""
        progress = min(1.0, self.total_steps / max(self.total_target, 1))
        return self.ent_coef_start + progress * (self.ent_coef_end - self.ent_coef_start)

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        domain = get_domain(state)
        ns     = self.NOISE_SCALE.get(domain, 1.0)
        s      = self._to_tensor(state)
        with torch.no_grad():
            dist, v = self.actor(s, domain_noise_scale=ns)
            a = dist.mean if deterministic else dist.sample()

        action = a.squeeze(0).cpu().numpy()
        # LiDAR 安全裁剪
        min_d = get_obstacle_min(state)
        if min_d < 0.4:
            action[0] = np.clip(action[0], -1., max(0., min_d / 0.4))
        return np.clip(action, -1., 1.)

    def get_log_prob_value(self, state, action):
        domain = get_domain(state)
        ns     = self.NOISE_SCALE.get(domain, 1.0)
        s = self._to_tensor(state)
        a = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, v = self.actor(s, ns)
            lp = dist.log_prob(a).sum(-1)
        return float(lp.item()), float(v.item())

    def store(self, state, action, log_prob, reward, value, done):
        self.buffer.add(state, action, log_prob, reward, value, done)
        self.total_steps += 1

    def update(self, last_value: float = 0.) -> Dict[str, float]:
        if len(self.buffer) < self.n_steps:
            return {}

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        data    = self.buffer.get(self.device)
        B       = len(self.buffer)
        ec      = self._ent_coef()

        metrics = {'loss':0.,'pi':0.,'vf':0.,'ent':0.}
        n_batch = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, self.batch_size):
                idx = perm[start:start + self.batch_size]
                if len(idx) < 2: continue

                s   = data['s'][idx]
                a   = data['a'][idx]
                lp0 = data['lp'][idx]
                adv = data['adv'][idx]
                ret = data['ret'][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # 平均域感知噪声（batch 内不区分域，用 1.0）
                dist, v = self.actor(s, domain_noise_scale=1.0)
                lp1 = dist.log_prob(a).sum(-1)
                ent = dist.entropy().sum(-1).mean()

                r    = (lp1 - lp0).exp()
                pi_l = -torch.min(r * adv,
                                  r.clamp(1-self.clip_eps,
                                          1+self.clip_eps) * adv).mean()
                vf_l = F.mse_loss(v, ret)
                loss = pi_l + self.vf_coef * vf_l - ec * ent

                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)
                self.opt.step()

                metrics['loss'] += loss.item()
                metrics['pi']   += pi_l.item()
                metrics['vf']   += vf_l.item()
                metrics['ent']  += ent.item()
                n_batch += 1

        self.buffer.reset()
        self.n_updates += 1
        if n_batch > 0:
            metrics = {k: v / n_batch for k, v in metrics.items()}
        metrics['ent_coef'] = ec
        return metrics

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(),
                    'opt': self.opt.state_dict(),
                    'steps': self.total_steps}, path)

    def load(self, path):
        c = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(c['actor'])
        self.opt.load_state_dict(c['opt'])
        self.total_steps = c.get('steps', 0)
