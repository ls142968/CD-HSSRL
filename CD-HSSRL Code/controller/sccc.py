#!/usr/bin/env python3
"""
sccc.py — Safety-Constrained Continuous Controller
论文 Section 3.5, Eq.(17)–(20)

四个核心组件：
  1. SACNetworks        — π_L actor + Q_θ critic 网络定义
  2. ReplayBuffer       — 经验回放缓冲区 D
  3. SafetyProjection   — 安全投影层 a_safe (Eq.18)
  4. SCCC               — 顶层 SAC + 安全 + 风险奖励 (Eq.17-20)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random


# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SCCCConfig:
    # 网络结构 (Section 4.4)
    state_dim:    int   = 17
    action_dim:   int   = 2
    n_options:    int   = 3       # one-hot option 编码
    hidden_size:  int   = 256
    n_hidden:     int   = 2

    # SAC 超参数 (Section 4.4)
    lr:           float = 3e-4
    gamma:        float = 0.99
    tau:          float = 0.005   # 软更新系数
    batch_size:   int   = 256
    buffer_size:  int   = 1_000_000
    learning_starts: int = 1000
    gradient_steps:  int = 1

    # 自动熵调节
    target_entropy: float = -2.0   # = -dim(A)

    # 安全约束 (Eq.18)
    collision_thresh: float = 0.4   # m，超声波距离阈值
    grounding_thresh: float = 0.05  # m，z坐标阈值（地面）

    # 风险惩罚 (Eq.19)
    kappa: float = 1.0             # κ，风险惩罚系数

    # 动作范围
    action_max: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. 网络定义  (Eq.17)
# ─────────────────────────────────────────────────────────────────────────────

LOG_STD_MIN = -5.0
LOG_STD_MAX =  2.0


def _mlp(input_dim: int, hidden_size: int, n_hidden: int,
         output_dim: int) -> nn.Sequential:
    """构建 MLP：n_hidden × Linear(hidden_size) + ReLU。"""
    layers = []
    in_d   = input_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(in_d, hidden_size), nn.ReLU()]
        in_d    = hidden_size
    layers.append(nn.Linear(in_d, output_dim))
    return nn.Sequential(*layers)


class LowLevelActor(nn.Module):
    """
    π_L(a|s_t, o_t; θ_L) — 随机 Actor  (Eq.17)

    输入: [state(17), option_onehot(3)]  → dim = 20
    输出: 高斯分布的 mean 和 log_std
          动作经 tanh 压缩到 [-1, 1]²

    (Section 4.4: "actor–critic architecture with two fully connected
                   hidden layers of 256 units, ReLU activation")
    """

    def __init__(self, cfg: SCCCConfig):
        super().__init__()
        input_dim = cfg.state_dim + cfg.n_options
        self.net      = _mlp(input_dim, cfg.hidden_size,
                             cfg.n_hidden, cfg.hidden_size)
        self.mean_out = nn.Linear(cfg.hidden_size, cfg.action_dim)
        self.std_out  = nn.Linear(cfg.hidden_size, cfg.action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_out.weight, gain=0.01)
        nn.init.orthogonal_(self.std_out.weight,  gain=0.01)

    def forward(self, state: torch.Tensor,
                option_oh: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (mean, log_std)，均未经 tanh。"""
        x       = torch.cat([state, option_oh], dim=-1)
        h       = self.net(x)
        mean    = self.mean_out(h)
        log_std = self.std_out(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor,
               option_oh: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        重参数化采样：a ~ π_L(a|s,o)
        Returns:
            action   : tanh 压缩后的动作 ∈ [-1,1]²
            log_prob : log π_L(a|s,o)（含 Jacobian 修正）
            mean     : deterministic mean（评估时用）
        """
        mean, log_std = self.forward(state, option_oh)
        std  = log_std.exp()
        eps  = torch.randn_like(mean)
        x_t  = mean + eps * std          # 重参数化

        # tanh 压缩
        y_t  = torch.tanh(x_t)

        # log prob：正态分布 log prob + Jacobian 修正
        log_prob_normal = Normal(mean, std).log_prob(x_t)
        # 修正：∂tanh(x)/∂x = 1 - tanh²(x)
        log_prob = (log_prob_normal
                    - torch.log(1.0 - y_t.pow(2) + 1e-6)
                    ).sum(dim=-1, keepdim=True)

        return y_t, log_prob, torch.tanh(mean)


class SoftQNetwork(nn.Module):
    """
    Q_θQ(s_t, a_t) — 软 Q 网络  (Eq.20)

    输入: [state(17), option_onehot(3), action(2)]  → dim = 22
    输出: Q 值标量

    使用 Twin Q-Networks 减少过估计。
    """

    def __init__(self, cfg: SCCCConfig):
        super().__init__()
        input_dim = cfg.state_dim + cfg.n_options + cfg.action_dim
        self.q1 = _mlp(input_dim, cfg.hidden_size, cfg.n_hidden, 1)
        self.q2 = _mlp(input_dim, cfg.hidden_size, cfg.n_hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state:     torch.Tensor,
                      option_oh: torch.Tensor,
                      action:    torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, option_oh, action], dim=-1)
        return self.q1(x), self.q2(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 经验回放缓冲区 D
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    论文 Algorithm 1 中的 Replay Buffer D。
    存储 (s, o, a, r_safe, s', o', done) 元组。
    """

    def __init__(self, capacity: int = 1_000_000):
        self._buf = deque(maxlen=capacity)

    def add(self, state:      np.ndarray,
                  option:     int,
                  action:     np.ndarray,
                  reward:     float,
                  next_state: np.ndarray,
                  next_option:int,
                  done:       bool):
        self._buf.append((
            state.astype(np.float32),
            np.int32(option),
            action.astype(np.float32),
            np.float32(reward),
            next_state.astype(np.float32),
            np.int32(next_option),
            np.float32(done),
        ))

    def sample(self, batch_size: int,
               device: str = 'cpu') -> Dict[str, torch.Tensor]:
        batch = random.sample(self._buf, batch_size)
        (states, options, actions, rewards,
         next_states, next_options, dones) = zip(*batch)

        def ft(x): return torch.FloatTensor(np.array(x)).to(device)
        def li(x): return torch.LongTensor(np.array(x)).to(device)

        # one-hot 编码 option
        n = len(options)
        oh      = torch.zeros(n, 3, device=device)
        next_oh = torch.zeros(n, 3, device=device)
        for i, (o, no) in enumerate(zip(options, next_options)):
            oh[i, o]       = 1.0
            next_oh[i, no] = 1.0

        return {
            'states':      ft(states),
            'options':     li(options),
            'option_oh':   oh,
            'actions':     ft(actions),
            'rewards':     ft(rewards).unsqueeze(1),
            'next_states': ft(next_states),
            'next_oh':     next_oh,
            'dones':       ft(dones).unsqueeze(1),
        }

    def __len__(self): return len(self._buf)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 安全投影层  (Eq.18)
# ─────────────────────────────────────────────────────────────────────────────

class SafetyProjection:
    """
    a_safe = argmin_{a∈A} ||a - a_t||²   s.t. g(s_t, a) ≤ 0   (Eq.18)

    安全约束 g(s, a) 编码：
      g1: 碰撞避免   — 超声波距离 < thresh 时限制前进
      g2: 搁浅防止   — 过渡区/浅水区限制速度
      g3: 稳定性约束 — 过渡区限制角速度防止侧翻

    实现采用梯度自由的解析投影（QP 解析解），O(1) 时间复杂度。
    """

    def __init__(self, cfg: SCCCConfig):
        self.col_thresh   = cfg.collision_thresh
        self.gnd_thresh   = cfg.grounding_thresh
        self.kappa        = cfg.kappa

    # ------------------------------------------------------------------
    def compute_collision_risk(self, state: np.ndarray) -> float:
        """
        P(collision|s_t) — 用于 Eq.19 的风险估计。
        基于超声波最小距离的 sigmoid 风险函数。

        State 索引：[8:12] 为 4 个超声波距离。
        """
        if len(state) > 12:
            us_dists = state[8:12]
            min_dist = float(np.min(us_dists))
        else:
            min_dist = 5.0

        # sigmoid: risk → 1 当 dist → 0，risk → 0 当 dist → ∞
        # 斜率系数 k=8 使过渡在 0.4m 附近发生
        risk = 1.0 / (1.0 + np.exp(8.0 * (min_dist - self.col_thresh)))
        return float(np.clip(risk, 0.0, 1.0))

    # ------------------------------------------------------------------
    def project(self, raw_action: np.ndarray,
                state:      np.ndarray,
                domain:     int) -> np.ndarray:
        """
        将原始动作 a_t 投影到安全集合。

        解析 QP 投影步骤：
          1. 计算各约束的允许动作范围
          2. 取各约束交集
          3. 将 a_t 裁剪到交集内（最小 L2 距离投影）

        Args:
            raw_action : [vx, wz] ∈ [-1, 1]²
            state      : 17维状态向量
            domain     : 0=水 1=过渡 2=陆

        Returns:
            safe_action : [vx_safe, wz_safe] ∈ [-1, 1]²
        """
        a = raw_action.copy().astype(np.float64)

        # 超声波最小距离（4个传感器）
        us    = state[8:12] if len(state) > 12 else np.ones(4) * 5.0
        min_d = float(np.min(us))
        risk  = self.compute_collision_risk(state)

        # ── 约束 g1: 碰撞避免 ─────────────────────────────────────
        # 当障碍物很近时，限制前进速度（不能向障碍物方向运动）
        if min_d < self.col_thresh:
            # 线性渐进限速：dist=0 → vx_max=0, dist=thresh → vx_max=1
            vx_max = max(0.0, min_d / self.col_thresh)
            a[0] = np.clip(a[0], -1.0, vx_max)

        # ── 约束 g2: 过渡区稳定性 ─────────────────────────────────
        # 过渡区限制速度防止冲击（坡面稳定性）
        if domain == 1:
            a[0] = np.clip(a[0], -0.6, 0.6)   # 限制前进速度
            a[1] = np.clip(a[1], -0.5, 0.5)   # 限制转向

        # ── 约束 g3: 风险自适应减速 ────────────────────────────────
        # risk 越高，速度上限越低
        if risk > 0.3:
            speed_scale = max(0.1, 1.0 - risk)
            a[0] *= speed_scale

        # ── 硬约束：动作空间边界 ──────────────────────────────────
        a = np.clip(a, -1.0, 1.0)

        return a.astype(np.float32)

    # ------------------------------------------------------------------
    def risk_shaped_reward(self, base_reward: float,
                           state: np.ndarray) -> Tuple[float, float]:
        """
        R_safe = R_t - κ · P(collision or grounding | s_t, a_t)   (Eq.19)

        Returns:
            (r_safe, risk)
        """
        risk   = self.compute_collision_risk(state)
        r_safe = base_reward - self.kappa * risk
        return float(r_safe), float(risk)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAC 更新器  (Eq.20)
# ─────────────────────────────────────────────────────────────────────────────

class SACUpdater:
    """
    SAC 优化目标 (Eq.20):
        L_L(θ_L) = E_{(s,a)~D} [ α·log π_L(a|s,o) - Q_θQ(s,a) ]

    包含：
      - Critic 更新（soft Bellman residual）
      - Actor 更新（最大化 E[Q - α·log π]）
      - 自动熵系数 α 调节
      - Target network 软更新（τ）
    """

    def __init__(self, actor:  LowLevelActor,
                       critic: SoftQNetwork,
                       cfg:    SCCCConfig,
                       device: str = 'cpu'):
        self.actor  = actor
        self.critic = critic
        self.cfg    = cfg
        self.device = device

        # Target network（不参与梯度）
        self.critic_target = SoftQNetwork(cfg).to(device)
        self.critic_target.load_state_dict(critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # 自动熵系数 α（论文 Section 4.4: "entropy coefficient α is automatically tuned"）
        self.log_alpha  = torch.zeros(1, requires_grad=True, device=device)
        self.alpha      = self.log_alpha.exp().item()
        self.target_ent = cfg.target_entropy   # = -dim(A) = -2

        # 优化器
        self.actor_opt  = optim.Adam(actor.parameters(),  lr=cfg.lr)
        self.critic_opt = optim.Adam(critic.parameters(), lr=cfg.lr)
        self.alpha_opt  = optim.Adam([self.log_alpha],    lr=cfg.lr)

        self.n_updates = 0

    # ------------------------------------------------------------------
    def update(self, buffer: ReplayBuffer) -> Dict[str, float]:
        """
        执行一步 SAC 更新（gradient_steps 次）。
        Returns: 各损失项均值
        """
        if len(buffer) < self.cfg.learning_starts:
            return {}

        metrics = {
            'q_loss': 0., 'actor_loss': 0.,
            'alpha_loss': 0., 'alpha': 0.,
        }

        for _ in range(self.cfg.gradient_steps):
            batch = buffer.sample(self.cfg.batch_size, self.device)
            s     = batch['states']
            oh    = batch['option_oh']
            a     = batch['actions']
            r     = batch['rewards']
            ns    = batch['next_states']
            noh   = batch['next_oh']
            done  = batch['dones']

            # ── Critic 更新 ─────────────────────────────────────────
            with torch.no_grad():
                # 下一步动作采样
                na, log_pi_next, _ = self.actor.sample(ns, noh)
                # Target Q
                q1_t, q2_t = self.critic_target(ns, noh, na)
                q_min      = torch.min(q1_t, q2_t)
                # soft Bellman target（含熵项）
                y = r + self.cfg.gamma * (1.0 - done) * (
                    q_min - self.alpha * log_pi_next)

            q1, q2     = self.critic(s, oh, a)
            q_loss     = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            self.critic_opt.zero_grad()
            q_loss.backward()
            self.critic_opt.step()

            # ── Actor 更新  L_L (Eq.20) ─────────────────────────────
            new_a, log_pi, _ = self.actor.sample(s, oh)
            q1_pi, q2_pi     = self.critic(s, oh, new_a)
            q_pi             = torch.min(q1_pi, q2_pi)
            # L_L = E[α·log π_L - Q]  →  最大化 E[Q - α·log π]
            actor_loss = (self.alpha * log_pi - q_pi).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # ── 熵系数 α 更新 ─────────────────────────────────────
            alpha_loss = -(self.log_alpha * (
                log_pi + self.target_ent).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

            # ── Target network 软更新 τ ────────────────────────────
            for p, pt in zip(self.critic.parameters(),
                             self.critic_target.parameters()):
                pt.data.copy_(
                    self.cfg.tau * p.data + (1 - self.cfg.tau) * pt.data)

            metrics['q_loss']     += q_loss.item()
            metrics['actor_loss'] += actor_loss.item()
            metrics['alpha_loss'] += alpha_loss.item()
            metrics['alpha']      += self.alpha

        n = self.cfg.gradient_steps
        self.n_updates += 1
        return {k: v / n for k, v in metrics.items()}

    # ------------------------------------------------------------------
    def soft_update_target(self):
        """手动触发软更新（外部调用）。"""
        for p, pt in zip(self.critic.parameters(),
                         self.critic_target.parameters()):
            pt.data.copy_(
                self.cfg.tau * p.data + (1 - self.cfg.tau) * pt.data)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 顶层接口：SCCC
# ─────────────────────────────────────────────────────────────────────────────

class SafetyConstrainedContinuousController:
    """
    Safety-Constrained Continuous Controller (Section 3.5)

    完整流程（单步）:
      1. 采样原始动作 a_t ~ π_L(a|s_t, o_t)        (Eq.17)
      2. 投影到安全集合 a_safe                       (Eq.18)
      3. 执行 a_safe，计算 R_safe                    (Eq.19)
      4. 存入 D，执行 SAC 更新 L_L                   (Eq.20)
    """

    def __init__(self, cfg: SCCCConfig, device: str = 'cpu'):
        self.cfg    = cfg
        self.device = device

        # 网络
        self.actor  = LowLevelActor(cfg).to(device)
        self.critic = SoftQNetwork(cfg).to(device)

        # 模块
        self.buffer   = ReplayBuffer(cfg.buffer_size)
        self.safety   = SafetyProjection(cfg)
        self.updater  = SACUpdater(self.actor, self.critic, cfg, device)

        self.total_steps = 0

    # ------------------------------------------------------------------
    def option_onehot(self, option: int) -> torch.Tensor:
        """option 整数 → one-hot tensor (1, n_options)。"""
        oh = torch.zeros(1, self.cfg.n_options, device=self.device)
        oh[0, option] = 1.0
        return oh

    # ------------------------------------------------------------------
    def select_action(self, state:      np.ndarray,
                            option:     int,
                            deterministic: bool = False
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 1+2: 采样并投影。

        Returns:
            safe_action : 投影后安全动作 ∈ [-1,1]²
            raw_action  : 原始动作（存 buffer 用）
        """
        s_t  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        oh   = self.option_onehot(option)

        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(s_t, oh)
            else:
                action, _, _ = self.actor.sample(s_t, oh)

        raw_action  = action.squeeze(0).cpu().numpy()
        domain      = int(state[-1])
        safe_action = self.safety.project(raw_action, state, domain)

        return safe_action, raw_action

    # ------------------------------------------------------------------
    def compute_safe_reward(self, base_reward: float,
                            state: np.ndarray
                            ) -> Tuple[float, float]:
        """Step 3: R_safe = R_t - κ·P(collision) (Eq.19)。"""
        return self.safety.risk_shaped_reward(base_reward, state)

    # ------------------------------------------------------------------
    def store(self, state:       np.ndarray,
                    option:      int,
                    action:      np.ndarray,
                    reward:      float,
                    next_state:  np.ndarray,
                    next_option: int,
                    done:        bool):
        """Step 4a: 存入 Replay Buffer D。"""
        self.buffer.add(state, option, action, reward,
                        next_state, next_option, done)
        self.total_steps += 1

    # ------------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        """Step 4b: SAC 更新 L_L (Eq.20)。"""
        return self.updater.update(self.buffer)

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            'actor':       self.actor.state_dict(),
            'critic':      self.critic.state_dict(),
            'critic_tgt':  self.updater.critic_target.state_dict(),
            'log_alpha':   self.updater.log_alpha.data,
            'n_updates':   self.updater.n_updates,
            'total_steps': self.total_steps,
        }, path)
        print(f"[SCCC] 保存到 {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.updater.critic_target.load_state_dict(ckpt['critic_tgt'])
        self.updater.log_alpha.data = ckpt['log_alpha']
        self.updater.alpha          = self.updater.log_alpha.exp().item()
        self.updater.n_updates      = ckpt['n_updates']
        self.total_steps            = ckpt['total_steps']
        print(f"[SCCC] 从 {path} 加载，已更新 {self.updater.n_updates} 次")
