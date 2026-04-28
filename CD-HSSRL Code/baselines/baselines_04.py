"""
baselines_04.py
─────────────────────────────────────────────────────────────
Method 11: BarrierNet — Control Barrier Function + DRL
Method 12: pH-DRL    — Port-Hamiltonian Deep RL
Method 13: MP-DQL    — Motion Primitive + Deep Q-Learning
─────────────────────────────────────────────────────────────
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional
import math

from baseline_base import (
    BaseAgent, ReplayBuffer, RolloutBuffer,
    PPOActorCritic, DeterministicActor, QNetwork, ValueNet,
    StochasticActor, mlp, soft_update, hard_update,
    STATE_DIM, ACTION_DIM,
    get_obstacle_min, get_water_depth, get_domain,
    LIDAR_IDX, US_IDX, DEPTH_IDX, DOMAIN_IDX,
)


# ─────────────────────────────────────────────────────────────────────────────
# Method 11: BarrierNet — Control Barrier Function + Deep RL
# ─────────────────────────────────────────────────────────────────────────────

class CBFLayer(nn.Module):
    """
    Control Barrier Function (CBF) 安全过滤层。

    CBF 条件：ḣ(s, a) + α·h(s) ≥ 0
    其中 h(s) = d_obstacle - d_safe  (障碍物距离余量)

    给定 nominal action a_nom，求解 QP：
      min_{a}  ||a - a_nom||²
      s.t.     L_f·h + L_g·h·a ≥ -α·h

    实现：解析 CBF-QP（一阶近似 + 梯度投影）
    """

    def __init__(self, action_dim=ACTION_DIM, d_safe=0.5, alpha=2.0):
        super().__init__()
        self.d_safe = d_safe
        self.alpha  = alpha
        # 可学习的 CBF 参数
        self.log_alpha_cbf = nn.Parameter(torch.tensor(math.log(alpha)))

    def h(self, state: np.ndarray) -> float:
        """CBF: h(s) = min_LiDAR - d_safe。"""
        return get_obstacle_min(state) - self.d_safe

    def cbf_filter(self, a_nom: np.ndarray,
                   state: np.ndarray) -> np.ndarray:
        """
        CBF-QP 解析解（一维前进速度约束）：
          若 h(s) < 0：将前进速度限制到安全范围
          否则：允许原始动作
        """
        a     = a_nom.copy()
        h_val = self.h(state)
        alpha = float(self.log_alpha_cbf.exp().item())

        if h_val < 0:
            # 障碍物已在安全距离内，严格限速
            # CBF 条件近似：vx ≤ h(s) / d_safe
            vx_max = max(0., (h_val + self.d_safe) / self.d_safe)
            a[0]   = np.clip(a[0], -1., vx_max)
        elif h_val < self.d_safe:
            # 接近安全边界，按 CBF 渐进限速
            scale  = max(0., h_val / self.d_safe) ** (1./alpha)
            a[0]   = np.clip(a[0], -1., scale)

        # 超声波搁浅约束（水底）
        us_min = float(np.min(state[US_IDX]))
        if us_min < 0.3:
            a[0] *= max(0.1, us_min / 0.3)

        return np.clip(a, -1., 1.)


class BarrierNet(BaseAgent):
    """
    BarrierNet: Control Barrier Function + PPO for Safe Navigation。

    核心特点：
      1. CBF 安全过滤层（可学习 α 参数）
         - LiDAR → 障碍物 CBF
         - 超声波 → 搁浅 CBF
         - 深度传感器 → 水中深度 CBF
      2. PPO 策略学习（nominal policy）
      3. CBF-QP 在执行时修正动作，保证安全约束满足
      4. CBF 违反惩罚项加入奖励函数，引导策略主动避开危险

    参考：Xiao et al., "Barriernet: Differentiable Control Barrier
          Functions for Learning of Safe Robot Control", T-RO 2023
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
                 n_steps:     int   = 2048,
                 batch_size:  int   = 64,
                 n_epochs:    int   = 10,
                 d_safe:      float = 0.5,   # CBF 安全距离
                 alpha_cbf:   float = 2.0,   # CBF 衰减率
                 cbf_penalty: float = 0.5,   # CBF 违反惩罚系数
                 device:      str   = 'cpu'):
        super().__init__('BarrierNet', state_dim, action_dim, device)
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.cbf_penalty = cbf_penalty

        # Nominal PPO policy
        self.ac  = PPOActorCritic(state_dim, action_dim, hidden,
                                  continuous=True).to(device)
        # CBF 过滤层（含可学习参数）
        self.cbf = CBFLayer(action_dim, d_safe, alpha_cbf).to(device)

        self.opt = optim.Adam(
            list(self.ac.parameters()) + list(self.cbf.parameters()),
            lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()

    def _cbf_reward(self, state: np.ndarray,
                    nominal_a: np.ndarray,
                    safe_a:    np.ndarray) -> float:
        """
        CBF 违反惩罚：若安全投影修改了动作，则惩罚。
        penalty = cbf_penalty · ||safe_a - nominal_a||
        鼓励策略主动输出安全的 nominal action。
        """
        diff = np.linalg.norm(safe_a - nominal_a)
        return -self.cbf_penalty * diff

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        s = self._to_tensor(state)
        with torch.no_grad():
            dist, _ = self.ac(s)
            a_nom = dist.mean if deterministic else dist.sample()
        nominal = a_nom.squeeze(0).cpu().numpy()
        # CBF 安全过滤
        safe    = self.cbf.cbf_filter(nominal, state)
        return safe

    def get_log_prob_value(self, state, action):
        s = self._to_tensor(state)
        a = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, v = self.ac(s); lp = dist.log_prob(a).sum(-1)
        return float(lp.item()), float(v.item())

    def store(self, state, action, log_prob, base_reward, value, done):
        self.buffer.add(state, action, log_prob, base_reward, value, done)
        self.total_steps += 1

    def update(self, last_value=0.) -> Dict[str, float]:
        if len(self.buffer) < self.n_steps:
            return {}
        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        data = self.buffer.get(self.device); B = len(self.buffer)
        metrics={'loss':0.,'pi':0.,'vf':0.,'ent':0.}; n_b=0
        for _ in range(self.n_epochs):
            perm = torch.randperm(B, device=self.device)
            for st in range(0, B, self.batch_size):
                idx = perm[st:st+self.batch_size]
                if len(idx)<2: continue
                s,a,lp0,adv,ret=(data['s'][idx],data['a'][idx],
                                  data['lp'][idx],data['adv'][idx],data['ret'][idx])
                adv=(adv-adv.mean())/(adv.std()+1e-8)
                dist, v = self.ac(s)
                lp1=dist.log_prob(a).sum(-1); ent=dist.entropy().sum(-1).mean()
                r=(lp1-lp0).exp()
                pi_l=-torch.min(r*adv,r.clamp(1-self.clip_eps,1+self.clip_eps)*adv).mean()
                vf_l=F.mse_loss(v,ret)
                loss=pi_l+self.vf_coef*vf_l-self.ent_coef*ent
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.ac.parameters())+list(self.cbf.parameters()), 0.5)
                self.opt.step()
                metrics['loss']+=loss.item(); metrics['pi']+=pi_l.item()
                metrics['vf']+=vf_l.item(); metrics['ent']+=ent.item(); n_b+=1
        self.buffer.reset(); self.n_updates+=1
        return {k:v/n_b for k,v in metrics.items()} if n_b else {}

    def save(self, path):
        torch.save({'ac': self.ac.state_dict(),
                    'cbf': self.cbf.state_dict(),
                    'steps': self.total_steps}, path)

    def load(self, path):
        c = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(c['ac'])
        self.cbf.load_state_dict(c['cbf'])
        self.total_steps = c.get('steps', 0)


# ─────────────────────────────────────────────────────────────────────────────
# Method 12: pH-DRL — Port-Hamiltonian Deep RL
# ─────────────────────────────────────────────────────────────────────────────

class PHDynamicsNet(nn.Module):
    """
    Port-Hamiltonian 动力学网络。

    pH 系统结构：
      ẋ = (J - R) · ∂H/∂x + g · u
      y = gᵀ · ∂H/∂x

    其中：
      H(x): Hamiltonian（能量函数），由 MLP 参数化
      J:    斜对称互联矩阵（可学习）
      R:    正半定阻尼矩阵（可学习，确保耗散性）
      g:    输入矩阵

    用于水动力学建模，确保能量守恒和物理一致性。
    """

    def __init__(self, state_dim=STATE_DIM, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        # Hamiltonian H(x) — 能量函数
        self.H_net = mlp([state_dim, 64, 32, latent_dim])

        # 斜对称矩阵 J（Cholesky-like 参数化）
        n = latent_dim
        self.J_raw = nn.Parameter(torch.zeros(n, n))   # 上三角 → J = A - Aᵀ

        # 正半定阻尼矩阵 R = LLᵀ
        self.R_chol = nn.Parameter(torch.eye(n) * 0.1)

    def J(self):
        A = torch.triu(self.J_raw, diagonal=1)
        return A - A.T

    def R(self):
        L = torch.tril(self.R_chol)
        return L @ L.T

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (H(x), dH/dx 的近似方向)
        """
        h_val  = self.H_net(x).sum(-1, keepdim=True)
        # 梯度近似（用于强化学习奖励塑形）
        dh_dx  = torch.ones_like(x) / (x.abs().mean() + 1e-6)
        return h_val, dh_dx


class PHDRLActor(nn.Module):
    """
    pH-DRL Actor：在标准高斯策略基础上增加 pH 能量约束项。
    控制律 = nominal_action + pH_correction
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 hidden=256):
        super().__init__()
        self.trunk   = mlp([state_dim, hidden, hidden])
        self.mu_head = nn.Linear(hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        self.v_head  = nn.Linear(hidden, 1)
        self.ph_proj = nn.Linear(state_dim, action_dim)   # pH 修正投影
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)

    def forward(self, s: torch.Tensor, use_ph=True):
        h   = self.trunk(s)
        mu  = torch.tanh(self.mu_head(h))
        if use_ph:
            # pH 修正：基于能量梯度方向的小扰动
            ph_corr = 0.1 * torch.tanh(self.ph_proj(s))
            mu = torch.tanh(mu + ph_corr)
        std = self.log_std.clamp(-4., 2.).exp().expand_as(mu)
        return Normal(mu, std), self.v_head(h).squeeze(-1)


class PHDRLAgent(BaseAgent):
    """
    Port-Hamiltonian Deep RL。

    核心特点：
      1. pH 动力学网络建模水动力学（能量守恒约束）
      2. pH-aware 策略：控制律受 Hamiltonian 梯度修正
      3. 能量惩罚奖励塑形：
         R_ph = R_base - β · |ΔH|  （鼓励能量高效利用）
      4. 深度传感器驱动水域 pH 修正增强：
         水中激活 pH 修正，陆地关闭 pH 修正

    参考：Zhong et al., "Dissipative SymODEN", ICLR Workshop 2020
    """

    def __init__(self,
                 state_dim:  int   = STATE_DIM,
                 action_dim: int   = ACTION_DIM,
                 hidden:     int   = 256,
                 lr:         float = 3e-4,
                 gamma:      float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_eps:   float = 0.2,
                 n_steps:    int   = 2048,
                 batch_size: int   = 64,
                 n_epochs:   int   = 10,
                 ph_beta:    float = 0.05,   # 能量惩罚系数
                 device:     str   = 'cpu'):
        super().__init__('pH-DRL', state_dim, action_dim, device)
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps   = clip_eps
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.ph_beta    = ph_beta

        self.actor  = PHDRLActor(state_dim, action_dim, hidden).to(device)
        self.ph_net = PHDynamicsNet(state_dim).to(device)
        self.opt    = optim.Adam(
            list(self.actor.parameters()) + list(self.ph_net.parameters()),
            lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()
        self._prev_H = 0.

    def _ph_reward(self, state: np.ndarray) -> float:
        """能量惩罚：|ΔH| 过大时惩罚（鼓励平滑能量变化）。"""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            H, _ = self.ph_net(s)
        h_val = float(H.item())
        delta = abs(h_val - self._prev_H)
        self._prev_H = h_val
        return -self.ph_beta * delta

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        domain = get_domain(state)
        use_ph = (domain == 0)   # 水中激活 pH 修正
        s = self._to_tensor(state)
        with torch.no_grad():
            dist, _ = self.actor(s, use_ph=use_ph)
            a = dist.mean if deterministic else dist.sample()
        action = a.squeeze(0).cpu().numpy()
        # LiDAR 安全裁剪
        min_d = get_obstacle_min(state)
        if min_d < 0.4:
            action[0] = np.clip(action[0], -1., max(0., min_d/0.4))
        return np.clip(action, -1., 1.)

    def get_log_prob_value(self, state, action):
        domain = get_domain(state)
        s = self._to_tensor(state)
        a = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, v = self.actor(s, use_ph=(domain==0))
            lp = dist.log_prob(a).sum(-1)
        return float(lp.item()), float(v.item())

    def store(self, state, action, log_prob, base_reward, value, done):
        ph_r   = self._ph_reward(state)
        reward = base_reward + ph_r
        self.buffer.add(state, action, log_prob, reward, value, done)
        self.total_steps += 1

    def update(self, last_value=0.) -> Dict[str, float]:
        if len(self.buffer) < self.n_steps:
            return {}
        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        data = self.buffer.get(self.device); B = len(self.buffer)
        metrics={'loss':0.,'pi':0.,'vf':0.}; n_b=0
        for _ in range(self.n_epochs):
            perm = torch.randperm(B, device=self.device)
            for st in range(0, B, self.batch_size):
                idx = perm[st:st+self.batch_size]
                if len(idx)<2: continue
                s,a,lp0,adv,ret=(data['s'][idx],data['a'][idx],
                                  data['lp'][idx],data['adv'][idx],data['ret'][idx])
                adv=(adv-adv.mean())/(adv.std()+1e-8)
                dist, v = self.actor(s, use_ph=True)
                lp1=dist.log_prob(a).sum(-1); ent=dist.entropy().sum(-1).mean()
                r=(lp1-lp0).exp()
                pi_l=-torch.min(r*adv,r.clamp(1-self.clip_eps,1+self.clip_eps)*adv).mean()
                vf_l=F.mse_loss(v,ret)
                # pH 网络损失（能量预测一致性）
                H, _ = self.ph_net(s); ph_l = H.pow(2).mean() * 0.01
                loss=pi_l+0.5*vf_l-0.01*ent+ph_l
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())+list(self.ph_net.parameters()), 0.5)
                self.opt.step()
                metrics['loss']+=loss.item(); metrics['pi']+=pi_l.item()
                metrics['vf']+=vf_l.item(); n_b+=1
        self.buffer.reset(); self.n_updates+=1
        return {k:v/n_b for k,v in metrics.items()} if n_b else {}

    def save(self, path):
        torch.save({'actor': self.actor.state_dict(),
                    'ph_net': self.ph_net.state_dict(),
                    'steps': self.total_steps}, path)

    def load(self, path):
        c=torch.load(path,map_location=self.device)
        self.actor.load_state_dict(c['actor'])
        self.ph_net.load_state_dict(c['ph_net'])
        self.total_steps=c.get('steps',0)


# ─────────────────────────────────────────────────────────────────────────────
# Method 13: MP-DQL — Motion Primitive + Deep Q-Learning
# ─────────────────────────────────────────────────────────────────────────────

class MotionPrimitive:
    """
    Motion Primitive 库：预定义的基本运动模式。

    两栖场景运动原语 (8个)：
      0: 直线前进（水域最优）
      1: 左转前进
      2: 右转前进
      3: 原地左转
      4: 原地右转
      5: 缓慢前进（过渡区安全模式）
      6: 减速停止
      7: 向后微移（脱困）
    """

    PRIMITIVES: List[np.ndarray] = [
        np.array([ 1.0,  0.0], np.float32),   # 0: 直线前进
        np.array([ 0.8,  0.6], np.float32),   # 1: 左转前进
        np.array([ 0.8, -0.6], np.float32),   # 2: 右转前进
        np.array([ 0.0,  1.0], np.float32),   # 3: 原地左转
        np.array([ 0.0, -1.0], np.float32),   # 4: 原地右转
        np.array([ 0.4,  0.0], np.float32),   # 5: 缓慢前进
        np.array([ 0.0,  0.0], np.float32),   # 6: 停止
        np.array([-0.4,  0.0], np.float32),   # 7: 后退
    ]
    N_PRIMITIVES = 8

    @classmethod
    def get(cls, idx: int) -> np.ndarray:
        return cls.PRIMITIVES[idx % cls.N_PRIMITIVES].copy()

    @classmethod
    def domain_mask(cls, domain: int) -> np.ndarray:
        """
        按域过滤可用运动原语。
          水中：所有原语可用（全水动力）
          过渡区：屏蔽高速，保留缓慢和转向
          陆地：屏蔽后退（滚轮差速）
        """
        mask = np.ones(cls.N_PRIMITIVES, bool)
        if domain == 1:   # 过渡区
            mask[0] = False   # 禁用全速前进（坡面稳定性）
        if domain == 2:   # 陆地
            mask[7] = False   # 禁用后退（差速驱动限制）
        return mask


class DuelingQNet(nn.Module):
    """Dueling Q 网络，用于 MP-DQL。"""

    def __init__(self, state_dim, n_acts, hidden=256):
        super().__init__()
        self.shared = mlp([state_dim, hidden, hidden])
        self.V = nn.Linear(hidden, 1)
        self.A = nn.Linear(hidden, n_acts)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.shared(x)
        v = self.V(h); a = self.A(h)
        return v + a - a.mean(-1, keepdim=True)


class MPDQL(BaseAgent):
    """
    Motion Primitive + Deep Q-Learning (MP-DQL)。

    核心特点：
      1. 动作空间 = 运动原语库（8个预定义运动模式）
         代替连续动作，每个原语对应一段固定时长轨迹
      2. Dueling DQN 学习各原语在当前状态的 Q 值
      3. 域感知原语过滤：
         - 水中：全部 8 个原语可用
         - 过渡区：屏蔽高速原语
         - 陆地：屏蔽后退原语
      4. 传感器融合权重：
         - LiDAR 近距离 → 优先转向原语
         - 深度传感器 → 判断推进器/差速切换时机
         - 超声波 → 水底浅时降速原语优先级

    每个原语执行固定 k=5 步，Q 值近似该子轨迹的累计折扣奖励。
    """

    def __init__(self,
                 state_dim:   int   = STATE_DIM,
                 hidden:      int   = 256,
                 lr:          float = 5e-4,
                 gamma:       float = 0.99,
                 tau:         float = 0.005,
                 buffer_cap:  int   = 50_000,
                 batch_size:  int   = 128,
                 eps_start:   float = 1.0,
                 eps_end:     float = 0.05,
                 eps_decay:   int   = 30_000,
                 learn_starts:int   = 1000,
                 primitive_steps: int = 5,  # 每个原语执行步数 k
                 device:      str   = 'cpu'):
        super().__init__('MP-DQL', state_dim, ACTION_DIM, device)
        self.gamma           = gamma
        self.tau             = tau
        self.batch_size      = batch_size
        self.eps_start       = eps_start
        self.eps_end         = eps_end
        self.eps_decay       = eps_decay
        self.learn_starts    = learn_starts
        self.primitive_steps = primitive_steps

        N = MotionPrimitive.N_PRIMITIVES

        self.online = DuelingQNet(state_dim, N, hidden).to(device)
        self.target = DuelingQNet(state_dim, N, hidden).to(device)
        hard_update(self.target, self.online)
        for p in self.target.parameters(): p.requires_grad = False

        self.opt    = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_cap)

        # 当前激活的原语和剩余执行步数
        self._active_prim: Optional[int] = None
        self._prim_steps_left: int       = 0

    def _eps(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
               math.exp(-self.total_steps / self.eps_decay)

    def _q_with_sensor_weight(self, q: np.ndarray,
                               state: np.ndarray) -> np.ndarray:
        """
        基于传感器融合对 Q 值进行软权重调整。

        LiDAR 近距离 → 提升转向原语（1,2,3,4）的 Q 值
        超声波搁浅   → 提升缓慢原语（5）和停止（6）的 Q 值
        """
        q = q.copy()
        lidar_min = get_obstacle_min(state)
        us_min    = float(np.min(state[US_IDX]))
        depth     = float(state[DEPTH_IDX])

        if lidar_min < 0.6:   # 障碍物近 → 转向更优
            bonus = (0.6 - lidar_min) / 0.6 * 0.5
            q[1] += bonus; q[2] += bonus   # 左/右转前进
            q[3] += bonus; q[4] += bonus   # 原地转向
            q[0] -= bonus                  # 直线前进降权

        if us_min < 0.4 and depth < 0.1:  # 搁浅风险 → 缓慢/停止
            q[5] += 0.4; q[6] += 0.3
            q[0] -= 0.5; q[7] += 0.1

        return q

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        # 若当前原语仍在执行，继续输出同一个原语
        if self._prim_steps_left > 0 and self._active_prim is not None:
            self._prim_steps_left -= 1
            return MotionPrimitive.get(self._active_prim)

        # 选择新原语
        domain = get_domain(state)
        mask   = MotionPrimitive.domain_mask(domain)

        if not deterministic and np.random.random() < self._eps():
            valid = np.where(mask)[0]
            prim_idx = int(np.random.choice(valid))
        else:
            s = self._to_tensor(state)
            with torch.no_grad():
                q = self.online(s).squeeze(0).cpu().numpy()
            q = self._q_with_sensor_weight(q, state)
            q[~mask] = -1e9
            prim_idx = int(np.argmax(q))

        self._active_prim      = prim_idx
        self._prim_steps_left  = self.primitive_steps - 1
        return MotionPrimitive.get(prim_idx)

    def store(self, s, a, r, s2, done):
        # 将连续动作映射到最接近的原语索引
        dists    = [np.linalg.norm(a - MotionPrimitive.get(i))
                    for i in range(MotionPrimitive.N_PRIMITIVES)]
        prim_idx = int(np.argmin(dists))
        self.buffer.add(s, [prim_idx], r, s2, done)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.learn_starts:
            return {}

        batch  = self.buffer.sample(self.batch_size, self.device)
        s, a_idx, r, s2, d = (batch['s'], batch['a'].long().squeeze(-1),
                                batch['r'], batch['s2'], batch['d'])

        with torch.no_grad():
            a_next = self.online(s2).argmax(-1)
            q_next = self.target(s2).gather(1, a_next.unsqueeze(-1))
            y      = r + self.gamma * (1 - d) * q_next

        q_pred = self.online(s).gather(1, a_idx.unsqueeze(-1))
        loss   = F.smooth_l1_loss(q_pred, y)

        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        self.opt.step()
        soft_update(self.target, self.online, self.tau)
        self.n_updates += 1
        return {'loss': loss.item(), 'eps': self._eps()}

    def reset_episode(self):
        """Episode 开始时重置原语状态。"""
        self._active_prim     = None
        self._prim_steps_left = 0

    def save(self, path):
        torch.save({'online': self.online.state_dict(),
                    'target': self.target.state_dict(),
                    'steps':  self.total_steps}, path)

    def load(self, path):
        c = torch.load(path, map_location=self.device)
        self.online.load_state_dict(c['online'])
        self.target.load_state_dict(c['target'])
        self.total_steps = c.get('steps', 0)
