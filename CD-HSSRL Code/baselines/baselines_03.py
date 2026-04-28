"""
baselines_03.py
─────────────────────────────────────────────────────────────
Method 7:  MORL-based    — Multi-Objective RL
Method 8:  RLCA          — RL for Collision Avoidance
Method 9:  APF-D3QNPER   — APF + Double Dueling DQN + PER
Method 10: CLPPO-GIC     — Constrained PPO + Goal Inference
─────────────────────────────────────────────────────────────
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple
import math

from baseline_base import (
    BaseAgent, ReplayBuffer, RolloutBuffer,
    PrioritizedReplayBuffer,
    PPOActorCritic, DeterministicActor, QNetwork, ValueNet,
    mlp, soft_update, hard_update,
    STATE_DIM, ACTION_DIM,
    get_obstacle_min, get_water_depth, get_domain,
    LIDAR_IDX, US_IDX, DEPTH_IDX,
)


# ─────────────────────────────────────────────────────────────────────────────
# Method 7: MORL-based — Multi-Objective Reinforcement Learning
# ─────────────────────────────────────────────────────────────────────────────

class MORLBased(BaseAgent):
    """
    Multi-Objective RL (MORL) for amphibious navigation。

    目标向量 f = [f_nav, f_safe, f_energy, f_smooth]：
      f_nav    = -d_goal / d_init               (导航效率)
      f_safe   = -P(collision | LiDAR)          (安全性)
      f_energy = -||a||² / 2                    (能耗)
      f_smooth = -||a_t - a_{t-1}||²            (平滑性)

    权重向量 w（策略条件化）：
      w_nav=0.6, w_safe=0.25, w_energy=0.1, w_smooth=0.05

    方法：线性标量化 MORL + PPO 策略更新
    权重 w 拼接到状态中，允许在线调整优先级。
    """

    def __init__(self,
                 state_dim:    int   = STATE_DIM,
                 action_dim:   int   = ACTION_DIM,
                 hidden:       int   = 256,
                 lr:           float = 3e-4,
                 gamma:        float = 0.99,
                 gae_lambda:   float = 0.95,
                 clip_eps:     float = 0.2,
                 n_steps:      int   = 2048,
                 batch_size:   int   = 64,
                 n_epochs:     int   = 10,
                 w_nav:        float = 0.60,
                 w_safe:       float = 0.25,
                 w_energy:     float = 0.10,
                 w_smooth:     float = 0.05,
                 device:       str   = 'cpu'):
        super().__init__('MORL-based', state_dim, action_dim, device)
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps   = clip_eps
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.n_epochs   = n_epochs

        # 目标权重向量（拼接到状态）
        self.w = np.array([w_nav, w_safe, w_energy, w_smooth], np.float32)

        # PPO AC（权重向量维度 +4）
        aug_dim = state_dim + 4
        self.ac  = PPOActorCritic(aug_dim, action_dim, hidden,
                                  continuous=True).to(device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)
        self.buffer  = RolloutBuffer()
        self._prev_a = np.zeros(action_dim, np.float32)
        self._d_init = 24.0   # 初始到目标距离估计

    def _augment(self, state: np.ndarray) -> np.ndarray:
        return np.concatenate([state, self.w])

    def _mo_reward(self, base_reward, state, action, d_goal) -> float:
        """计算多目标标量化奖励。"""
        f_nav    = -d_goal / max(self._d_init, 1.)
        d_obs    = get_obstacle_min(state)
        risk     = 1. / (1. + math.exp(8. * (d_obs - 0.4)))
        f_safe   = -risk
        f_energy = -float(np.sum(action**2)) / 2.
        f_smooth = -float(np.sum((action - self._prev_a)**2))
        r = (self.w[0] * f_nav + self.w[1] * f_safe +
             self.w[2] * f_energy + self.w[3] * f_smooth)
        # 保留到达/碰撞的大奖励
        r += base_reward * 0.1
        self._prev_a = action.copy()
        return float(r)

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        aug = self._augment(state)
        s   = torch.FloatTensor(aug).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, _ = self.ac(s)
            a = dist.mean if deterministic else dist.sample()
        action = a.squeeze(0).cpu().numpy()
        # LiDAR 安全裁剪
        min_d = get_obstacle_min(state)
        if min_d < 0.4:
            action[0] = np.clip(action[0], -1., max(0., min_d/0.4))
        return np.clip(action, -1., 1.)

    def get_log_prob_value(self, state, action):
        aug = self._augment(state)
        s   = torch.FloatTensor(aug).unsqueeze(0).to(self.device)
        a   = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, v = self.ac(s)
            lp = dist.log_prob(a).sum(-1)
        return float(lp.item()), float(v.item())

    def store(self, state, action, log_prob, base_reward, value, done,
              d_goal=0.):
        reward = self._mo_reward(base_reward, state, action, d_goal)
        self.buffer.add(self._augment(state), action, log_prob,
                        reward, value, done)
        self.total_steps += 1

    def update(self, last_value=0.) -> Dict[str, float]:
        if len(self.buffer) < self.n_steps:
            return {}
        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        data = self.buffer.get(self.device)
        B    = len(self.buffer)
        metrics = {'loss':0.,'pi':0.,'vf':0.}; n_b = 0
        for _ in range(self.n_epochs):
            perm = torch.randperm(B, device=self.device)
            for st in range(0, B, self.batch_size):
                idx = perm[st:st+self.batch_size]
                if len(idx) < 2: continue
                s,a,lp0,adv,ret = (data['s'][idx], data['a'][idx],
                                    data['lp'][idx], data['adv'][idx], data['ret'][idx])
                adv = (adv-adv.mean())/(adv.std()+1e-8)
                dist, v = self.ac(s)
                lp1 = dist.log_prob(a).sum(-1)
                ent = dist.entropy().sum(-1).mean()
                r   = (lp1-lp0).exp()
                pi_l= -torch.min(r*adv, r.clamp(1-self.clip_eps,1+self.clip_eps)*adv).mean()
                vf_l= F.mse_loss(v, ret)
                loss= pi_l + 0.5*vf_l - 0.01*ent
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()
                metrics['loss']+=loss.item(); metrics['pi']+=pi_l.item()
                metrics['vf']+=vf_l.item(); n_b+=1
        self.buffer.reset(); self.n_updates+=1
        return {k:v/n_b for k,v in metrics.items()} if n_b else {}

    def save(self, path):
        torch.save({'ac': self.ac.state_dict(), 'steps': self.total_steps}, path)

    def load(self, path):
        c = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(c['ac']); self.total_steps = c.get('steps',0)


# ─────────────────────────────────────────────────────────────────────────────
# Method 8: RLCA — Reinforcement Learning for Collision Avoidance
# ─────────────────────────────────────────────────────────────────────────────

class RLCA(BaseAgent):
    """
    RLCA — RL-based Collision Avoidance for Mobile Robots。

    核心特点：
      1. 速度障碍物约束（Velocity Obstacle）：
         基于 LiDAR 8扇区计算每个方向的安全速度上限
      2. 社会力模型增强奖励：
         R = R_goal - λ·∑ exp(-d_i / σ)  (障碍物排斥项)
      3. PPO + 安全动作投影层（解析 QP）
      4. 超声波搁浅检测：过渡区降低速度

    参考：Chen et al., "Decentralized Non-communicating Multiagent
          Collision Avoidance with Deep Reinforcement Learning", ICRA 2017
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
                 lambda_rep: float = 0.8,   # 社会力排斥系数
                 sigma_rep:  float = 0.5,   # 排斥衰减系数
                 device:     str   = 'cpu'):
        super().__init__('RLCA', state_dim, action_dim, device)
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps   = clip_eps
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.lambda_rep = lambda_rep
        self.sigma_rep  = sigma_rep

        self.ac  = PPOActorCritic(state_dim, action_dim, hidden,
                                  continuous=True).to(device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()

    def _velocity_obstacle_limit(self, state: np.ndarray) -> float:
        """
        基于 LiDAR 计算安全前进速度上限（速度障碍物方法简化版）。
        vx_max = min_lidar / d_safe_ref，线性缩放。
        """
        lidar_min = get_obstacle_min(state)
        d_ref     = 2.0   # 参考安全距离
        vx_max    = min(1.0, lidar_min / d_ref)
        return max(0.0, vx_max)

    def _social_reward(self, state: np.ndarray) -> float:
        """
        社会力排斥项：-λ·∑ exp(-d_i / σ)
        基于 LiDAR 8扇区距离。
        """
        lidar = state[LIDAR_IDX]
        rep   = -self.lambda_rep * np.sum(np.exp(-lidar / self.sigma_rep))
        return float(rep)

    def _grounding_check(self, state: np.ndarray, action: np.ndarray
                         ) -> np.ndarray:
        """超声波搁浅检测：水底过近时限制速度。"""
        a     = action.copy()
        us    = state[US_IDX]
        us_min = float(np.min(us))
        depth = float(state[DEPTH_IDX])
        if us_min < 0.3 and depth < 0.1:   # 搁浅风险
            a[0] *= 0.3
        return a

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        s = self._to_tensor(state)
        with torch.no_grad():
            dist, _ = self.ac(s)
            a = dist.mean if deterministic else dist.sample()
        action = a.squeeze(0).cpu().numpy()
        # 速度障碍物限速
        vx_max = self._velocity_obstacle_limit(state)
        action[0] = np.clip(action[0], -1., vx_max)
        # 超声波搁浅检测
        action = self._grounding_check(state, action)
        return np.clip(action, -1., 1.)

    def get_log_prob_value(self, state, action):
        s = self._to_tensor(state)
        a = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, v = self.ac(s)
            lp = dist.log_prob(a).sum(-1)
        return float(lp.item()), float(v.item())

    def store(self, state, action, log_prob, base_reward, value, done):
        reward = base_reward + self._social_reward(state)
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
                dist, v = self.ac(s)
                lp1=dist.log_prob(a).sum(-1); ent=dist.entropy().sum(-1).mean()
                r=(lp1-lp0).exp()
                pi_l=-torch.min(r*adv,r.clamp(1-self.clip_eps,1+self.clip_eps)*adv).mean()
                vf_l=F.mse_loss(v,ret)
                loss=pi_l+0.5*vf_l-0.01*ent
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()
                metrics['loss']+=loss.item(); metrics['pi']+=pi_l.item()
                metrics['vf']+=vf_l.item(); n_b+=1
        self.buffer.reset(); self.n_updates+=1
        return {k:v/n_b for k,v in metrics.items()} if n_b else {}

    def save(self, path):
        torch.save({'ac': self.ac.state_dict(), 'steps': self.total_steps}, path)

    def load(self, path):
        c = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(c['ac']); self.total_steps=c.get('steps',0)


# ─────────────────────────────────────────────────────────────────────────────
# Method 9: APF-D3QNPER — APF + Double Dueling DQN + PER
# ─────────────────────────────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """Dueling 网络结构：Q(s,a) = V(s) + A(s,a) - mean(A)。"""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.shared = mlp([state_dim, hidden, hidden])
        self.value  = nn.Linear(hidden, 1)
        self.adv    = nn.Linear(hidden, n_actions)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h   = self.shared(x)
        v   = self.value(h)
        a   = self.adv(h)
        return v + a - a.mean(-1, keepdim=True)


class APFD3QNPER(BaseAgent):
    """
    APF + Double Dueling DQN + Prioritized Experience Replay。

    综合三项技术：
      1. APF：势场力引导初始探索方向（同 APF-DQN）
      2. Double DQN：online 选 action，target 评估 Q（减少过估计）
      3. Dueling：V(s) + A(s,a) 分离，提升价值估计稳定性
      4. PER (α=0.6, β=0.4→1.0 退火)：优先回放高 TD-error 经验

    动作空间：9 个离散动作（同 APF-DQN）
    """

    VX_BINS = np.array([-0.8, 0., 0.8], np.float32)
    WZ_BINS = np.array([-0.8, 0., 0.8], np.float32)
    N_ACTS  = 9
    XI, ETA, D0 = 1.0, 2.0, 1.5    # APF 参数

    def __init__(self,
                 state_dim:   int   = STATE_DIM,
                 hidden:      int   = 256,
                 lr:          float = 5e-4,
                 gamma:       float = 0.99,
                 tau:         float = 0.005,
                 buffer_cap:  int   = 100_000,
                 batch_size:  int   = 128,
                 per_alpha:   float = 0.6,
                 per_beta:    float = 0.4,
                 per_beta_inc:float = 2e-6,
                 eps_start:   float = 1.0,
                 eps_end:     float = 0.05,
                 eps_decay:   int   = 50_000,
                 learn_starts:int   = 1000,
                 device:      str   = 'cpu'):
        super().__init__('APF-D3QNPER', state_dim, ACTION_DIM, device)
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.per_beta     = per_beta
        self.per_beta_inc = per_beta_inc
        self.eps_start    = eps_start
        self.eps_end      = eps_end
        self.eps_decay    = eps_decay
        self.learn_starts = learn_starts

        aug_dim = state_dim + 4   # + APF force
        self.online = DuelingDQN(aug_dim, self.N_ACTS, hidden).to(device)
        self.target = DuelingDQN(aug_dim, self.N_ACTS, hidden).to(device)
        hard_update(self.target, self.online)
        for p in self.target.parameters(): p.requires_grad = False

        self.opt    = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = PrioritizedReplayBuffer(buffer_cap, per_alpha)

        self._action_table = np.array([
            [vx, wz] for vx in self.VX_BINS for wz in self.WZ_BINS
        ], np.float32)

    def _apf_force(self, state: np.ndarray, goal_dir: float = 0.) -> np.ndarray:
        d_goal = float(state[10]); yaw = float(state[9])
        f_att_x = np.clip(self.XI * d_goal * math.cos(goal_dir - yaw), -2., 2.)
        f_att_y = np.clip(self.XI * d_goal * math.sin(goal_dir - yaw), -2., 2.)
        f_rep_x = f_rep_y = 0.
        for i, d in enumerate(state[LIDAR_IDX]):
            if d < self.D0:
                ang = yaw + (-math.pi + i * math.pi/4)
                m   = self.ETA * (1./max(d,0.1) - 1./self.D0) / (d**2)
                f_rep_x -= m * math.cos(ang)
                f_rep_y -= m * math.sin(ang)
        return np.array([f_att_x, f_att_y,
                         np.clip(f_rep_x,-3.,3.),
                         np.clip(f_rep_y,-3.,3.)], np.float32)

    def _augment(self, state, goal_dir=0.):
        return np.concatenate([state, self._apf_force(state, goal_dir)])

    def _eps(self):
        return self.eps_end + (self.eps_start - self.eps_end) * \
               math.exp(-self.total_steps / self.eps_decay)

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False,
                      goal_dir: float = 0.) -> np.ndarray:
        aug = self._augment(state, goal_dir)
        if not deterministic and np.random.random() < self._eps():
            idx = np.random.randint(self.N_ACTS)
        else:
            s = torch.FloatTensor(aug).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q = self.online(s).squeeze(0).cpu().numpy()
            idx = int(np.argmax(q))
        return self._action_table[idx]

    def store(self, s, a, r, s2, done, goal_dir=0.):
        aug_s  = self._augment(s,  goal_dir)
        aug_s2 = self._augment(s2, goal_dir)
        dists  = np.linalg.norm(self._action_table - np.array(a), axis=1)
        idx    = int(np.argmin(dists))
        self.buffer.add(aug_s, [idx], r, aug_s2, done)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.learn_starts:
            return {}
        self.per_beta = min(1.0, self.per_beta + self.per_beta_inc)
        b = self.buffer.sample(self.batch_size, self.per_beta, self.device)
        s, a_idx, r, s2, d, w = (b['s'], b['a'].long(),
                                   b['r'].unsqueeze(1), b['s2'],
                                   b['d'].unsqueeze(1), b['weights'].unsqueeze(1))

        with torch.no_grad():
            a_next = self.online(s2).argmax(-1)
            q_next = self.target(s2).gather(1, a_next.unsqueeze(-1))
            y      = r + self.gamma * (1-d) * q_next

        q_pred = self.online(s).gather(1, a_idx)
        td_err = (q_pred - y).abs().detach().cpu().numpy().flatten()
        self.buffer.update_priorities(b['idx'], td_err)

        loss = (w * F.smooth_l1_loss(q_pred, y, reduction='none')).mean()
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        self.opt.step()
        soft_update(self.target, self.online, self.tau)
        self.n_updates += 1
        return {'loss': loss.item(), 'beta': self.per_beta, 'eps': self._eps()}

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
# Method 10: CLPPO-GIC — Constrained PPO + Goal Inference Curriculum
# ─────────────────────────────────────────────────────────────────────────────

class GoalInferenceCurriculum:
    """
    Goal Inference Curriculum (GIC) 模块。

    课程学习策略：
      - 初期：目标距离短，障碍物稀疏（简单任务）
      - 后期：目标距离长，障碍物密集（困难任务）
      - 难度由 progress（训练进度）驱动

    目标状态预测：基于当前状态预测下一个子目标。
    """

    def __init__(self, state_dim=STATE_DIM, hidden=64):
        # 目标推断网络：s_t → Δ(goal)
        self.goal_net = mlp([state_dim, hidden, hidden, 2])  # 预测 [Δx, Δy]

    def infer_subgoal(self, state: np.ndarray) -> np.ndarray:
        """基于当前状态推断下一子目标方向。"""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            delta = self.goal_net(s).squeeze(0).cpu().numpy()
        return np.clip(delta, -5., 5.)

    def curriculum_difficulty(self, progress: float) -> float:
        """0→easy，1→hard。sigmoid 课程。"""
        return 1. / (1. + math.exp(-10. * (progress - 0.5)))


class CLPPOGIC(BaseAgent):
    """
    Constrained PPO + Goal Inference Curriculum。

    核心特点：
      1. 约束 PPO（CPO 变体）：
         - 约束：期望碰撞率 E[cost] ≤ δ_c = 0.1
         - Lagrangian 松弛：L = L_PPO + λ·(E[cost] - δ_c)
         - λ 对偶变量自适应更新
      2. 目标推断课程：
         - GIC 预测下一子目标，提供导航引导奖励
         - 随训练进度增加任务难度
      3. 安全约束层：
         - LiDAR + 超声波约束（同 RLCA）

    参考：Achiam et al., "Constrained Policy Optimization", ICML 2017
    """

    def __init__(self,
                 state_dim:   int   = STATE_DIM,
                 action_dim:  int   = ACTION_DIM,
                 hidden:      int   = 256,
                 lr:          float = 3e-4,
                 lr_lambda:   float = 0.01,
                 gamma:       float = 0.99,
                 gae_lambda:  float = 0.95,
                 clip_eps:    float = 0.2,
                 vf_coef:     float = 0.5,
                 ent_coef:    float = 0.01,
                 n_steps:     int   = 2048,
                 batch_size:  int   = 64,
                 n_epochs:    int   = 10,
                 cost_limit:  float = 0.1,   # δ_c 碰撞率约束
                 total_steps: int   = 2_000_000,
                 device:      str   = 'cpu'):
        super().__init__('CLPPO-GIC', state_dim, action_dim, device)
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.cost_limit = cost_limit
        self.lr_lambda  = lr_lambda
        self.total_target = total_steps

        # PPO AC
        self.ac  = PPOActorCritic(state_dim, action_dim, hidden,
                                  continuous=True).to(device)
        # 约束 Cost Critic（估计期望 cost）
        self.cost_critic = ValueNet(state_dim, hidden).to(device)
        self.opt = optim.Adam(
            list(self.ac.parameters()) + list(self.cost_critic.parameters()),
            lr=lr, eps=1e-5)

        # Lagrangian 乘子 λ（可学习）
        self.log_lambda = torch.tensor(0., requires_grad=True, device=device)
        self.lambda_opt = optim.Adam([self.log_lambda], lr=lr_lambda)

        # GIC 模块
        self.gic = GoalInferenceCurriculum(state_dim)
        self.gic.goal_net = self.gic.goal_net.to(device)
        self.gic_opt = optim.Adam(self.gic.goal_net.parameters(), lr=1e-3)

        self.buffer      = RolloutBuffer()
        self.cost_buffer: List[float] = []   # 约束 cost 历史

    def _cost(self, state: np.ndarray) -> float:
        """
        单步 cost：碰撞指示函数近似。
        LiDAR < 0.3m → cost=1，否则 sigmoid 软近似。
        """
        d = get_obstacle_min(state)
        return float(1. / (1. + math.exp(20. * (d - 0.3))))

    def _gic_reward(self, state: np.ndarray) -> float:
        """子目标推断奖励：与推断方向一致时给正奖励。"""
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            delta = self.gic.goal_net(s).squeeze(0).cpu().numpy()
        # 若机器人朝向与子目标方向接近，给小正奖励
        yaw = float(state[9])
        goal_angle = math.atan2(delta[1], delta[0])
        align = math.cos(yaw - goal_angle)
        return float(max(0., align) * 0.1)

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        s = self._to_tensor(state)
        with torch.no_grad():
            dist, _ = self.ac(s)
            a = dist.mean if deterministic else dist.sample()
        action = a.squeeze(0).cpu().numpy()
        # LiDAR 安全限速
        min_d = get_obstacle_min(state)
        if min_d < 0.4:
            action[0] = np.clip(action[0], -1., max(0., min_d/0.4))
        # 超声波搁浅
        us_min = float(np.min(state[US_IDX]))
        if us_min < 0.3 and float(state[DEPTH_IDX]) < 0.1:
            action[0] *= 0.3
        return np.clip(action, -1., 1.)

    def get_log_prob_value(self, state, action):
        s = self._to_tensor(state)
        a = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, v = self.ac(s); lp = dist.log_prob(a).sum(-1)
        return float(lp.item()), float(v.item())

    def store(self, state, action, log_prob, base_reward, value, done):
        gic_r  = self._gic_reward(state)
        cost   = self._cost(state)
        self.cost_buffer.append(cost)
        reward = base_reward + gic_r
        self.buffer.add(state, action, log_prob, reward, value, done)
        self.total_steps += 1

    def update(self, last_value=0.) -> Dict[str, float]:
        if len(self.buffer) < self.n_steps:
            return {}
        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        data = self.buffer.get(self.device); B = len(self.buffer)

        # 当前 Lagrangian 乘子
        lam = self.log_lambda.exp().item()

        # 估计当前期望 cost
        mean_cost = float(np.mean(self.cost_buffer[-self.n_steps:]))
        self.cost_buffer = self.cost_buffer[-10000:]

        # 更新 λ：λ ← max(0, λ + lr·(E[cost] - δ_c))
        cost_violation = mean_cost - self.cost_limit
        lambda_loss    = -self.log_lambda * cost_violation
        self.lambda_opt.zero_grad(); lambda_loss.backward()
        self.lambda_opt.step()
        self.log_lambda.data.clamp_(min=-5., max=5.)

        metrics = {'loss':0.,'pi':0.,'vf':0.,'cost':mean_cost,'lambda':lam}
        n_b = 0
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
                # Cost critic（简化：用 cost_violation 正则化策略损失）
                cost_pen = lam * cost_violation
                loss = pi_l + self.vf_coef*vf_l - self.ent_coef*ent + cost_pen
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.ac.parameters())+list(self.cost_critic.parameters()), 0.5)
                self.opt.step()
                metrics['loss']+=loss.item(); metrics['pi']+=pi_l.item()
                metrics['vf']+=vf_l.item(); n_b+=1
        self.buffer.reset(); self.n_updates+=1
        return {k:v/n_b if k in('loss','pi','vf') else v
                for k,v in metrics.items()} if n_b else {}

    def save(self, path):
        torch.save({'ac': self.ac.state_dict(),
                    'log_lambda': self.log_lambda.data,
                    'steps': self.total_steps}, path)

    def load(self, path):
        c=torch.load(path,map_location=self.device)
        self.ac.load_state_dict(c['ac'])
        self.log_lambda.data=c['log_lambda']
        self.total_steps=c.get('steps',0)
