"""
baselines_02.py
─────────────────────────────────────────────────────────────
Method 4: IMTCMO  — Intrinsic Motivation + Task-Conditioned
                    Multi-Objective RL
Method 5: APF-DQN — Artificial Potential Field + DQN
Method 6: I-DDPG  — Improved DDPG with target noise
─────────────────────────────────────────────────────────────
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, Tuple
import math

from baseline_base import (
    BaseAgent, ReplayBuffer, RolloutBuffer,
    StochasticActor, DeterministicActor, QNetwork, ValueNet,
    PPOActorCritic, mlp, soft_update, hard_update,
    STATE_DIM, ACTION_DIM,
    get_obstacle_min, get_water_depth, get_domain,
    LIDAR_IDX, US_IDX, DEPTH_IDX,
)


# ─────────────────────────────────────────────────────────────────────────────
# Method 4: IMTCMO — Intrinsic Motivation + Task-Conditioned Multi-Objective
# ─────────────────────────────────────────────────────────────────────────────

class ICMModule(nn.Module):
    """
    Intrinsic Curiosity Module (Pathak et al., 2017)。

    生成内在奖励：r_int = η · ||φ(s') - φ̂(s')||²
    其中 φ̂(s') 由前向模型 (state, action) → 预测 φ(s') 得到。
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 feature_dim=64, hidden=128):
        super().__init__()
        # 特征编码器 φ
        self.encoder = mlp([state_dim, hidden, feature_dim], nn.ReLU)
        # 前向模型：(φ(s), a) → φ̂(s')
        self.forward_net = mlp([feature_dim + action_dim, hidden, feature_dim])
        # 逆向模型：(φ(s), φ(s')) → â
        self.inverse_net = mlp([feature_dim * 2, hidden, action_dim])

    def forward(self, s, a, s_next):
        phi_s  = self.encoder(s)
        phi_s2 = self.encoder(s_next)
        # 前向损失
        phi_pred = self.forward_net(torch.cat([phi_s, a], -1))
        fwd_loss = F.mse_loss(phi_pred, phi_s2.detach())
        # 逆向损失
        a_pred   = self.inverse_net(torch.cat([phi_s, phi_s2], -1))
        inv_loss = F.mse_loss(a_pred, a)
        # 内在奖励
        r_int    = 0.5 * (phi_pred.detach() - phi_s2.detach()).pow(2).sum(-1)
        return r_int, fwd_loss, inv_loss


class IMTCMO(BaseAgent):
    """
    Intrinsic Motivation + Task-Conditioned Multi-Objective RL。

    核心特点：
      1. ICM 内在奖励驱动水-陆过渡区探索
      2. 多目标加权奖励：R = w_nav·R_nav + w_safe·R_safe + w_int·R_int
         - R_nav:  导航进度奖励
         - R_safe: 安全约束惩罚（LiDAR 近障碍物）
         - R_int:  ICM 内在好奇心奖励（鼓励探索未知域）
      3. PPO 策略更新
      4. 任务条件化：将域标签 one-hot 嵌入状态

    权重参数：w_nav=1.0, w_safe=0.5, w_int=0.1
    """

    def __init__(self,
                 state_dim:   int   = STATE_DIM,
                 action_dim:  int   = ACTION_DIM,
                 hidden:      int   = 256,
                 lr:          float = 3e-4,
                 lr_icm:      float = 1e-3,
                 gamma:       float = 0.99,
                 gae_lambda:  float = 0.95,
                 clip_eps:    float = 0.2,
                 vf_coef:     float = 0.5,
                 ent_coef:    float = 0.01,
                 n_steps:     int   = 2048,
                 batch_size:  int   = 64,
                 n_epochs:    int   = 10,
                 w_nav:       float = 1.0,
                 w_safe:      float = 0.5,
                 w_int:       float = 0.1,
                 device:      str   = 'cpu'):
        super().__init__('IMTCMO', state_dim, action_dim, device)
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps   = clip_eps
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef
        self.n_steps    = n_steps
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.w_nav      = w_nav
        self.w_safe     = w_safe
        self.w_int      = w_int

        # PPO AC（域 one-hot 嵌入：state_dim + 3）
        aug_dim = state_dim + 3
        self.ac  = PPOActorCritic(aug_dim, action_dim, hidden,
                                  continuous=True).to(device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr, eps=1e-5)

        # ICM
        self.icm     = ICMModule(state_dim, action_dim).to(device)
        self.icm_opt = optim.Adam(self.icm.parameters(), lr=lr_icm)

        self.buffer   = RolloutBuffer()
        self._prev_s  = None
        self._prev_a  = None

    def _augment(self, state: np.ndarray) -> np.ndarray:
        """将域标签转为 one-hot 并拼接到状态。"""
        domain = get_domain(state)
        oh     = np.zeros(3, np.float32)
        if 0 <= domain <= 2:
            oh[domain] = 1.0
        return np.concatenate([state, oh])

    def _safety_reward(self, state: np.ndarray) -> float:
        """基于 LiDAR 最近障碍物计算安全惩罚。"""
        d = get_obstacle_min(state)
        if d < 0.3:   return -1.0
        if d < 0.6:   return -0.3
        return 0.0

    def compute_intrinsic_reward(self, s, a, s2) -> float:
        """计算 ICM 内在奖励。"""
        s_t  = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        a_t  = torch.FloatTensor(a).unsqueeze(0).to(self.device)
        s2_t = torch.FloatTensor(s2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            r_int, _, _ = self.icm(s_t, a_t, s2_t)
        return float(r_int.mean().item())

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        aug = self._augment(state)
        s   = torch.FloatTensor(aug).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, _ = self.ac(s)
            a = dist.mean if deterministic else dist.sample()
        action = a.squeeze(0).cpu().numpy()
        self._prev_s = state.copy()
        self._prev_a = action.copy()
        # LiDAR 安全裁剪
        min_d = get_obstacle_min(state)
        if min_d < 0.4:
            action[0] = np.clip(action[0], -1., max(0., min_d / 0.4))
        return np.clip(action, -1., 1.)

    def get_log_prob_value(self, state, action):
        aug = self._augment(state)
        s   = torch.FloatTensor(aug).unsqueeze(0).to(self.device)
        a   = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, v = self.ac(s)
            lp = dist.log_prob(a).sum(-1)
        return float(lp.item()), float(v.item())

    def store(self, state, action, log_prob,
              base_reward, value, done,
              next_state=None):
        # 多目标奖励合成
        r_nav  = base_reward
        r_safe = self._safety_reward(state)
        r_int  = (self.compute_intrinsic_reward(state, action, next_state)
                  if next_state is not None else 0.)
        reward = (self.w_nav * r_nav
                  + self.w_safe * r_safe
                  + self.w_int  * r_int)
        self.buffer.add(self._augment(state), action, log_prob,
                        reward, value, done)
        self.total_steps += 1

    def update_icm(self, states, actions, next_states):
        """更新 ICM 网络。"""
        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.FloatTensor(actions).to(self.device)
        s2 = torch.FloatTensor(next_states).to(self.device)
        _, fwd_l, inv_l = self.icm(s, a, s2)
        loss = 0.5 * fwd_l + 0.5 * inv_l
        self.icm_opt.zero_grad(); loss.backward()
        self.icm_opt.step()
        return float(loss.item())

    def update(self, last_value: float = 0.) -> Dict[str, float]:
        if len(self.buffer) < self.n_steps:
            return {}

        self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)
        data = self.buffer.get(self.device)
        B    = len(self.buffer)

        metrics = {'loss':0., 'pi':0., 'vf':0., 'ent':0.}
        n_b = 0
        for _ in range(self.n_epochs):
            perm = torch.randperm(B, device=self.device)
            for start in range(0, B, self.batch_size):
                idx = perm[start:start + self.batch_size]
                if len(idx) < 2: continue
                s   = data['s'][idx]; a = data['a'][idx]
                lp0 = data['lp'][idx]; adv = data['adv'][idx]; ret = data['ret'][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                dist, v = self.ac(s)
                lp1 = dist.log_prob(a).sum(-1)
                ent = dist.entropy().sum(-1).mean()
                r   = (lp1 - lp0).exp()
                pi_l= -torch.min(r*adv, r.clamp(1-self.clip_eps,1+self.clip_eps)*adv).mean()
                vf_l= F.mse_loss(v, ret)
                loss= pi_l + self.vf_coef*vf_l - self.ent_coef*ent
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()
                metrics['loss']+=loss.item(); metrics['pi']+=pi_l.item()
                metrics['vf']+=vf_l.item(); metrics['ent']+=ent.item()
                n_b += 1

        self.buffer.reset(); self.n_updates += 1
        return {k: v/n_b for k, v in metrics.items()} if n_b else {}

    def save(self, path):
        torch.save({'ac': self.ac.state_dict(),
                    'icm': self.icm.state_dict(),
                    'steps': self.total_steps}, path)

    def load(self, path):
        c = torch.load(path, map_location=self.device)
        self.ac.load_state_dict(c['ac'])
        self.icm.load_state_dict(c['icm'])
        self.total_steps = c.get('steps', 0)


# ─────────────────────────────────────────────────────────────────────────────
# Method 5: APF-DQN — Artificial Potential Field + DQN
# ─────────────────────────────────────────────────────────────────────────────

class APFDQN(BaseAgent):
    """
    Artificial Potential Field + DQN 两栖导航。

    APF 势场构建：
      U_total = U_att + U_rep
      U_att = 0.5 · ξ · d(q, q_goal)²        (吸引势)
      U_rep = 0.5 · η · (1/d - 1/d0)²  if d<d0  (排斥势，d=LiDAR 最小距离)
             = 0                          otherwise

    DQN 接收 [state, APF_force] 作为增强状态，输出离散动作 Q 值。
    APF 提供初始导航方向，DQN 学习残差控制。

    动作空间：9 个（3×3 vx × wz 组合）
    """

    VX_BINS = np.array([-0.8, 0., 0.8], np.float32)
    WZ_BINS = np.array([-0.8, 0., 0.8], np.float32)
    N_ACTS  = 9

    # APF 参数
    XI   = 1.0    # 吸引势系数
    ETA  = 2.0    # 排斥势系数
    D0   = 1.5    # 排斥生效距离 m

    def __init__(self,
                 state_dim:   int   = STATE_DIM,
                 hidden:      int   = 256,
                 lr:          float = 1e-3,
                 gamma:       float = 0.99,
                 tau:         float = 0.005,
                 buffer_cap:  int   = 50_000,
                 batch_size:  int   = 128,
                 eps_start:   float = 1.0,
                 eps_end:     float = 0.05,
                 eps_decay:   int   = 40_000,
                 learn_starts:int   = 1000,
                 device:      str   = 'cpu'):
        super().__init__('APF-DQN', state_dim, ACTION_DIM, device)
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.eps_start    = eps_start
        self.eps_end      = eps_end
        self.eps_decay    = eps_decay
        self.learn_starts = learn_starts

        # 增强状态维度：原始状态 + APF 力 [F_att_x, F_att_y, F_rep_x, F_rep_y]
        aug_dim = state_dim + 4

        self.online = mlp([aug_dim, hidden, hidden, self.N_ACTS]).to(device)
        self.target = mlp([aug_dim, hidden, hidden, self.N_ACTS]).to(device)
        hard_update(self.target, self.online)
        for p in self.target.parameters(): p.requires_grad = False

        self.opt    = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_cap)

        self._action_table = np.array([
            [vx, wz] for vx in self.VX_BINS for wz in self.WZ_BINS
        ], dtype=np.float32)  # (9, 2)

    def _apf_force(self, state: np.ndarray,
                   goal_direction: float) -> np.ndarray:
        """
        计算 APF 合力 [F_att_x, F_att_y, F_rep_x, F_rep_y]。
        输入:
          state         : 25维状态
          goal_direction: 目标方向角 (rad)，由 d_goal 和 yaw 计算
        """
        d_goal = float(state[10])
        yaw    = float(state[9])

        # 吸引力（沿目标方向）
        f_att_x = self.XI * d_goal * math.cos(goal_direction - yaw)
        f_att_y = self.XI * d_goal * math.sin(goal_direction - yaw)
        f_att_x = np.clip(f_att_x, -2., 2.)
        f_att_y = np.clip(f_att_y, -2., 2.)

        # 排斥力（LiDAR 8扇区，取最近扇区）
        lidar = state[LIDAR_IDX]  # 8 values
        f_rep_x = f_rep_y = 0.
        for i, d in enumerate(lidar):
            if d < self.D0:
                angle   = yaw + (-np.pi + i * np.pi/4)   # 扇区角度（均匀分布）
                rep_mag = self.ETA * (1./max(d, 0.1) - 1./self.D0) / (d**2)
                f_rep_x -= rep_mag * math.cos(angle)
                f_rep_y -= rep_mag * math.sin(angle)
        f_rep_x = np.clip(f_rep_x, -3., 3.)
        f_rep_y = np.clip(f_rep_y, -3., 3.)

        return np.array([f_att_x, f_att_y, f_rep_x, f_rep_y], np.float32)

    def _augment(self, state: np.ndarray,
                 goal_dir: float = 0.) -> np.ndarray:
        force = self._apf_force(state, goal_dir)
        return np.concatenate([state, force])

    def _eps(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
               math.exp(-self.total_steps / self.eps_decay)

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False,
                      goal_dir: float = 0.) -> np.ndarray:
        aug = self._augment(state, goal_dir)
        if not deterministic and np.random.random() < self._eps():
            # APF 引导的 ε-greedy：以 0.5 概率跟随 APF 方向
            if np.random.random() < 0.5:
                force = aug[-4:]
                vx_apf = np.clip(force[0] + force[2], -1., 1.)
                wz_apf = np.clip(force[1] + force[3], -1., 1.)
                dists  = np.linalg.norm(
                    self._action_table - np.array([vx_apf, wz_apf]), axis=1)
                idx = int(np.argmin(dists))
            else:
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
        dists  = np.linalg.norm(self._action_table - a, axis=1)
        idx    = int(np.argmin(dists))
        self.buffer.add(aug_s, [idx], r, aug_s2, done)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.learn_starts:
            return {}

        batch = self.buffer.sample(self.batch_size, self.device)
        s = batch['s']; a_idx = batch['a'].long().squeeze(-1)
        r = batch['r']; s2 = batch['s2']; d = batch['d']

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
# Method 6: I-DDPG — Improved DDPG
# ─────────────────────────────────────────────────────────────────────────────

class IDDPG(BaseAgent):
    """
    Improved DDPG for amphibious navigation。

    改进点：
      1. Target Policy Smoothing：目标动作加截断高斯噪声，减少过拟合
         ã = clip(a_target + clip(ε, -c, c),  -1, 1)，ε~N(0, σ)
      2. 延迟策略更新：每 policy_delay 步更新一次 Actor
      3. 域感知探索噪声：Ornstein-Uhlenbeck 过程，σ 随域自适应
      4. LiDAR 安全层：前向速度受最近障碍物线性限制

    参考 TD3 (Fujimoto et al., 2018)。
    """

    def __init__(self,
                 state_dim:   int   = STATE_DIM,
                 action_dim:  int   = ACTION_DIM,
                 hidden:      int   = 256,
                 lr_actor:    float = 1e-4,
                 lr_critic:   float = 1e-3,
                 gamma:       float = 0.99,
                 tau:         float = 0.005,
                 buffer_cap:  int   = 100_000,
                 batch_size:  int   = 256,
                 learn_starts:int   = 1000,
                 policy_delay:int   = 2,
                 noise_std:   float = 0.2,
                 noise_clip:  float = 0.5,
                 device:      str   = 'cpu'):
        super().__init__('I-DDPG', state_dim, action_dim, device)
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.learn_starts = learn_starts
        self.policy_delay = policy_delay
        self.noise_std    = noise_std
        self.noise_clip   = noise_clip

        # Actor (deterministic)
        self.actor        = DeterministicActor(state_dim, action_dim, hidden).to(device)
        self.actor_target = DeterministicActor(state_dim, action_dim, hidden).to(device)
        hard_update(self.actor_target, self.actor)

        # Twin Critic
        self.critic        = QNetwork(state_dim, action_dim, hidden, twin=True).to(device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden, twin=True).to(device)
        hard_update(self.critic_target, self.critic)
        for p in list(self.actor_target.parameters()) + \
                 list(self.critic_target.parameters()):
            p.requires_grad = False

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.buffer     = ReplayBuffer(buffer_cap)

        # OU 噪声状态
        self._ou_theta = 0.15
        self._ou_sigma_base = 0.3
        self._ou_x = np.zeros(action_dim, np.float32)

    def _ou_noise(self, domain: int) -> np.ndarray:
        """域感知 OU 噪声：水中探索更强。"""
        sigma = {0: 0.4, 1: 0.3, 2: 0.2}.get(domain, 0.3) * self._ou_sigma_base
        self._ou_x += (self._ou_theta * (-self._ou_x) +
                       sigma * np.random.randn(self.action_dim)).astype(np.float32)
        return self._ou_x.copy()

    def select_action(self, state: np.ndarray,
                      deterministic: bool = False) -> np.ndarray:
        s = self._to_tensor(state)
        with torch.no_grad():
            a = self.actor(s).squeeze(0).cpu().numpy()
        if not deterministic:
            a += self._ou_noise(get_domain(state))
        # LiDAR 安全裁剪
        min_d = get_obstacle_min(state)
        if min_d < 0.4:
            a[0] = np.clip(a[0], -1., max(0., min_d / 0.4))
        return np.clip(a, -1., 1.)

    def store(self, s, a, r, s2, done):
        self.buffer.add(s, a, r, s2, done)
        self.total_steps += 1

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.learn_starts:
            return {}

        b  = self.buffer.sample(self.batch_size, self.device)
        s, a, r, s2, d = b['s'], b['a'], b['r'], b['s2'], b['d']

        # Critic update
        with torch.no_grad():
            noise  = (torch.randn_like(a) * self.noise_std).clamp(
                -self.noise_clip, self.noise_clip)
            a_next = (self.actor_target(s2) + noise).clamp(-1., 1.)
            q1_t, q2_t = self.critic_target(s2, a_next)
            y      = r + self.gamma * (1 - d) * torch.min(q1_t, q2_t)

        q1, q2  = self.critic(s, a)
        q_loss  = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad(); q_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.)
        self.critic_opt.step()

        actor_loss_val = 0.
        # Delayed Actor update
        if self.n_updates % self.policy_delay == 0:
            q1_pi, _ = self.critic(s, self.actor(s))
            actor_loss = -q1_pi.mean()
            self.actor_opt.zero_grad(); actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            self.actor_opt.step()
            soft_update(self.actor_target,  self.actor,  self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
            actor_loss_val = actor_loss.item()

        self.n_updates += 1
        return {'q_loss': q_loss.item(), 'actor_loss': actor_loss_val}

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'steps': self.total_steps,
        }, path)

    def load(self, path):
        c = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(c['actor'])
        self.actor_target.load_state_dict(c['actor_target'])
        self.critic.load_state_dict(c['critic'])
        self.critic_target.load_state_dict(c['critic_target'])
        self.total_steps = c.get('steps', 0)
