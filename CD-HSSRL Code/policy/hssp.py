"""
HSSP: Hierarchical Safe Switching Policy
论文 Section 3.4, Eq.(13)–(16)

四个核心组件：
  1. HighLevelPolicyNet   — π_H 网络定义       (Eq.13)
  2. TerminationCondition — β(o_t|s_t)         (Eq.14)
  3. SwitchingRegularizer — L_sw               (Eq.15)
  4. PPOUpdater           — L_H 优化            (Eq.16)
  5. HierarchicalSafeSwitchingPolicy — 顶层封装
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, NamedTuple
from collections import deque
import copy


# ─────────────────────────────────────────────────────────────────────────────
# 常量：Motion Option 枚举
# ─────────────────────────────────────────────────────────────────────────────

OPTION_WATER      = 0   # 水域模式
OPTION_TRANSITION = 1   # 过渡区模式
OPTION_LAND       = 2   # 陆地模式
OPTION_NAMES      = ['Water', 'Transition', 'Land']
N_OPTIONS         = 3


# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HSSPConfig:
    # 网络结构 (Section 4.4)
    state_dim:    int   = 25      # 观测维度 (GPS+IMU+LiDAR+US+Depth+domain)
    waypoint_dim: int   = 4       # 航点编码维度 [Δx, Δy, dist, angle]
    hidden_size:  int   = 256
    n_hidden:     int   = 2

    # PPO 超参数 (Section 4.4)
    lr:           float = 3e-4
    gamma:        float = 0.99
    gae_lambda:   float = 0.95
    clip_eps:     float = 0.2     # ε in Eq.16
    vf_coef:      float = 0.5
    ent_coef:     float = 0.01
    max_grad_norm:float = 0.5
    n_epochs:     int   = 10
    batch_size:   int   = 64
    n_steps:      int   = 2048   # rollout 长度

    # 切换正则化 (Eq.15)
    lambda_sw:    float = 0.05   # λ_sw


# ─────────────────────────────────────────────────────────────────────────────
# 1. 神经网络定义  (Eq.13)
# ─────────────────────────────────────────────────────────────────────────────

class HighLevelPolicyNet(nn.Module):
    """
    π_H(o|s_t; θ_H) — Eq.(13)

    输入: [state_t(25), waypoint_encoding(4)]  → dim = state_dim + waypoint_dim
         state: GPS(3)+IMU_accel(3)+IMU_gyro(3)+yaw+d_goal+LiDAR(8)+US(4)+depth+domain
    输出:
      option_probs : softmax 概率分布，shape (..., N_OPTIONS)
      value        : 状态价值 V(s)，shape (..., 1)

    结构: 2 × Linear(256) + ReLU → 分叉为 option_head + value_head
    (Section 4.4: "multilayer perceptron with two hidden layers of 256 units,
                   followed by a softmax output layer for option selection")
    """

    def __init__(self, cfg: HSSPConfig):
        super().__init__()
        input_dim = cfg.state_dim + cfg.waypoint_dim

        # 共享主干
        layers = []
        in_dim = input_dim
        for _ in range(cfg.n_hidden):
            layers += [nn.Linear(in_dim, cfg.hidden_size), nn.ReLU()]
            in_dim  = cfg.hidden_size
        self.trunk = nn.Sequential(*layers)

        # Option 头：softmax 输出  (Eq.13)
        self.option_head = nn.Linear(cfg.hidden_size, N_OPTIONS)
        # Value 头：PPO critic
        self.value_head  = nn.Linear(cfg.hidden_size, 1)

        # 正交初始化（PPO 常用）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # option head 用较小增益
        nn.init.orthogonal_(self.option_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight,  gain=1.0)

    def forward(self,
                state:    torch.Tensor,
                waypoint: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state    : (..., state_dim)
            waypoint : (..., waypoint_dim)
        Returns:
            probs    : (..., N_OPTIONS)   softmax option 分布
            value    : (..., 1)           状态价值
        """
        x     = torch.cat([state, waypoint], dim=-1)
        h     = self.trunk(x)
        probs = F.softmax(self.option_head(h), dim=-1)
        value = self.value_head(h)
        return probs, value

    def get_dist(self,
                 state:    torch.Tensor,
                 waypoint: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """返回 Categorical 分布和 value，方便采样和计算 log_prob。"""
        probs, value = self.forward(state, waypoint)
        return Categorical(probs), value.squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 终止条件  (Eq.14)
# ─────────────────────────────────────────────────────────────────────────────

class TerminationCondition:
    """
    β(o_t|s_t) — Eq.(14)

    β = 1  if  s_t ∈ S_t  (过渡区)  or  waypoint_reached
    β = 0  otherwise

    当 β=1 时，高层策略重新选择 option；β=0 时保持当前 option。
    """

    def __init__(self,
                 transition_x_min: float = -2.5,
                 transition_x_max: float =  2.5,
                 waypoint_radius:  float =  1.5):
        self.x_min  = transition_x_min
        self.x_max  = transition_x_max
        self.wp_rad = waypoint_radius

    def __call__(self,
                 robot_pos:      np.ndarray,
                 waypoint:       np.ndarray,
                 domain_label:   int) -> bool:
        """
        Returns True（需要重新选择 option）当且仅当：
          1. 机器人在过渡区 (domain_label == OPTION_TRANSITION), 或
          2. 到达当前航点
        """
        in_transition  = (domain_label == OPTION_TRANSITION)
        wp_dist        = float(np.linalg.norm(robot_pos[:2] - waypoint[:2]))
        waypoint_reached = wp_dist < self.wp_rad
        return bool(in_transition or waypoint_reached)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 切换正则化损失  (Eq.15)
# ─────────────────────────────────────────────────────────────────────────────

class SwitchingRegularizer:
    """
    L_sw = λ_sw · ||π_H(·|s_t) - π_H(·|s_{t-1})||²_2   (Eq.15)

    软性惩罚连续决策步之间 option 分布的突变，
    促进平滑稳定的域切换行为。
    """

    def __init__(self, lambda_sw: float = 0.05):
        self.lambda_sw = lambda_sw

    def __call__(self,
                 probs_t:    torch.Tensor,
                 probs_prev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs_t    : (..., N_OPTIONS)  当前步 option 分布
            probs_prev : (..., N_OPTIONS)  上一步 option 分布（detach）
        Returns:
            L_sw scalar
        """
        diff = probs_t - probs_prev.detach()
        return self.lambda_sw * (diff ** 2).sum(dim=-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rollout 缓冲区
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    存储 n_steps 个时间步的轨迹数据，供 PPO 更新使用。
    每条记录：(state, waypoint, option, log_prob, reward, value, done, probs)
    其中 probs 用于计算 L_sw (Eq.15)。
    """

    def __init__(self):
        self.states:    List[np.ndarray] = []
        self.waypoints: List[np.ndarray] = []
        self.options:   List[int]        = []
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[bool]       = []
        self.probs:     List[np.ndarray] = []   # 完整分布，用于 L_sw
        self.advantages: np.ndarray      = np.array([])
        self.returns_:   np.ndarray      = np.array([])

    def add(self,
            state:    np.ndarray,
            waypoint: np.ndarray,
            option:   int,
            log_prob: float,
            reward:   float,
            value:    float,
            done:     bool,
            probs:    np.ndarray):
        self.states.append(state.copy())
        self.waypoints.append(waypoint.copy())
        self.options.append(option)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.probs.append(probs.copy())

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, last_value: float,
                    gamma: float, gae_lambda: float):
        """
        Generalized Advantage Estimation (GAE)。
        δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        Â_t = Σ_{k=0}^{T-t} (γλ)^k · δ_{t+k}
        """
        T      = len(self.rewards)
        adv    = np.zeros(T, dtype=np.float32)
        gae    = 0.0
        values = self.values + [last_value]

        for t in reversed(range(T)):
            nv     = values[t + 1] * (1.0 - float(self.dones[t]))
            delta  = self.rewards[t] + gamma * nv - values[t]
            gae    = delta + gamma * gae_lambda * (1.0 - float(self.dones[t])) * gae
            adv[t] = gae

        self.advantages = adv
        self.returns_   = adv + np.array(self.values, dtype=np.float32)

    def get(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """返回用于 PPO 更新的张量字典。"""
        def t(x):  return torch.FloatTensor(np.array(x)).to(device)
        def ti(x): return torch.LongTensor(np.array(x)).to(device)
        return {
            'states':     t(self.states),
            'waypoints':  t(self.waypoints),
            'options':    ti(self.options),
            'log_probs':  t(self.log_probs),
            'advantages': t(self.advantages),
            'returns':    t(self.returns_),
            'probs':      t(self.probs),        # 用于 L_sw
        }

    def reset(self):
        self.__init__()


# ─────────────────────────────────────────────────────────────────────────────
# 5. PPO 更新器  (Eq.16)
# ─────────────────────────────────────────────────────────────────────────────

class PPOUpdater:
    """
    L_H(θ_H) = E_t [ min( r_t(θ_H)·Â_t,
                           clip(r_t(θ_H), 1-ε, 1+ε)·Â_t ) ]   (Eq.16)

    其中 r_t(θ_H) = π_H(o_t|s_t;θ_H) / π_H(o_t|s_t;θ_H^old)

    总损失 = L_H + vf_coef·L_value - ent_coef·H + L_sw        (Eq.21 partial)
    """

    def __init__(self, policy: HighLevelPolicyNet,
                 cfg: HSSPConfig, device: str = 'cpu'):
        self.policy    = policy
        self.cfg       = cfg
        self.device    = device
        self.optimizer = optim.Adam(policy.parameters(), lr=cfg.lr,
                                    eps=1e-5)
        self.sw_reg    = SwitchingRegularizer(cfg.lambda_sw)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        对 buffer 中的数据执行 n_epochs 轮 PPO 更新。
        Returns: 各损失项均值的字典
        """
        data = buffer.get(self.device)
        B    = len(buffer)

        metrics = {
            'loss_total': 0., 'loss_policy': 0.,
            'loss_value': 0., 'loss_entropy': 0.,
            'loss_sw': 0.,    'clip_frac': 0.,
        }
        n_batches = 0

        for _ in range(self.cfg.n_epochs):
            perm = torch.randperm(B, device=self.device)

            for start in range(0, B, self.cfg.batch_size):
                idx = perm[start : start + self.cfg.batch_size]
                if len(idx) < 2:          # batch 太小跳过
                    continue

                s   = data['states'][idx]
                wp  = data['waypoints'][idx]
                opt = data['options'][idx]
                old_lp  = data['log_probs'][idx]
                adv     = data['advantages'][idx]
                ret     = data['returns'][idx]
                old_probs = data['probs'][idx]

                # 归一化 advantage（减少方差）
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # ── 前向传播 ────────────────────────────────────────
                dist, value = self.policy.get_dist(s, wp)
                new_lp  = dist.log_prob(opt)
                entropy = dist.entropy().mean()
                new_probs, _ = self.policy(s, wp)

                # ── PPO 裁剪目标 L_H (Eq.16) ────────────────────────
                ratio  = torch.exp(new_lp - old_lp)
                surr1  = ratio * adv
                surr2  = torch.clamp(ratio,
                                     1.0 - self.cfg.clip_eps,
                                     1.0 + self.cfg.clip_eps) * adv
                loss_policy = -torch.min(surr1, surr2).mean()

                # 裁剪比例（监控训练稳定性）
                clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_eps).float().mean()

                # ── Value 损失 ──────────────────────────────────────
                loss_value = F.mse_loss(value, ret)

                # ── 切换正则化 L_sw (Eq.15) ────────────────────────
                # 用 batch 内相邻时间步近似 s_t vs s_{t-1}
                probs_cur  = new_probs
                probs_prev = torch.roll(old_probs, shifts=1, dims=0)
                probs_prev[0] = probs_cur[0].detach()  # 边界处理
                loss_sw = self.sw_reg(probs_cur, probs_prev)

                # ── 总损失 (Eq.21 H-part) ────────────────────────────
                loss = (loss_policy
                        + self.cfg.vf_coef  * loss_value
                        - self.cfg.ent_coef * entropy
                        + loss_sw)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                metrics['loss_total']   += loss.item()
                metrics['loss_policy']  += loss_policy.item()
                metrics['loss_value']   += loss_value.item()
                metrics['loss_entropy'] += entropy.item()
                metrics['loss_sw']      += loss_sw.item()
                metrics['clip_frac']    += clip_frac.item()
                n_batches += 1

        if n_batches > 0:
            metrics = {k: v / n_batches for k, v in metrics.items()}
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def encode_waypoint(robot_pos: np.ndarray,
                    waypoint:  np.ndarray) -> np.ndarray:
    """
    将下一个航点编码为机器人坐标系下的4维向量:
    [Δx, Δy, distance, angle]

    供 π_H 使用，使策略感知当前航点引导方向。
    """
    dx    = float(waypoint[0] - robot_pos[0])
    dy    = float(waypoint[1] - robot_pos[1])
    dist  = float(np.sqrt(dx**2 + dy**2))
    angle = float(np.arctan2(dy, dx))
    return np.array([dx, dy, dist, angle], dtype=np.float32)


def infer_domain_label(robot_z: float,
                       robot_x: float,
                       shore_x: float       = 0.0,
                       water_z_thr: float   = -0.05,
                       land_z_thr:  float   =  0.05) -> int:
    """根据机器人 z 坐标判断当前域标签（水/过渡/陆）。"""
    if robot_z < water_z_thr:
        return OPTION_WATER
    elif robot_z > land_z_thr:
        return OPTION_LAND
    else:
        return OPTION_TRANSITION


# ─────────────────────────────────────────────────────────────────────────────
# 顶层接口：HSSP
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalSafeSwitchingPolicy:
    """
    Hierarchical Safe Switching Policy (Section 3.4)

    完整流程（单步）:
      1. 判断终止条件 β(o_t|s_t)         (Eq.14)
      2. 若 β=1，从 π_H 采样新 option    (Eq.13)
      3. 将 (s,wp,o,logp,probs) 存入缓冲
      4. 满 n_steps 后调用 update() 优化  (Eq.15,16)

    评估指标:
      SSI = 1 - N_switch / T_total        (Eq.29)
    """

    def __init__(self, cfg: HSSPConfig, device: str = 'cpu'):
        self.cfg    = cfg
        self.device = device

        self.policy  = HighLevelPolicyNet(cfg).to(device)
        self.updater = PPOUpdater(self.policy, cfg, device)
        self.buffer  = RolloutBuffer()
        self.term    = TerminationCondition()

        # 当前 episode 状态
        self.current_option: int          = OPTION_WATER
        self.prev_probs: Optional[np.ndarray] = None

        # 统计
        self.n_updates:    int = 0
        self.switch_count: int = 0
        self.step_count:   int = 0

    # ------------------------------------------------------------------
    def select_option(self,
                      state:        np.ndarray,
                      waypoint_enc: np.ndarray,
                      domain_label: int,
                      deterministic: bool = False
                      ) -> Tuple[int, float, float, np.ndarray]:
        """
        执行一步高层决策。

        Args:
            state        : 当前状态观测 s_t, shape (state_dim,)
            waypoint_enc : 航点编码 [Δx,Δy,dist,angle]
            domain_label : 当前域 {0,1,2}
            deterministic: 评估模式（取 argmax 而非采样）

        Returns:
            option   : 选中的 motion option
            log_prob : log π_H(o_t|s_t)
            value    : V(s_t)
            probs    : 完整 option 分布，shape (N_OPTIONS,)
        """
        s_t  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        wp_t = torch.FloatTensor(waypoint_enc).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, value = self.policy.get_dist(s_t, wp_t)
            probs_np    = dist.probs.squeeze(0).cpu().numpy()

            if deterministic:
                option = int(probs_np.argmax())
            else:
                option = int(dist.sample().item())

            log_prob = float(dist.log_prob(
                torch.tensor([option], device=self.device)).item())
            val_f    = float(value.item())

        # 统计切换次数
        if self.step_count > 0 and option != self.current_option:
            self.switch_count += 1
        self.current_option = option
        self.step_count    += 1
        self.prev_probs     = probs_np

        return option, log_prob, val_f, probs_np

    # ------------------------------------------------------------------
    def should_reselect(self,
                        robot_pos:    np.ndarray,
                        waypoint:     np.ndarray,
                        domain_label: int) -> bool:
        """
        β(o_t|s_t) — Eq.(14)
        Returns True 时需要重新调用 select_option()。
        """
        return self.term(robot_pos, waypoint, domain_label)

    # ------------------------------------------------------------------
    def store(self,
              state:    np.ndarray,
              waypoint: np.ndarray,
              option:   int,
              log_prob: float,
              reward:   float,
              value:    float,
              done:     bool,
              probs:    np.ndarray):
        """将一步经验存入 rollout buffer。"""
        self.buffer.add(state, waypoint, option, log_prob,
                        reward, value, done, probs)

    # ------------------------------------------------------------------
    def ready_to_update(self) -> bool:
        """buffer 满 n_steps 时触发 PPO 更新。"""
        return len(self.buffer) >= self.cfg.n_steps

    # ------------------------------------------------------------------
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        执行 PPO + L_sw 联合更新，清空 buffer。
        Returns: 各损失项均值
        """
        self.buffer.compute_gae(last_value,
                                self.cfg.gamma,
                                self.cfg.gae_lambda)
        metrics = self.updater.update(self.buffer)
        self.buffer.reset()
        self.n_updates += 1
        return metrics

    # ------------------------------------------------------------------
    def switching_stability_index(self) -> float:
        """SSI = 1 - N_switch / T_total   (Eq.29)"""
        if self.step_count == 0:
            return 1.0
        return 1.0 - self.switch_count / self.step_count

    # ------------------------------------------------------------------
    def reset_episode(self):
        """Episode 结束时重置 episode 级别的状态。"""
        self.current_option = OPTION_WATER
        self.prev_probs     = None
        self.switch_count   = 0
        self.step_count     = 0

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            'policy':    self.policy.state_dict(),
            'optimizer': self.updater.optimizer.state_dict(),
            'n_updates': self.n_updates,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy'])
        self.updater.optimizer.load_state_dict(ckpt['optimizer'])
        self.n_updates = ckpt['n_updates']
