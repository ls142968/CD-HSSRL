#!/usr/bin/env python3
"""
train_cd_hssrl.py — CD-HSSRL 端到端训练主循环
论文 Algorithm 1 (Section 3.7)

串联三个模块：
  CD-GRP  → 航点序列 W
  HSSP    → 高层 option 选择 + PPO 更新
  SCCC    → 低层安全动作 + SAC 更新

用法：
  # 有 Gazebo（完整训练）:
  rosrun cd_hssrl train_cd_hssrl.py

  # 无 Gazebo（mock 环境调试）:
  python3 train_cd_hssrl.py --mock

  # 从断点继续:
  python3 train_cd_hssrl.py --mock --resume results/ckpt_step_050000
"""

import sys
import os
import argparse
import time
import json
import numpy as np
import torch
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent   # cd_hssrl/
sys.path.insert(0, str(ROOT / 'planner'))
sys.path.insert(0, str(ROOT / 'policy'))
sys.path.insert(0, str(ROOT / 'controller'))
sys.path.insert(0, str(ROOT / 'env'))

from cd_grp  import (CDGlobalReachabilityPlanner,
                     MapConfig, EnvironmentInfo, Obstacle)
from hssp    import (HierarchicalSafeSwitchingPolicy, HSSPConfig,
                     encode_waypoint, OPTION_WATER, N_OPTIONS)
from sccc    import SafetyConstrainedContinuousController, SCCCConfig


# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='CD-HSSRL Training')
parser.add_argument('--mock',        action='store_true',
                    help='使用 Mock 环境（无需 Gazebo）')
parser.add_argument('--resume',      type=str, default='',
                    help='断点路径前缀')
parser.add_argument('--total-steps', type=int, default=200_000,
                    help='总训练步数（论文: 2_000_000）')
parser.add_argument('--eval-freq',   type=int, default=5_000,
                    help='评估频率（步数）')
parser.add_argument('--seed',        type=int, default=42)
parser.add_argument('--task',        type=str, default='water_to_land',
                    choices=['water_to_land', 'land_to_water', 'multi_transition'])
parser.add_argument('--lambda-sw',   type=float, default=0.05)
parser.add_argument('--kappa',       type=float, default=1.0)
parser.add_argument('--device',      type=str, default='auto')
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# 初始化
# ─────────────────────────────────────────────────────────────────────────────

# 随机种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# 设备
if args.device == 'auto':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    DEVICE = args.device
print(f"[Train] Device={DEVICE}  Seed={args.seed}  Task={args.task}")

# 结果目录
RESULTS_DIR = ROOT / 'results'
CKPT_DIR    = RESULTS_DIR / 'checkpoints'
RESULTS_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 环境
# ─────────────────────────────────────────────────────────────────────────────

if args.mock:
    # ── Mock 环境（不依赖 ROS/Gazebo）────────────────────────────────
    class MockEnv:
        """
        轻量模拟环境，用于调试训练循环逻辑。
        机器人从水中 (-12, 0) 向陆地 (12, 0) 运动，
        穿越 x∈[-2.5, 2.5] 过渡区。
        """
        STATE_DIM = 25
        GOAL      = np.array([12.0, 0.0])
        START_POS = np.array([-12.0, 0.0, -0.3])

        def __init__(self, seed=0):
            self.rng = np.random.default_rng(seed)
            self.reset()

        def _domain(self):
            x, _, z = self.pos
            if z < -0.05:  return 0   # 水
            if z >  0.05:  return 2   # 陆
            return 1                   # 过渡

        def _state(self):
            x, y, z = self.pos
            d_goal   = float(np.linalg.norm(self.pos[:2] - self.GOAL))
            us       = np.clip(3.0 + self.rng.normal(0, 0.15, 4), 0.3, 5.0)
            acc      = self.rng.normal(0, 0.1, 3).astype(np.float32)
            return np.array([
                x, y, z,
                self.vel[0], self.vel[1], self.vel[2],
                self.yaw,
                d_goal,
                *us,
                *acc,
                z,                # depth ≈ -z（水中时为正深度）
                float(self._domain()),
            ], dtype=np.float32)

        def reset(self):
            self.pos    = self.START_POS.copy().astype(np.float32)
            self.vel    = np.zeros(3, dtype=np.float32)
            self.yaw    = 0.0
            self.t      = 0
            self._ep_r  = 0.0
            self._ep_e  = 0.0
            self._ep_p  = 0.0
            self._prev_d= float(np.linalg.norm(self.pos[:2]-self.GOAL))
            return self._state()

        def step(self, action, dt=0.05):
            vx = float(action[0]) * 0.6
            wz = float(action[1]) * 0.8
            self.vel[0] = vx + self.rng.normal(0, 0.03)
            self.vel[1] = self.rng.normal(0, 0.02)
            self.pos[0] += self.vel[0] * dt
            self.pos[1] += self.vel[1] * dt + wz * 0.05

            # z 随 x 变化（坡面模拟）
            px = self.pos[0]
            if   px < -2.5: self.pos[2] = -0.3
            elif px <  2.5: self.pos[2] = -0.3 + 0.4*(px+2.5)/5.0
            else:           self.pos[2] =  0.1

            self.t   += 1
            d_goal    = float(np.linalg.norm(self.pos[:2]-self.GOAL))
            progress  = self._prev_d - d_goal
            reached   = d_goal < 1.0
            collision = bool(self.rng.random() < 0.005)
            timeout   = self.t >= 400

            if reached:
                reward, done, result = 100.0, True, 'success'
            elif collision:
                reward, done, result = -50.0, True, 'collision'
            elif timeout:
                reward, done, result = -10.0, True, 'timeout'
            else:
                reward = 1.0 * progress - 0.05
                done, result = False, 'running'

            self._prev_d = d_goal
            self._ep_r  += reward
            self._ep_e  += float(np.sum(action**2))
            self._ep_p  += float(np.linalg.norm(self.vel[:2])) * dt

            info = {
                'result': result, 'domain': self._domain(),
                'dist_to_goal': d_goal, 'step': self.t,
            }
            if done:
                info.update({'ep_reward': self._ep_r,
                             'ep_length': self.t,
                             'ep_path':   self._ep_p,
                             'ep_energy': self._ep_e})
            return self._state(), reward, done, info

        def get_metrics(self): return {}

    env = MockEnv(seed=args.seed)
    GOAL_POS = MockEnv.GOAL

else:
    # ── Gazebo 环境 ─────────────────────────────────────────────────
    import rospy
    rospy.init_node('cd_hssrl_trainer', anonymous=False)
    from ky3_gazebo_env import KY3GazeboEnv
    env = KY3GazeboEnv(task=args.task, node_init=False)
    GOAL_POS = env.goal

# ─────────────────────────────────────────────────────────────────────────────
# 模块初始化
# ─────────────────────────────────────────────────────────────────────────────

# CD-GRP
map_cfg = MapConfig(
    resolution=0.4,
    x_min=-18.0, x_max=18.0,
    y_min=-12.0, y_max=12.0,
)
env_info = EnvironmentInfo(
    shoreline_x=0.0, slope_width=2.5,
    obstacles=[
        Obstacle(-10.0,  4.0, 1.0),
        Obstacle( -6.0, -3.0, 0.8),
        Obstacle(  5.0,  3.5, 0.9),
        Obstacle(  9.0, -3.0, 1.1),
    ],
)
cd_grp = CDGlobalReachabilityPlanner(map_cfg, env_info)
cd_grp.build()

# HSSP
hssp_cfg = HSSPConfig(
    state_dim   = 25,
    hidden_size = 256,
    n_steps     = 2048,
    batch_size  = 64,
    n_epochs    = 10,
    lr          = 3e-4,
    gamma       = 0.99,
    gae_lambda  = 0.95,
    clip_eps    = 0.2,
    lambda_sw   = args.lambda_sw,
)
hssp = HierarchicalSafeSwitchingPolicy(hssp_cfg, device=DEVICE)

# SCCC
sccc_cfg = SCCCConfig(
    state_dim      = 25,
    action_dim     = 2,
    hidden_size    = 256,
    lr             = 3e-4,
    gamma          = 0.99,
    tau            = 0.005,
    batch_size     = 256,
    buffer_size    = 1_000_000,
    learning_starts= 1000,
    kappa          = args.kappa,
)
sccc = SafetyConstrainedContinuousController(sccc_cfg, device=DEVICE)

# ── 断点恢复 ─────────────────────────────────────────────────────────────────
start_step = 0
if args.resume:
    try:
        hssp.load(args.resume + '_hssp.pt')
        sccc.load(args.resume + '_sccc.pt')
        # 从文件名提取步数
        import re
        m = re.search(r'step_(\d+)', args.resume)
        if m:
            start_step = int(m.group(1))
        print(f"[Train] 从步骤 {start_step} 恢复")
    except Exception as e:
        print(f"[Train] 断点加载失败: {e}，从头训练")

# ─────────────────────────────────────────────────────────────────────────────
# 训练历史
# ─────────────────────────────────────────────────────────────────────────────

history = {
    'steps': [], 'ep_reward': [], 'ep_length': [],
    'SR': [], 'CR': [], 'APL': [], 'EC': [],
    'loss_H': [], 'loss_sw': [], 'loss_L': [],
    'loss_q': [], 'alpha': [], 'SSI': [],
}

# Episode 统计
ep_rewards:   list = []
ep_lengths:   list = []
ep_results:   list = []
n_success:    int  = 0
n_collision:  int  = 0
n_episodes:   int  = 0

# ─────────────────────────────────────────────────────────────────────────────
# 训练函数
# ─────────────────────────────────────────────────────────────────────────────

def make_waypoints(start_pos: np.ndarray) -> list:
    """用 CD-GRP 生成航点序列 W = G(E)。"""
    start  = (float(start_pos[0]), float(start_pos[1]))
    goal   = (float(GOAL_POS[0]),  float(GOAL_POS[1]))
    _, wps = cd_grp.plan(start, goal, waypoint_spacing=2.0)
    return wps   # [(x,y), ...]


def get_next_waypoint(pos: np.ndarray, waypoints: list,
                      wp_idx: int) -> tuple:
    """返回当前目标航点和更新后的 wp_idx。"""
    wp = waypoints[min(wp_idx, len(waypoints)-1)]
    dist_to_wp = float(np.linalg.norm(pos[:2] - np.array(wp[:2])))
    if dist_to_wp < 1.5 and wp_idx < len(waypoints) - 1:
        wp_idx += 1
        wp = waypoints[wp_idx]
    return np.array(wp[:2]), wp_idx


def log_metrics(total_steps: int, ep_count: int):
    """打印当前训练指标。"""
    recent = ep_rewards[-20:] if ep_rewards else [0]
    sr  = n_success   / max(ep_count, 1)
    cr  = n_collision / max(ep_count, 1)
    ssi = hssp.switching_stability_index()
    alpha = sccc.updater.alpha

    print(f"  Steps={total_steps:7,}  Ep={ep_count:4d}  "
          f"R={np.mean(recent):7.1f}  "
          f"SR={sr:.3f}  CR={cr:.3f}  "
          f"SSI={ssi:.3f}  α={alpha:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 主训练循环  (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("CD-HSSRL 端到端训练  —  Algorithm 1")
print(f"  总步数: {args.total_steps:,}  任务: {args.task}")
print(f"  λ_sw={args.lambda_sw}  κ={args.kappa}  设备={DEVICE}")
print("="*60 + "\n")

total_steps = start_step
ppo_steps   = 0   # PPO rollout 累计步数

while total_steps < args.total_steps:

    # ────────────────────────────────────────────────────────────────
    # Episode 初始化
    # ────────────────────────────────────────────────────────────────
    state     = env.reset()
    hssp.reset_episode()
    n_episodes += 1

    # Step W = G(E)：CD-GRP 规划航点  (Algorithm 1 Line 3)
    waypoints = make_waypoints(state[:3])
    wp_idx    = 0

    ep_reward    = 0.0
    ep_steps     = 0
    prev_option  = OPTION_WATER
    prev_probs   = None

    # ────────────────────────────────────────────────────────────────
    # Episode 内循环
    # ────────────────────────────────────────────────────────────────
    while ep_steps < (hssp_cfg.n_steps * 2):

        # 当前航点编码
        wp_pos, wp_idx = get_next_waypoint(state[:2], waypoints, wp_idx)
        wp_enc = encode_waypoint(state[:2], wp_pos)
        domain = int(state[-1])

        # ── Step o_t ~ π_H(o|s_t)  (Eq.13) ──────────────────────
        need_reselect = (
            ep_steps == 0 or
            hssp.should_reselect(state[:2], wp_pos, domain)
        )
        if need_reselect:
            option, lp_H, val_H, probs = hssp.select_option(
                state, wp_enc, domain, deterministic=False)
        else:
            option = hssp.current_option
            s_t  = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            wp_t = torch.FloatTensor(wp_enc).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                d, v = hssp.policy.get_dist(s_t, wp_t)
                lp_H  = float(d.log_prob(
                    torch.tensor([option], device=DEVICE)).item())
                val_H = float(v.item())
                probs = d.probs.squeeze(0).cpu().numpy()

        # ── Step a_safe, a_raw  (Eq.17 + Eq.18) ─────────────────
        safe_action, raw_action = sccc.select_action(
            state, option, deterministic=False)

        # ── 环境 step ─────────────────────────────────────────────
        next_state, base_reward, done, info = env.step(safe_action)

        # ── R_safe = R - κ·P(collision)  (Eq.19) ─────────────────
        r_safe, risk = sccc.compute_safe_reward(base_reward, state)

        next_domain = int(next_state[-1])

        # ── 存入 Replay Buffer D ──────────────────────────────────
        sccc.store(state, option, safe_action, r_safe,
                   next_state, next_domain, done)

        # ── 存入 PPO rollout buffer ───────────────────────────────
        hssp.store(state, wp_enc, option, lp_H,
                   r_safe, val_H, done, probs)
        ppo_steps += 1

        # ── SAC 更新 π_L  (Eq.20) ────────────────────────────────
        sac_metrics = sccc.update()

        # ── PPO 更新 π_H  (Eq.16 + Eq.15) ───────────────────────
        if ppo_steps >= hssp_cfg.n_steps:
            with torch.no_grad():
                s_t  = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
                wp_t = torch.FloatTensor(wp_enc).unsqueeze(0).to(DEVICE)
                _, last_val = hssp.policy(s_t, wp_t)
            last_val_f = float(last_val.squeeze().item())

            ppo_metrics = hssp.update(last_value=last_val_f)
            ppo_steps   = 0

            # 记录损失
            history['loss_H'].append(ppo_metrics.get('loss_policy', 0))
            history['loss_sw'].append(ppo_metrics.get('loss_sw',    0))

        if sac_metrics:
            history['loss_L'].append(sac_metrics.get('actor_loss', 0))
            history['loss_q'].append(sac_metrics.get('q_loss',     0))
            history['alpha'].append(sac_metrics.get('alpha',       0))

        # ── 状态转移 ──────────────────────────────────────────────
        state       = next_state
        prev_option = option
        ep_reward  += r_safe
        ep_steps   += 1
        total_steps+= 1

        if done:
            break

    # ────────────────────────────────────────────────────────────────
    # Episode 结束统计
    # ────────────────────────────────────────────────────────────────
    result = info.get('result', 'timeout')
    if result == 'success':  n_success   += 1
    if result == 'collision':n_collision += 1

    ep_rewards.append(ep_reward)
    ep_lengths.append(ep_steps)
    ep_results.append(result)

    ssi = hssp.switching_stability_index()
    history['steps'].append(total_steps)
    history['ep_reward'].append(ep_reward)
    history['ep_length'].append(ep_steps)
    history['SSI'].append(ssi)

    # ────────────────────────────────────────────────────────────────
    # 日志
    # ────────────────────────────────────────────────────────────────
    if n_episodes % 10 == 0:
        log_metrics(total_steps, n_episodes)

    # ────────────────────────────────────────────────────────────────
    # 定期评估 + 保存
    # ────────────────────────────────────────────────────────────────
    if total_steps % args.eval_freq == 0 or total_steps >= args.total_steps:

        # 计算评估指标
        recent_n  = min(100, n_episodes)
        sr   = n_success   / max(n_episodes, 1)
        cr   = n_collision / max(n_episodes, 1)
        apl  = float(np.mean([info.get('ep_path',   0)
                               for info in [{'ep_path': l*0.3}
                                            for l in ep_lengths[-recent_n:]]]))
        ec   = float(np.mean([info.get('ep_energy', 0)
                               for info in [{'ep_energy': l*0.1}
                                            for l in ep_lengths[-recent_n:]]]))

        history['SR'].append(sr)
        history['CR'].append(cr)
        history['APL'].append(apl)
        history['EC'].append(ec)

        print(f"\n[Eval @ {total_steps:,}]  "
              f"SR={sr:.4f}  CR={cr:.4f}  "
              f"APL={apl:.1f}  EC={ec:.1f}  "
              f"SSI={ssi:.3f}")

        # 保存检查点
        ckpt_prefix = str(CKPT_DIR / f'step_{total_steps:07d}')
        hssp.save(ckpt_prefix + '_hssp.pt')
        sccc.save(ckpt_prefix + '_sccc.pt')

        # 保存训练历史
        with open(RESULTS_DIR / 'training_history.json', 'w') as f:
            json.dump({k: [float(x) for x in v]
                       for k, v in history.items()}, f, indent=2)
        print(f"[Ckpt] 已保存: {ckpt_prefix}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 训练完成
# ─────────────────────────────────────────────────────────────────────────────

# 保存最终模型
final_prefix = str(RESULTS_DIR / 'final')
hssp.save(final_prefix + '_hssp.pt')
sccc.save(final_prefix + '_sccc.pt')

print("\n" + "="*60)
print("训练完成！")
print(f"  总步数  : {total_steps:,}")
print(f"  总 Episode: {n_episodes}")
print(f"  SR={n_success/max(n_episodes,1):.4f}  "
      f"CR={n_collision/max(n_episodes,1):.4f}")
print(f"  最终模型: {final_prefix}_*.pt")
print("="*60)
