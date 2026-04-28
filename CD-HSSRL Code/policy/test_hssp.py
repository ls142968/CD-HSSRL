"""
HSSP 单元测试 + 可视化
运行: python3 test_hssp.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'planner'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Tuple
import time

from hssp import (
    HSSPConfig, HighLevelPolicyNet, TerminationCondition,
    SwitchingRegularizer, RolloutBuffer, PPOUpdater,
    HierarchicalSafeSwitchingPolicy,
    encode_waypoint, infer_domain_label,
    OPTION_WATER, OPTION_TRANSITION, OPTION_LAND, OPTION_NAMES, N_OPTIONS,
)


# ─────────────────────────────────────────────────────────────────────────────
# 模拟环境（不依赖 Gazebo）
# ─────────────────────────────────────────────────────────────────────────────

class MockAmphibiousEnv:
    """
    最小化仿真环境：模拟机器人从水域→过渡区→陆地的导航过程。
    dim=25: GPS(3)+IMU_accel(3)+IMU_gyro(3)+yaw+d_goal+LiDAR(8)+US(4)+depth+domain
    """
    STATE_DIM = 25
    GOAL      = np.array([14.0, 0.0])
    START     = np.array([-14.0, 0.0, -0.3])   # [x, y, z]

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.pos     = self.START.copy().astype(np.float32)   # [x,y,z]
        self.vel     = np.zeros(3, dtype=np.float32)
        self.yaw     = 0.0
        self.t       = 0
        self.done    = False
        self.collision = False
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        构建 25 维状态向量（对齐五传感器配置）。
        [0:3]  GPS    [3:6]  IMU_accel  [6:9]  IMU_gyro
        [9]    yaw    [10]   d_goal
        [11:19] LiDAR 8扇区（障碍物）
        [19:23] 超声波 4路（水底深度）
        [23]   深度传感器（水中深度）  [24] 域标签
        """
        x, y, z  = self.pos
        domain   = infer_domain_label(z, self.pos[0])
        d_goal   = float(np.linalg.norm(self.pos[:2] - self.GOAL))
        # 1. GPS（含噪声）
        gps      = self.pos + self.rng.normal(0, 0.3, 3).astype(np.float32)
        # 2. IMU 线加速度（含噪声）
        imu_acc  = np.array([0.3, 0.0, 0.0], np.float32) + self.rng.normal(0, 0.05, 3)
        # 3. IMU 角速度（含噪声）
        imu_gyro = self.rng.normal(0, 0.01, 3).astype(np.float32)
        # 4. LiDAR 8扇区（障碍物感知，模拟）
        lidar    = np.clip(5.0 + self.rng.normal(0, 0.3, 8), 0.3, 10.0).astype(np.float32)
        # 5a. 超声波 4路（水底深度感知）
        us_base  = 2.0 if domain == 0 else 5.0
        ultrasonic = np.clip(us_base + self.rng.normal(0, 0.1, 4), 0.1, 5.0).astype(np.float32)
        # 5b. 深度传感器（机器人在水中的深度）
        water_depth = float(max(0.0, -z) + self.rng.normal(0, 0.01))
        return np.concatenate([
            gps,                      # [0:3]
            imu_acc,                  # [3:6]
            imu_gyro,                 # [6:9]
            [self.yaw],               # [9]
            [d_goal],                 # [10]
            lidar,                    # [11:19]
            ultrasonic,               # [19:23]
            [water_depth],            # [23]
            [float(domain)],          # [24]
        ]).astype(np.float32)

    def step(self, option: int, dt: float = 0.05
             ) -> Tuple['np.ndarray', float, bool, dict]:
        x, y, z = self.pos

        # 按 option 决定运动模式
        if option == OPTION_WATER:
            vx = 0.4 + self.rng.normal(0, 0.05)
            vy = self.rng.normal(0, 0.05)
        elif option == OPTION_TRANSITION:
            vx = 0.2 + self.rng.normal(0, 0.03)
            vy = self.rng.normal(0, 0.03)
        else:   # LAND
            vx = 0.5 + self.rng.normal(0, 0.06)
            vy = self.rng.normal(0, 0.06)

        self.vel[:2] = [vx, vy]
        self.pos[0] += vx * dt
        self.pos[1] += vy * dt

        # z 跟随 x 变化（坡面）
        px = self.pos[0]
        if   px < -2.5: self.pos[2] = -0.3
        elif px <  2.5: self.pos[2] = -0.3 + 0.4 * (px + 2.5) / 5.0
        else:           self.pos[2] =  0.1

        self.t += 1
        d_goal  = float(np.linalg.norm(self.pos[:2] - self.GOAL))
        reached = d_goal < 0.8
        timeout = self.t >= 500

        # 基础奖励
        if reached:
            reward = 100.0; done = True
        elif timeout:
            reward = -10.0; done = True
        else:
            reward = -0.05 - 0.01 * d_goal; done = False

        self.done = done
        return self._get_state(), reward, done, {'d_goal': d_goal}



# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — 网络前向传播  (Eq.13)
# ─────────────────────────────────────────────────────────────────────────────

def test_network():
    print("\n[Test 1] 网络前向传播 (Eq.13)")
    cfg = HSSPConfig(state_dim=25, hidden_size=256)
    net = HighLevelPolicyNet(cfg)

    B = 8
    s  = torch.randn(B, 25)
    wp = torch.randn(B, 4)

    probs, value = net(s, wp)
    print(f"  输入  : state {tuple(s.shape)}  waypoint {tuple(wp.shape)}")
    print(f"  probs : {tuple(probs.shape)}  sum={probs.sum(-1).mean():.4f}  "
          f"min={probs.min():.4f}  max={probs.max():.4f}")
    print(f"  value : {tuple(value.shape)}")

    assert probs.shape  == (B, N_OPTIONS)
    assert value.shape  == (B, 1)
    assert torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5), "softmax sum≠1"

    # 单样本
    dist, val = net.get_dist(s[:1], wp[:1])
    o   = dist.sample()
    lp  = dist.log_prob(o)
    print(f"  sample option={o.item()}  log_prob={lp.item():.3f}")
    print("  ✓ 网络测试通过")
    return cfg, net


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — 终止条件  (Eq.14)
# ─────────────────────────────────────────────────────────────────────────────

def test_termination():
    print("\n[Test 2] 终止条件 β(o_t|s_t) (Eq.14)")
    term = TerminationCondition(waypoint_radius=1.5)

    cases = [
        (np.array([-10.0, 0.0]), np.array([5.0, 0.0]),  OPTION_WATER,      False, "水域行进"),
        (np.array([-0.5,  0.0]), np.array([5.0, 0.0]),  OPTION_TRANSITION,  True, "过渡区→重选"),
        (np.array([ 5.0,  0.0]), np.array([5.3, 0.0]),  OPTION_LAND,        True, "到达航点"),
        (np.array([ 8.0,  2.0]), np.array([5.0, 0.0]),  OPTION_LAND,       False, "陆地行进"),
    ]

    all_pass = True
    for pos, wp, domain, expected, desc in cases:
        result = term(pos, wp, domain)
        mark   = "✓" if result == expected else "✗"
        print(f"  {mark} {desc:12s}: β={int(result)}  (期望={int(expected)})")
        if result != expected:
            all_pass = False

    assert all_pass, "终止条件测试失败"
    print("  ✓ 终止条件测试通过")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — 切换正则化损失  (Eq.15)
# ─────────────────────────────────────────────────────────────────────────────

def test_switching_regularizer():
    print("\n[Test 3] 切换正则化损失 L_sw (Eq.15)")
    reg = SwitchingRegularizer(lambda_sw=0.05)

    # 情形1：分布完全相同 → L_sw ≈ 0
    p = torch.softmax(torch.randn(4, 3), dim=-1)
    lsw_same = reg(p, p.clone())
    print(f"  分布相同时  L_sw = {lsw_same.item():.6f}  (期望≈0)")

    # 情形2：分布差异大 → L_sw 较大
    p1 = torch.tensor([[1.0, 0.0, 0.0]] * 4)   # 全是 Water
    p2 = torch.tensor([[0.0, 0.0, 1.0]] * 4)   # 全是 Land
    lsw_diff = reg(p1, p2)
    print(f"  分布差异大  L_sw = {lsw_diff.item():.4f}  (期望>0)")

    # 情形3：λ_sw=0 时损失为0
    reg0 = SwitchingRegularizer(lambda_sw=0.0)
    lsw0 = reg0(p1, p2)
    print(f"  λ_sw=0 时   L_sw = {lsw0.item():.6f}  (期望=0)")

    assert lsw_same.item() < 1e-5
    assert lsw_diff.item() > 0.01
    assert lsw0.item()     < 1e-8
    print("  ✓ L_sw 测试通过")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — GAE 优势估计
# ─────────────────────────────────────────────────────────────────────────────

def test_gae():
    print("\n[Test 4] GAE 优势估计")
    buf = RolloutBuffer()
    T   = 20
    for t in range(T):
        buf.add(
            state    = np.random.randn(25).astype(np.float32),
            waypoint = np.random.randn(4).astype(np.float32),
            option   = np.random.randint(0, 3),
            log_prob = -1.2,
            reward   = float(np.random.randn()),
            value    = float(np.random.randn()),
            done     = (t == T - 1),
            probs    = np.array([0.33, 0.33, 0.34], dtype=np.float32),
        )
    buf.compute_gae(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    print(f"  T={T} 步  advantage: mean={buf.advantages.mean():.3f}  "
          f"std={buf.advantages.std():.3f}")
    print(f"  returns:   mean={buf.returns_.mean():.3f}  "
          f"std={buf.returns_.std():.3f}")
    assert buf.advantages.shape == (T,)
    print("  ✓ GAE 测试通过")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — PPO 更新  (Eq.16)
# ─────────────────────────────────────────────────────────────────────────────

def test_ppo_update():
    print("\n[Test 5] PPO 更新 L_H (Eq.16) + L_sw (Eq.15)")
    cfg     = HSSPConfig(state_dim=25, n_steps=128, batch_size=32, n_epochs=4)
    policy  = HighLevelPolicyNet(cfg)
    updater = PPOUpdater(policy, cfg, device='cpu')

    # 填充 buffer
    buf = RolloutBuffer()
    for _ in range(128):
        buf.add(
            state    = np.random.randn(25).astype(np.float32),
            waypoint = np.random.randn(4).astype(np.float32),
            option   = np.random.randint(0, 3),
            log_prob = float(np.log(1/3)),
            reward   = float(np.random.randn() * 0.1),
            value    = float(np.random.randn() * 0.1),
            done     = False,
            probs    = np.array([1/3]*3, dtype=np.float32),
        )
    buf.compute_gae(0.0, gamma=0.99, gae_lambda=0.95)

    # 记录更新前参数
    params_before = [p.clone() for p in policy.parameters()]

    t0 = time.perf_counter()
    metrics = updater.update(buf)
    elapsed = time.perf_counter() - t0

    # 确认参数发生了变化
    params_changed = any(
        not torch.allclose(pb, pa)
        for pb, pa in zip(params_before, policy.parameters())
    )

    print(f"  loss_total  = {metrics['loss_total']:.4f}")
    print(f"  loss_policy = {metrics['loss_policy']:.4f}")
    print(f"  loss_value  = {metrics['loss_value']:.4f}")
    print(f"  loss_sw     = {metrics['loss_sw']:.6f}")
    print(f"  clip_frac   = {metrics['clip_frac']:.4f}")
    print(f"  参数更新    : {'✓ 是' if params_changed else '✗ 否'}")
    print(f"  更新耗时    : {elapsed*1000:.1f} ms")

    assert params_changed, "PPO 更新后参数应改变"
    print("  ✓ PPO 更新测试通过")
    return cfg, policy


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — 完整 Episode 仿真（不依赖 Gazebo）
# ─────────────────────────────────────────────────────────────────────────────

def test_full_episode(cfg):
    print("\n[Test 6] 完整 Episode 仿真")
    hssp = HierarchicalSafeSwitchingPolicy(cfg, device='cpu')
    env  = MockAmphibiousEnv(seed=42)

    # 模拟航点序列
    waypoints = [
        np.array([-10.0, 0.0]),
        np.array([ -5.0, 0.0]),
        np.array([  0.0, 0.0]),
        np.array([  5.0, 0.0]),
        np.array([ 10.0, 0.0]),
        np.array([ 14.0, 0.0]),
    ]
    wp_idx = 0

    state  = env.reset()
    hssp.reset_episode()
    ep_reward = 0.0
    option_log = []   # 记录每步的 option（用于可视化）
    pos_log    = []

    for step in range(300):
        wp = waypoints[min(wp_idx, len(waypoints)-1)]
        wp_enc      = encode_waypoint(state[:2], wp)
        domain      = int(state[-1])
        robot_pos   = state[:2]

        # β 判断：是否需要重选 option  (Eq.14)
        need_reselect = (step == 0 or
                         hssp.should_reselect(robot_pos, wp, domain))

        if need_reselect:
            option, lp, val, probs = hssp.select_option(
                state, wp_enc, domain, deterministic=False)
        else:
            option = hssp.current_option
            with torch.no_grad():
                s_t  = torch.FloatTensor(state).unsqueeze(0)
                wp_t = torch.FloatTensor(wp_enc).unsqueeze(0)
                d, v = hssp.policy.get_dist(s_t, wp_t)
                lp   = float(d.log_prob(torch.tensor([option])).item())
                val  = float(v.item())
                probs= d.probs.squeeze(0).cpu().numpy()

        # 环境 step
        next_state, reward, done, info = env.step(option)
        ep_reward += reward

        # 存入 buffer
        hssp.store(state, wp_enc, option, lp, reward, val, done, probs)

        option_log.append(option)
        pos_log.append(env.pos[:2].copy())

        # 航点推进
        if np.linalg.norm(robot_pos - wp) < 1.5 and wp_idx < len(waypoints)-1:
            wp_idx += 1

        # PPO 更新（满 n_steps）
        if hssp.ready_to_update():
            last_val = 0.0 if done else val
            metrics  = hssp.update(last_val)

        state = next_state
        if done:
            break

    ssi = hssp.switching_stability_index()
    print(f"  共 {step+1} 步  |  总奖励: {ep_reward:.1f}")
    print(f"  切换次数 N_sw = {hssp.switch_count}")
    print(f"  SSI = 1 - {hssp.switch_count}/{hssp.step_count} = {ssi:.4f}  (Eq.29)")
    print(f"  Option 分布: Water={option_log.count(0)}  "
          f"Trans={option_log.count(1)}  Land={option_log.count(2)}")
    print("  ✓ Episode 仿真测试通过")
    return option_log, pos_log, hssp


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def visualize(option_log, pos_log, hssp, cfg):
    print("\n[Viz] 生成图表...")

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#F8F9FA')
    gs  = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

    OPTION_COLORS = ['#1565C0', '#FF8F00', '#2E7D32']
    T = len(option_log)
    t = np.arange(T)

    # ── (0,0) 网络结构示意 ────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis('off')
    ax0.set_facecolor('#F8F9FA')

    def box(ax, x, y, w, h, txt, sub='', fc='#1565C0', fs=9):
        r = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle='round,pad=0.1', fc=fc, ec='white', lw=1.5, alpha=0.88)
        ax.add_patch(r)
        ax.text(x, y+(0.12 if sub else 0), txt, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='white')
        if sub:
            ax.text(x, y-0.22, sub, ha='center', va='center',
                    fontsize=7, color='white', alpha=0.9)

    def arr(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
            arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.5))

    ax0.set_xlim(0,10); ax0.set_ylim(0,8)
    box(ax0, 2.0, 6.5, 3.2, 0.8, 'Input', 'state(25) + waypoint(4)', '#455A64')
    box(ax0, 5.0, 6.5, 3.0, 0.8, 'Trunk', '2×Linear(256)+ReLU',     '#37474F')
    box(ax0, 2.5, 4.8, 3.5, 0.8, 'Option Head', 'Linear→Softmax(3)', '#1565C0')
    box(ax0, 7.0, 4.8, 2.5, 0.8, 'Value Head', 'Linear(1)',          '#6A1B9A')
    box(ax0, 1.2, 3.1, 2.0, 0.7, 'Water',      'o=0', '#1565C0', fs=8)
    box(ax0, 3.3, 3.1, 2.0, 0.7, 'Transition', 'o=1', '#FF8F00', fs=8)
    box(ax0, 5.4, 3.1, 2.0, 0.7, 'Land',       'o=2', '#2E7D32', fs=8)
    box(ax0, 7.0, 3.1, 2.5, 0.7, 'V(sₜ)',      '',    '#6A1B9A', fs=8)

    arr(ax0,3.6,6.1, 5.0,6.1); arr(ax0,5.0,6.1, 2.5,5.2); arr(ax0,5.0,6.1, 7.0,5.2)
    for xi in [1.2,3.3,5.4]: arr(ax0,2.5,4.4, xi,3.5)
    arr(ax0,7.0,4.4, 7.0,3.5)

    ax0.text(5, 7.6, 'π_H Network (Eq.13)', ha='center', fontsize=10,
             fontweight='bold', color='#263238')
    ax0.text(5, 7.1, 'MLP·256·256·Softmax', ha='center', fontsize=8,
             color='#455A64')

    # 终止条件框
    ax0.text(5, 2.5, 'Termination β(oₜ|sₜ)  (Eq.14)', ha='center',
             fontsize=9, fontweight='bold', color='#B71C1C')
    ax0.text(5, 2.0,
             'β=1 if s∈Sₜ (transition) or waypoint_reached',
             ha='center', fontsize=7.5, color='#455A64')
    ax0.text(5, 1.5,
             'β=0 otherwise  →  keep current option',
             ha='center', fontsize=7.5, color='#455A64')

    # L_sw
    ax0.text(5, 0.9,
             'L_sw = λ_sw·‖πH(·|sₜ)−πH(·|sₜ₋₁)‖²  (Eq.15)',
             ha='center', fontsize=8, color='#E65100',
             bbox=dict(boxstyle='round,pad=0.3', fc='#FFF3E0', ec='#E65100'))

    # ── (0,1) 切换序列  ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    colors_t = [OPTION_COLORS[o] for o in option_log]
    ax1.scatter(t, option_log, c=colors_t, s=12, zorder=3, alpha=0.8)
    ax1.plot(t, option_log, color='#90A4AE', lw=0.8, alpha=0.5, zorder=2)

    # 涂色背景
    for y0, y1, c in [(-0.4,0.5,'#E3F2FD'),(0.5,1.5,'#FFF8E1'),(1.5,2.4,'#E8F5E9')]:
        ax1.axhspan(y0, y1, alpha=0.3, color=c)

    ax1.set_yticks([0,1,2])
    ax1.set_yticklabels(['Water\n(o=0)', 'Transition\n(o=1)', 'Land\n(o=2)'], fontsize=8)
    ax1.set_xlabel('Time Step', fontsize=9)
    ax1.set_title('Option Switching Sequence\n(β-triggered reselection)', fontsize=10,
                  fontweight='bold')
    ax1.set_ylim(-0.5, 2.5); ax1.set_xlim(0, T)

    # 标注切换事件
    switches = [i for i in range(1,len(option_log))
                if option_log[i] != option_log[i-1]]
    for sw in switches[:10]:
        ax1.axvline(sw, color='red', lw=0.8, alpha=0.4, ls='--')

    ssi = hssp.switching_stability_index()
    ax1.text(0.97, 0.97,
             f'N_sw={hssp.switch_count}\nSSI={ssi:.3f}',
             transform=ax1.transAxes, ha='right', va='top', fontsize=9,
             color='#1565C0',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

    legend_patches = [
        mpatches.Patch(color=OPTION_COLORS[i], label=f'{OPTION_NAMES[i]} (o={i})')
        for i in range(N_OPTIONS)
    ]
    ax1.legend(handles=legend_patches, fontsize=8, loc='upper left')

    # ── (0,2) Option 概率随时间变化 ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    # 重新跑一次收集完整 probs
    cfg2 = HSSPConfig(state_dim=25, n_steps=512)
    hssp2 = HierarchicalSafeSwitchingPolicy(cfg2, device='cpu')
    env2  = MockAmphibiousEnv(seed=7)
    state = env2.reset(); hssp2.reset_episode()
    waypoints = [np.array([-10+i*4.0, 0.0]) for i in range(7)]
    wp_idx = 0
    probs_log = []
    for step in range(min(T, 200)):
        wp     = waypoints[min(wp_idx, len(waypoints)-1)]
        wp_enc = encode_waypoint(state[:2], wp)
        domain = int(state[-1])
        _, _, _, probs = hssp2.select_option(state, wp_enc, domain)
        probs_log.append(probs.copy())
        state, _, done, _ = env2.step(int(np.argmax(probs)))
        if np.linalg.norm(state[:2] - wp) < 1.5 and wp_idx < len(waypoints)-1:
            wp_idx += 1
        if done: break

    probs_arr = np.array(probs_log)
    T2 = len(probs_log)
    t2 = np.arange(T2)
    for i, (name, c) in enumerate(zip(OPTION_NAMES, OPTION_COLORS)):
        ax2.fill_between(t2, probs_arr[:,i], alpha=0.35, color=c)
        ax2.plot(t2, probs_arr[:,i], color=c, lw=1.5, label=f'P(o={i}|{name})')

    ax2.set_xlabel('Time Step', fontsize=9)
    ax2.set_ylabel('Option Probability', fontsize=9)
    ax2.set_title('π_H(o|sₜ) Distribution over Time\n(Eq.13 softmax output)',
                  fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 1); ax2.set_xlim(0, T2)
    ax2.legend(fontsize=8)
    ax2.axhline(1/3, color='gray', ls=':', lw=1, alpha=0.5, label='Uniform')

    # ── (1,0) 轨迹 + option 颜色 ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    if len(pos_log) > 1:
        pos_arr = np.array(pos_log)
        for i in range(len(pos_log)-1):
            c = OPTION_COLORS[option_log[i]]
            ax3.plot(pos_arr[i:i+2, 0], pos_arr[i:i+2, 1],
                     color=c, lw=2.2, solid_capstyle='round')

    ax3.axvline(-2.5, color='cyan',  lw=1.5, ls='--', alpha=0.7, label='Shore boundary')
    ax3.axvline( 2.5, color='cyan',  lw=1.5, ls='--', alpha=0.7)
    ax3.axvspan(-15, -2.5, alpha=0.06, color='dodgerblue')
    ax3.axvspan(-2.5,2.5,  alpha=0.06, color='orange')
    ax3.axvspan( 2.5, 15,  alpha=0.06, color='forestgreen')
    ax3.plot(-14, 0, 'go', ms=10, zorder=5, label='Start')
    ax3.plot( 14, 0, 'r*', ms=12, zorder=5, label='Goal')
    ax3.set_xlabel('X (m)', fontsize=9); ax3.set_ylabel('Y (m)', fontsize=9)
    ax3.set_title('Robot Trajectory\n(color = motion option)', fontsize=10,
                  fontweight='bold')
    ax3.legend(handles=legend_patches+[
        mpatches.Patch(color='none', label='Start:Go  Goal:★')], fontsize=7)
    ax3.set_xlim(-16, 16); ax3.set_ylim(-4, 4)

    # ── (1,1) L_sw 演示曲线 ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    reg = SwitchingRegularizer(lambda_sw=0.05)
    # 模拟不同 λ_sw 下的 L_sw 值
    steps_ = np.arange(50)
    for lsw, c, lbl in [(0.0,'#90A4AE','λ=0.00'),
                         (0.05,'#1565C0','λ=0.05 (paper)'),
                         (0.2, '#E65100','λ=0.20'),
                         (0.5, '#C62828','λ=0.50')]:
        vals = []
        p_prev = torch.tensor([[1/3,1/3,1/3]])
        for _ in steps_:
            noise  = torch.randn(1,3)*0.3
            p_curr = torch.softmax(torch.log(p_prev+1e-6)+noise, dim=-1)
            lv     = SwitchingRegularizer(lsw)(p_curr, p_prev).item()
            vals.append(lv)
            p_prev = p_curr.detach()
        ax4.plot(steps_, vals, color=c, lw=1.8, label=lbl, alpha=0.9)

    ax4.set_xlabel('Step', fontsize=9)
    ax4.set_ylabel('L_sw value', fontsize=9)
    ax4.set_title('Switching Regularization L_sw (Eq.15)\n'
                  'λ_sw controls stability penalty strength',
                  fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.set_ylim(bottom=0)

    # ── (1,2) PPO 损失收敛曲线 ───────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    cfg_t  = HSSPConfig(state_dim=25, n_steps=256, batch_size=64,
                        n_epochs=4, lr=3e-4)
    pol_t  = HighLevelPolicyNet(cfg_t)
    upd_t  = PPOUpdater(pol_t, cfg_t)

    loss_hist = {'policy':[], 'value':[], 'sw':[], 'entropy':[]}
    for iteration in range(20):
        buf = RolloutBuffer()
        for _ in range(256):
            buf.add(
                state    = np.random.randn(25).astype(np.float32),
                waypoint = np.random.randn(4).astype(np.float32),
                option   = np.random.randint(0,3),
                log_prob = float(np.log(1/3)),
                reward   = float(np.random.randn()*0.5 - 0.1),
                value    = float(np.random.randn()*0.2),
                done     = False,
                probs    = np.array([1/3]*3, np.float32),
            )
        buf.compute_gae(0.0, cfg_t.gamma, cfg_t.gae_lambda)
        m = upd_t.update(buf)
        loss_hist['policy'].append(m['loss_policy'])
        loss_hist['value'].append(m['loss_value'])
        loss_hist['sw'].append(m['loss_sw'])
        loss_hist['entropy'].append(m['loss_entropy'])

    iters = np.arange(20)
    ax5.plot(iters, loss_hist['policy'],  '#1565C0', lw=2, label='L_H (policy, Eq.16)')
    ax5.plot(iters, loss_hist['value'],   '#2E7D32', lw=2, label='L_value')
    ax5.plot(iters, loss_hist['sw'],      '#E65100', lw=2, label='L_sw (Eq.15)')
    ax5r = ax5.twinx()
    ax5r.plot(iters, loss_hist['entropy'], '#9C27B0', lw=1.5, ls='--', label='Entropy')
    ax5r.set_ylabel('Entropy', fontsize=8, color='#9C27B0')
    ax5r.tick_params(colors='#9C27B0', labelsize=7)

    ax5.set_xlabel('PPO Update Iteration', fontsize=9)
    ax5.set_ylabel('Loss', fontsize=9)
    ax5.set_title('PPO Loss Curves (Eq.16)\nL_total = L_H + L_value + L_sw',
                  fontsize=10, fontweight='bold')
    lines1, labs1 = ax5.get_legend_handles_labels()
    lines2, labs2 = ax5r.get_legend_handles_labels()
    ax5.legend(lines1+lines2, labs1+labs2, fontsize=8, loc='upper right')

    # ── 总标题 ────────────────────────────────────────────────────────
    fig.suptitle(
        'HSSP: Hierarchical Safe Switching Policy — Section 3.4, Eq.(13)–(16)',
        fontsize=13, fontweight='bold', y=1.01)

    out = '/mnt/user-data/outputs/hssp_result.png'
    plt.savefig(out, dpi=170, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [✓] 图表已保存: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HSSP 单元测试")
    print("=" * 60)

    cfg, net   = test_network()
    test_termination()
    test_switching_regularizer()
    test_gae()
    cfg, policy = test_ppo_update()
    option_log, pos_log, hssp = test_full_episode(cfg)
    visualize(option_log, pos_log, hssp, cfg)

    print("\n" + "=" * 60)
    print("全部测试通过 ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
