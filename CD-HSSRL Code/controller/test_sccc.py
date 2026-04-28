#!/usr/bin/env python3
"""
test_sccc.py — SCCC 单元测试
运行: python3 test_sccc.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'controller'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

from sccc import (
    SCCCConfig, LowLevelActor, SoftQNetwork,
    ReplayBuffer, SafetyProjection, SACUpdater,
    SafetyConstrainedContinuousController,
)


def make_state(x=0.0, domain=0, min_us=3.0):
    """构造一个 17 维测试状态。"""
    us = np.array([min_us, min_us+0.1, min_us+0.2, min_us+0.3])
    return np.array([
        x, 0.0, -0.3 if domain==0 else 0.1,   # pos
        0.3, 0.0, 0.0,                          # vel
        0.0,                                    # yaw
        float(np.sqrt((12-x)**2)),              # d_goal
        *us,                                    # ultrasonic
        0.0, 0.0, -9.8,                         # accel
        0.3 if domain==0 else 0.0,              # depth
        float(domain),                          # domain label
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────

def test_networks():
    print("\n[Test 1] 网络前向传播 (Eq.17)")
    cfg    = SCCCConfig(state_dim=17, action_dim=2)
    actor  = LowLevelActor(cfg)
    critic = SoftQNetwork(cfg)

    B  = 8
    s  = torch.randn(B, 17)
    oh = torch.zeros(B, 3); oh[:, 0] = 1.0   # Water option
    a  = torch.randn(B, 2)

    # Actor
    action, log_prob, mean = actor.sample(s, oh)
    print(f"  Actor  action  : {tuple(action.shape)}  "
          f"range=[{action.min():.3f}, {action.max():.3f}]")
    print(f"  Actor  log_prob: {tuple(log_prob.shape)}  "
          f"mean={log_prob.mean():.3f}")
    assert action.shape  == (B, 2)
    assert log_prob.shape == (B, 1)
    assert action.abs().max() <= 1.0 + 1e-5, "tanh 超出 [-1,1]"

    # Critic (twin Q)
    q1, q2 = critic(s, oh, a)
    print(f"  Critic Q1: {tuple(q1.shape)}  Q2: {tuple(q2.shape)}")
    assert q1.shape == (B, 1)
    print("  ✓ 网络测试通过")
    return cfg


def test_safety_projection():
    print("\n[Test 2] 安全投影层 (Eq.18)")
    cfg    = SCCCConfig(collision_thresh=0.4, kappa=1.0)
    safety = SafetyProjection(cfg)

    cases = [
        # (min_us, domain, raw_vx, desc, expect_vx_reduced)
        (3.0, 0, 1.0, "正常水域前进",     False),
        (0.2, 0, 1.0, "极近障碍物",        True),
        (3.0, 1, 1.0, "过渡区限速",        True),
        (3.0, 2, 1.0, "正常陆地前进",      False),
    ]

    all_pass = True
    for min_us, domain, raw_vx, desc, expect_reduced in cases:
        state      = make_state(domain=domain, min_us=min_us)
        raw_action = np.array([raw_vx, 0.0], dtype=np.float32)
        safe       = safety.project(raw_action, state, domain)
        reduced    = safe[0] < raw_vx - 1e-4
        mark       = '✓' if reduced == expect_reduced else '✗'
        print(f"  {mark} {desc:12s}: raw_vx={raw_vx:.2f} → "
              f"safe_vx={safe[0]:.3f}  (限速={'是' if reduced else '否'})")
        if reduced != expect_reduced:
            all_pass = False

    # 风险奖励
    state_safe   = make_state(min_us=3.0)
    state_danger = make_state(min_us=0.15)
    r_s, risk_s  = safety.risk_shaped_reward(1.0, state_safe)
    r_d, risk_d  = safety.risk_shaped_reward(1.0, state_danger)
    print(f"\n  R_safe(安全环境): base=1.0 → {r_s:.4f}  risk={risk_s:.4f}")
    print(f"  R_safe(危险环境): base=1.0 → {r_d:.4f}  risk={risk_d:.4f}")
    assert risk_d > risk_s, "危险环境风险应更高"
    assert all_pass, "安全投影测试失败"
    print("  ✓ 安全投影测试通过")


def test_replay_buffer():
    print("\n[Test 3] Replay Buffer D")
    buf = ReplayBuffer(capacity=1000)

    for i in range(200):
        s  = np.random.randn(17).astype(np.float32)
        ns = np.random.randn(17).astype(np.float32)
        buf.add(s, np.random.randint(0,3), np.random.randn(2).astype(np.float32),
                float(np.random.randn()), ns, np.random.randint(0,3), False)

    batch = buf.sample(64)
    print(f"  Buffer 大小   : {len(buf)}")
    print(f"  Batch states  : {tuple(batch['states'].shape)}")
    print(f"  Batch option_oh: {tuple(batch['option_oh'].shape)}")
    print(f"  Batch actions : {tuple(batch['actions'].shape)}")
    assert batch['states'].shape     == (64, 17)
    assert batch['option_oh'].shape  == (64, 3)
    assert batch['actions'].shape    == (64, 2)
    # one-hot 验证
    assert (batch['option_oh'].sum(dim=-1) == 1).all()
    print("  ✓ Replay Buffer 测试通过")


def test_sac_update():
    print("\n[Test 4] SAC 更新 L_L (Eq.20)")
    cfg    = SCCCConfig(state_dim=17, batch_size=64, learning_starts=50)
    actor  = LowLevelActor(cfg)
    critic = SoftQNetwork(cfg)
    buf    = ReplayBuffer(capacity=10000)
    upd    = SACUpdater(actor, critic, cfg)

    # 填充 buffer
    for _ in range(200):
        buf.add(
            np.random.randn(17).astype(np.float32),
            np.random.randint(0,3),
            np.random.randn(2).astype(np.float32) * 0.5,
            float(np.random.randn()),
            np.random.randn(17).astype(np.float32),
            np.random.randint(0,3),
            False,
        )

    params_before = [p.clone() for p in actor.parameters()]

    t0 = time.perf_counter()
    m  = upd.update(buf)
    elapsed = time.perf_counter() - t0

    params_changed = any(not torch.allclose(pb, pa)
                         for pb, pa in zip(params_before, actor.parameters()))

    print(f"  q_loss     = {m['q_loss']:.4f}")
    print(f"  actor_loss = {m['actor_loss']:.4f}")
    print(f"  alpha      = {m['alpha']:.4f}")
    print(f"  参数更新   : {'✓' if params_changed else '✗'}")
    print(f"  更新耗时   : {elapsed*1000:.1f} ms")
    assert params_changed
    print("  ✓ SAC 更新测试通过")
    return cfg


def test_full_sccc():
    print("\n[Test 5] 完整 SCCC 流程（20步）")
    cfg  = SCCCConfig(state_dim=17, learning_starts=50, batch_size=32)
    sccc = SafetyConstrainedContinuousController(cfg)

    states, actions, rewards, risks = [], [], [], []

    for step in range(100):
        # 模拟机器人从水中向陆地移动
        x      = -12.0 + step * 0.25
        domain = 0 if x < -2.5 else (1 if x < 2.5 else 2)
        state  = make_state(x=x, domain=domain, min_us=2.0 if x > -1 else 3.0)
        option = domain   # 简化：option = domain

        safe_a, raw_a = sccc.select_action(state, option)
        r_safe, risk  = sccc.compute_safe_reward(
            1.0 - 0.01*abs(x-12), state)

        next_x     = x + 0.25
        next_dom   = 0 if next_x < -2.5 else (1 if next_x < 2.5 else 2)
        next_state = make_state(x=next_x, domain=next_dom)
        done       = (step == 99)

        sccc.store(state, option, safe_a, r_safe,
                   next_state, next_dom, done)
        sccc.update()

        states.append(x)
        actions.append(safe_a.copy())
        rewards.append(r_safe)
        risks.append(risk)

    print(f"  总步数        : 100")
    print(f"  SAC 更新次数  : {sccc.updater.n_updates}")
    print(f"  Buffer 大小   : {len(sccc.buffer)}")
    print(f"  平均 safe_vx  : {np.mean([a[0] for a in actions]):.3f}")
    print(f"  平均 risk     : {np.mean(risks):.4f}")
    print(f"  平均 R_safe   : {np.mean(rewards):.3f}")
    assert sccc.updater.n_updates > 0
    print("  ✓ 完整 SCCC 测试通过")
    return states, actions, rewards, risks, sccc


def visualize(states, actions, rewards, risks, sccc, cfg):
    print("\n[Viz] 生成图表...")

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('#F8F9FA')
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    xs     = np.array(states)
    vx_arr = np.array([a[0] for a in actions])
    wz_arr = np.array([a[1] for a in actions])
    rw_arr = np.array(rewards)
    rk_arr = np.array(risks)
    t      = np.arange(len(xs))

    # ── (0,0) 网络结构 ────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis('off'); ax0.set_facecolor('#F8F9FA')
    ax0.set_xlim(0, 10); ax0.set_ylim(0, 8)

    import matplotlib.patches as mp
    def box(x, y, w, h, txt, sub='', fc='#1565C0', fs=9):
        r = mp.FancyBboxPatch((x-w/2,y-h/2),w,h,
            boxstyle='round,pad=0.1',fc=fc,ec='white',lw=1.5,alpha=0.9)
        ax0.add_patch(r)
        ax0.text(x, y+(0.15 if sub else 0), txt, ha='center', va='center',
                 fontsize=fs, fontweight='bold', color='white')
        if sub: ax0.text(x, y-0.25, sub, ha='center', va='center',
                         fontsize=7, color='white', alpha=0.9)
    def arr(x1,y1,x2,y2):
        ax0.annotate('',xy=(x2,y2),xytext=(x1,y1),
            arrowprops=dict(arrowstyle='->',color='#455A64',lw=1.5))

    box(5,7.2,7.5,0.7,'π_L Actor: [state(17)+option_onehot(3)] → Gaussian','LowLevelActor  (Eq.17)','#1565C0')
    box(2.5,5.8,4,0.65,'mean, log_std','→ tanh → action∈[-1,1]²','#37474F',fs=8)
    box(7.5,5.8,4,0.65,'Safety Projection','argmin‖a-aₜ‖  s.t. g≤0  (Eq.18)','#2E7D32',fs=8)
    box(5,4.3,7.5,0.65,'R_safe = R - κ·P(collision|s)   (Eq.19)','Risk-Sensitive Reward','#E65100',fs=8)
    box(2.5,2.8,4,0.65,'Q_θQ: Twin Q-Networks','SoftQNetwork  (Eq.20)','#6A1B9A',fs=8)
    box(7.5,2.8,4,0.65,'SAC Update','α·log π_L - Q → min  (Eq.20)','#00838F',fs=8)
    box(5,1.3,7.5,0.65,'Replay Buffer D','(s, o, a_safe, R_safe, s\', o\', done)','#455A64',fs=8)

    for (x1,y1,x2,y2) in [(5,6.85,2.5,6.12),(5,6.85,7.5,6.12),
                           (7.5,5.47,5,4.62),(2.5,5.47,5,4.62),
                           (5,3.97,2.5,3.12),(5,3.97,7.5,3.12),
                           (2.5,2.47,5,1.62),(7.5,2.47,5,1.62)]:
        arr(x1,y1,x2,y2)
    ax0.text(5,7.85,'SCCC Architecture (Section 3.5)',ha='center',
             fontsize=10,fontweight='bold',color='#263238')

    # ── (0,1) 安全动作 vs 位置 ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(xs, vx_arr, '#1565C0', lw=2, label='vx_safe (前进)')
    ax1.plot(xs, wz_arr, '#E65100', lw=1.5, ls='--', label='wz_safe (转向)')
    ax1.axvspan(-18,-2.5, alpha=0.05, color='dodgerblue')
    ax1.axvspan(-2.5,2.5, alpha=0.05, color='orange')
    ax1.axvspan(2.5,18,   alpha=0.05, color='forestgreen')
    ax1.axvline(-2.5, color='cyan', lw=1.5, ls='--', alpha=0.7)
    ax1.axvline( 2.5, color='cyan', lw=1.5, ls='--', alpha=0.7)
    ax1.set_xlabel('X position (m)', fontsize=9)
    ax1.set_ylabel('Action (normalized)', fontsize=9)
    ax1.set_title('Safe Action vs Position\n(Safety Projection Eq.18)',
                  fontsize=10, fontweight='bold')
    ax1.legend(fontsize=8); ax1.set_ylim(-1.1, 1.1)
    ax1.text(-10, 0.85, 'Water', fontsize=8, color='dodgerblue')
    ax1.text(-0.8, 0.85, 'Trans.', fontsize=8, color='darkorange')
    ax1.text(5, 0.85, 'Land', fontsize=8, color='forestgreen')

    # ── (0,2) 风险 + R_safe ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.fill_between(xs, rk_arr, alpha=0.3, color='#C62828')
    ax2.plot(xs, rk_arr, '#C62828', lw=2, label='P(collision)')
    ax2r = ax2.twinx()
    ax2r.plot(xs, rw_arr, '#1565C0', lw=1.8, ls='-', label='R_safe')
    ax2.set_xlabel('X position (m)', fontsize=9)
    ax2.set_ylabel('Collision Risk', fontsize=9, color='#C62828')
    ax2r.set_ylabel('R_safe', fontsize=9, color='#1565C0')
    ax2.set_title('Risk-Sensitive Reward (Eq.19)\nR_safe = R - κ·P(collision)',
                  fontsize=10, fontweight='bold')
    lines1,labs1 = ax2.get_legend_handles_labels()
    lines2,labs2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labs1+labs2, fontsize=8, loc='upper right')

    # ── (1,0) SAC 损失收敛 ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cfg_t = SCCCConfig(state_dim=17, batch_size=64,
                       learning_starts=50, gradient_steps=1)
    actor_t = LowLevelActor(cfg_t); critic_t = SoftQNetwork(cfg_t)
    buf_t   = ReplayBuffer(5000)
    upd_t   = SACUpdater(actor_t, critic_t, cfg_t)
    for _ in range(500):
        buf_t.add(np.random.randn(17).astype(np.float32),
                  np.random.randint(0,3),
                  np.random.randn(2).astype(np.float32)*0.3,
                  float(np.random.randn()*0.5),
                  np.random.randn(17).astype(np.float32),
                  np.random.randint(0,3), False)

    q_hist, pi_hist, alpha_hist = [], [], []
    for _ in range(40):
        m = upd_t.update(buf_t)
        q_hist.append(m['q_loss'])
        pi_hist.append(m['actor_loss'])
        alpha_hist.append(m['alpha'])

    iters = np.arange(40)
    ax3.plot(iters, q_hist,   '#1565C0', lw=2, label='Q loss (critic)')
    ax3.plot(iters, pi_hist,  '#E65100', lw=2, label='Actor loss L_L')
    ax3r = ax3.twinx()
    ax3r.plot(iters, alpha_hist, '#2E7D32', lw=1.5, ls='--', label='α (entropy coef)')
    ax3r.set_ylabel('α', fontsize=8, color='#2E7D32')
    ax3.set_xlabel('SAC Update Iteration', fontsize=9)
    ax3.set_ylabel('Loss', fontsize=9)
    ax3.set_title('SAC Loss Curves (Eq.20)\nL_L = E[α·log π_L - Q]',
                  fontsize=10, fontweight='bold')
    lines1,labs1 = ax3.get_legend_handles_labels()
    lines2,labs2 = ax3r.get_legend_handles_labels()
    ax3.legend(lines1+lines2, labs1+labs2, fontsize=8)

    # ── (1,1) 安全投影效果（kappa 敏感性）────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    us_range = np.linspace(0.05, 1.5, 100)
    for kappa, c, lbl in [(0.5,'#90CAF9','κ=0.5'),
                           (1.0,'#1565C0','κ=1.0 (paper)'),
                           (2.0,'#C62828','κ=2.0')]:
        cfg_k = SCCCConfig(kappa=kappa, collision_thresh=0.4)
        sp_k  = SafetyProjection(cfg_k)
        risks = []
        for d in us_range:
            s = make_state(min_us=d); risks.append(sp_k.compute_collision_risk(s))
        r_safe_vals = [1.0 - kappa*r for r in risks]
        ax4.plot(us_range, r_safe_vals, color=c, lw=2, label=lbl)

    ax4.axvline(0.4, color='gray', ls='--', lw=1.5, alpha=0.6,
                label='Collision thresh')
    ax4.set_xlabel('Min Obstacle Distance (m)', fontsize=9)
    ax4.set_ylabel('R_safe  (base R=1.0)', fontsize=9)
    ax4.set_title('κ Sensitivity — Risk Reward (Eq.19)',
                  fontsize=10, fontweight='bold')
    ax4.legend(fontsize=8); ax4.set_ylim(-1.5, 1.1)

    # ── (1,2) Buffer 大小 vs 更新次数 ────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    buf_sizes   = [50, 100, 200, 500, 1000, 2000]
    update_times= []
    for bs in buf_sizes:
        cfg_b  = SCCCConfig(batch_size=64, learning_starts=40)
        a_b    = LowLevelActor(cfg_b); c_b = SoftQNetwork(cfg_b)
        buf_b  = ReplayBuffer(5000)
        upd_b  = SACUpdater(a_b, c_b, cfg_b)
        for _ in range(bs):
            buf_b.add(np.random.randn(17).astype(np.float32), 0,
                      np.random.randn(2).astype(np.float32),
                      0.0, np.random.randn(17).astype(np.float32), 0, False)
        t0 = time.perf_counter()
        for _ in range(10):
            if len(buf_b) >= cfg_b.batch_size:
                upd_b.update(buf_b)
        update_times.append((time.perf_counter()-t0)/10*1000)

    ax5.bar(range(len(buf_sizes)), update_times, color='#1565C0', alpha=0.8)
    ax5.set_xticks(range(len(buf_sizes)))
    ax5.set_xticklabels([str(b) for b in buf_sizes])
    ax5.set_xlabel('Buffer Size', fontsize=9)
    ax5.set_ylabel('Update Time (ms)', fontsize=9)
    ax5.set_title('SAC Update Time vs Buffer Size',
                  fontsize=10, fontweight='bold')
    for i, v in enumerate(update_times):
        ax5.text(i, v+0.1, f'{v:.1f}', ha='center', fontsize=8)

    fig.suptitle('SCCC: Safety-Constrained Continuous Controller — '
                 'Section 3.5, Eq.(17)–(20)',
                 fontsize=13, fontweight='bold', y=1.01)

    out = '/mnt/user-data/outputs/sccc_result.png'
    plt.savefig(out, dpi=170, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [✓] 图表已保存: {out}")


def main():
    print("="*55)
    print("SCCC 单元测试")
    print("="*55)

    cfg = test_networks()
    test_safety_projection()
    test_replay_buffer()
    test_sac_update()
    states, actions, rewards, risks, sccc = test_full_sccc()
    visualize(states, actions, rewards, risks, sccc, cfg)

    print("\n" + "="*55)
    print("全部测试通过 ✓")
    print("="*55)


if __name__ == '__main__':
    main()
