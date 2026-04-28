"""
run_baselines.py — 全部 13 个基线方法的统一测试 + 评估

用法：
  python3 run_baselines.py                # 测试所有方法
  python3 run_baselines.py --method IPPO  # 只测试单个方法
  python3 run_baselines.py --steps 5000   # 指定训练步数
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'env'))
sys.path.insert(0, str(ROOT / 'baselines'))

from amphibious_sim_env import AmphibiousSimEnv, SimConfig, SimObstacle
from baseline_base import STATE_DIM, ACTION_DIM

# ── 方法注册表 ────────────────────────────────────────────────────────────────
def build_agent(name: str, device: str = 'cpu'):
    from baselines_01 import IPPO, DDQN, HEAPPO
    from baselines_02 import IMTCMO, APFDQN, IDDPG
    from baselines_03 import MORLBased, RLCA, APFD3QNPER, CLPPOGIC
    from baselines_04 import BarrierNet, PHDRLAgent, MPDQL

    registry = {
        'IPPO':        lambda: IPPO(state_dim=STATE_DIM, n_steps=256,
                                    batch_size=32, n_epochs=4, device=device),
        'DDQN':        lambda: DDQN(state_dim=STATE_DIM, buffer_cap=10000,
                                    learn_starts=200, device=device),
        'HEA-PPO':     lambda: HEAPPO(state_dim=STATE_DIM, n_steps=256,
                                      batch_size=32, n_epochs=4, device=device),
        'IMTCMO':      lambda: IMTCMO(state_dim=STATE_DIM, n_steps=256,
                                      batch_size=32, n_epochs=4, device=device),
        'APF-DQN':     lambda: APFDQN(state_dim=STATE_DIM, buffer_cap=10000,
                                      learn_starts=200, device=device),
        'I-DDPG':      lambda: IDDPG(state_dim=STATE_DIM, buffer_cap=10000,
                                     learn_starts=200, device=device),
        'MORL-based':  lambda: MORLBased(state_dim=STATE_DIM, n_steps=256,
                                          batch_size=32, n_epochs=4, device=device),
        'RLCA':        lambda: RLCA(state_dim=STATE_DIM, n_steps=256,
                                    batch_size=32, n_epochs=4, device=device),
        'APF-D3QNPER': lambda: APFD3QNPER(state_dim=STATE_DIM, buffer_cap=10000,
                                            learn_starts=200, device=device),
        'CLPPO-GIC':   lambda: CLPPOGIC(state_dim=STATE_DIM, n_steps=256,
                                          batch_size=32, n_epochs=4, device=device),
        'BarrierNet':  lambda: BarrierNet(state_dim=STATE_DIM, n_steps=256,
                                           batch_size=32, n_epochs=4, device=device),
        'pH-DRL':      lambda: PHDRLAgent(state_dim=STATE_DIM, n_steps=256,
                                           batch_size=32, n_epochs=4, device=device),
        'MP-DQL':      lambda: MPDQL(state_dim=STATE_DIM, buffer_cap=10000,
                                      learn_starts=200, device=device),
    }
    return registry[name]()


# ── 判断方法类型 ──────────────────────────────────────────────────────────────
PPO_METHODS = {'IPPO','HEA-PPO','IMTCMO','MORL-based','RLCA','CLPPO-GIC',
               'BarrierNet','pH-DRL'}
DQN_METHODS = {'DDQN','APF-DQN','APF-D3QNPER','MP-DQL'}
DDG_METHODS = {'I-DDPG'}


def make_env(seed=42):
    obs = [SimObstacle(-10.,4.,1.0), SimObstacle(-6.,-3.,0.8),
           SimObstacle(-3.,2.5,0.6), SimObstacle(5.,3.5,0.9),
           SimObstacle(9.,-3.,1.1)]
    return AmphibiousSimEnv(
        SimConfig(start_pos=np.array([-12.,0.,-0.3]),
                  goal_pos=np.array([12.,0.]),
                  max_steps=300, obstacles=obs), seed=seed)


def run_single(agent_name: str, total_steps: int,
               n_eval_ep: int = 20, device: str = 'cpu',
               seed: int = 42) -> Dict:
    """训练单个 baseline 方法 total_steps 步并评估。"""
    np.random.seed(seed); torch.manual_seed(seed)

    env      = make_env(seed)
    eval_env = make_env(seed + 1)
    agent    = build_agent(agent_name, device)

    is_ppo = agent_name in PPO_METHODS
    is_dqn = agent_name in DQN_METHODS
    is_ddg = agent_name in DDG_METHODS

    # 从 env 获取 PPO 所需辅助量
    def _get_logp_val(state, action):
        if hasattr(agent, 'get_log_prob_value'):
            return agent.get_log_prob_value(state, action)
        return 0., 0.

    steps = 0; n_ep = 0; n_suc = 0; n_col = 0
    ep_rewards:  List[float] = []
    ep_lengths:  List[int]   = []
    t0 = time.time()

    print(f"  [{agent_name}] Training {total_steps:,} steps...")

    while steps < total_steps:
        state = env.reset()
        if hasattr(agent, 'reset_episode'): agent.reset_episode()
        ep_r = 0.; prev_state = state.copy()
        info = {'result': 'timeout'}

        for ep_step in range(300):
            action = agent.select_action(state, deterministic=False)
            ns, r, done, info = env.step(action)

            # 存储经验
            if is_ppo:
                lp, v = _get_logp_val(state, action)
                if agent_name == 'IMTCMO':
                    agent.store(state, action, lp, r, v, done, next_state=ns)
                elif agent_name == 'MORL-based':
                    agent.store(state, action, lp, r, v, done,
                                d_goal=float(info.get('dist_to_goal', 10.)))
                else:
                    agent.store(state, action, lp, r, v, done)
            elif is_dqn:
                if agent_name in ('APF-DQN', 'APF-D3QNPER'):
                    agent.store(state, action, r, ns, done, goal_dir=0.)
                else:
                    agent.store(state, action, r, ns, done)
            elif is_ddg:
                agent.store(state, action, r, ns, done)

            # 更新
            if is_ppo:
                if hasattr(agent, 'n_steps') and len(agent.buffer) >= agent.n_steps:
                    agent.update(last_value=0.)
            else:
                agent.update()

            ep_r  += r
            steps += 1
            state  = ns
            if done: break

        n_ep += 1
        res = info.get('result', 'timeout')
        if res == 'success':   n_suc += 1
        if res == 'collision': n_col += 1
        ep_rewards.append(ep_r)
        ep_lengths.append(ep_step + 1)

    elapsed = time.time() - t0

    # ── 评估 ──────────────────────────────────────────────────────────
    ev_suc = ev_col = 0; ev_paths = []; ev_r = []
    for ep in range(n_eval_ep):
        s = eval_env.reset()
        if hasattr(agent, 'reset_episode'): agent.reset_episode()
        ep_path = 0.; ep_er = 0.; prev = s[:2].copy(); done = False; info = {}
        for _ in range(300):
            a  = agent.select_action(s, deterministic=True)
            ns, r, done, info = eval_env.step(a)
            ep_path += float(np.linalg.norm(ns[:2] - prev))
            prev = ns[:2].copy(); ep_er += r; s = ns
            if done: break
        ri = info.get('result', 'timeout')
        if ri == 'success':   ev_suc += 1
        if ri == 'collision': ev_col += 1
        ev_paths.append(ep_path); ev_r.append(ep_er)

    fps = steps / max(elapsed, 1)
    result = {
        'name':     agent_name,
        'SR':       round(ev_suc / n_eval_ep, 4),
        'CR':       round(ev_col / n_eval_ep, 4),
        'APL':      round(float(np.mean(ev_paths)), 2),
        'EP_R':     round(float(np.mean(ev_r)),     2),
        'train_SR': round(n_suc / max(n_ep, 1),     4),
        'n_ep':     n_ep,
        'steps':    steps,
        'elapsed':  round(elapsed, 1),
        'fps':      round(fps, 0),
        'updates':  agent.n_updates,
    }
    print(f"  [{agent_name}] Done: SR={result['SR']:.3f}  CR={result['CR']:.3f}  "
          f"APL={result['APL']:.1f}  {fps:.0f}fps")
    return result


def plot_comparison(results: List[Dict], out_path: str):
    """生成所有基线方法的对比柱状图。"""
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                          'axes.spines.top':False,'axes.spines.right':False,
                          'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--'})

    names = [r['name'] for r in results]
    SR    = [r['SR']   for r in results]
    CR    = [r['CR']   for r in results]
    APL   = [r['APL']  for r in results]

    x = np.arange(len(names)); w = 0.30
    COLORS_SR = ['#1565C0' if n=='CD-HSSRL' else '#90CAF9' for n in names]
    COLORS_CR = ['#C62828' if n=='CD-HSSRL' else '#FFCDD2' for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.patch.set_facecolor('#F8F9FA')

    for ax, vals, label, colors, title in [
        (axes[0], SR,  'SR ↑',  COLORS_SR, 'Success Rate (SR)'),
        (axes[1], CR,  'CR ↓',  COLORS_CR, 'Collision Rate (CR)'),
        (axes[2], APL, 'APL ↓', COLORS_SR, 'Avg Path Length (APL) m'),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor='white',
                      linewidth=1.5, width=0.65)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v + max(vals)*0.01,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('-','\n') for n in names], fontsize=8)
        ax.set_ylabel(label); ax.set_title(title, fontweight='bold')

    fig.suptitle('Baseline Comparison — Amphibious Navigation (Simulation)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] {out_path}")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='all',
                        help='方法名 或 all')
    parser.add_argument('--steps',  type=int, default=2000,
                        help='训练步数（快速测试用 2000，论文用 2000000）')
    parser.add_argument('--eval',   type=int, default=20,
                        help='评估 episode 数')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--out',    type=str, default='results/baselines')
    args = parser.parse_args()

    OUT = ROOT / args.out
    OUT.mkdir(parents=True, exist_ok=True)

    ALL_METHODS = ['IPPO','DDQN','HEA-PPO','IMTCMO','APF-DQN','I-DDPG',
                   'MORL-based','RLCA','APF-D3QNPER','CLPPO-GIC',
                   'BarrierNet','pH-DRL','MP-DQL']

    methods = ALL_METHODS if args.method == 'all' else [args.method]

    print(f"\n{'='*60}")
    print(f"Baseline Methods Evaluation")
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Steps: {args.steps:,}   Eval EP: {args.eval}")
    print(f"{'='*60}\n")

    results = []
    for m in methods:
        try:
            r = run_single(m, args.steps, args.eval, args.device, args.seed)
            results.append(r)
        except Exception as e:
            print(f"  [{m}] ERROR: {e}")
            import traceback; traceback.print_exc()

    # 打印汇总表
    print(f"\n{'='*75}")
    print(f"{'Method':<15} {'SR':>6} {'CR':>6} {'APL':>8} {'EP_R':>8} "
          f"{'Updates':>8} {'fps':>6}")
    print(f"{'-'*75}")
    for r in results:
        print(f"{r['name']:<15} {r['SR']:>6.3f} {r['CR']:>6.3f} "
              f"{r['APL']:>8.1f} {r['EP_R']:>8.1f} "
              f"{r['updates']:>8} {r['fps']:>6.0f}")
    print(f"{'='*75}\n")

    # 生成对比图（方法≥2时）
    if len(results) >= 2:
        plot_comparison(results, str(OUT / 'baseline_comparison.png'))

    # 保存 JSON
    import json
    with open(OUT / 'baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results: {OUT}/baseline_results.json")


if __name__ == '__main__':
    main()
