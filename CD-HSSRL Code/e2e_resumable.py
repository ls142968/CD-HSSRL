#!/usr/bin/env python3
"""
e2e_resumable.py
可断点续跑的端到端训练。
每 SAVE_FREQ 步保存一次模型 + 历史，下次运行自动恢复。

用法：
  python3 e2e_resumable.py           # 首次运行，跑满 TOTAL_STEPS
  python3 e2e_resumable.py           # 再次运行，自动从上次断点继续
  python3 e2e_resumable.py --reset   # 清除历史重新开始
"""
import sys, os, time, json, argparse
import numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from collections import deque

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'planner'))
sys.path.insert(0, str(ROOT/'policy'))
sys.path.insert(0, str(ROOT/'controller'))
sys.path.insert(0, str(ROOT/'env'))

from amphibious_sim_env import AmphibiousSimEnv, SimConfig, SimObstacle
from cd_grp import CDGlobalReachabilityPlanner, MapConfig, EnvironmentInfo
from hssp import HierarchicalSafeSwitchingPolicy, HSSPConfig, encode_waypoint
from sccc import SafetyConstrainedContinuousController, SCCCConfig

# ── 超参数（改这里就够了）────────────────────────────────────────────────────
TOTAL_STEPS = 40_000   # 总步数
SAVE_FREQ   = 1_000    # 每隔多少步存档一次
EVAL_FREQ   = 2_000    # 每隔多少步评估一次
N_EVAL_EP   = 20       # 评估时跑多少 episode
SEED        = 42
OUT_DIR     = ROOT / 'results' / 'e2e'

GOAL  = np.array([12., 0.])
START = np.array([-12., 0., -0.3])
# ────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--reset', action='store_true', help='清除历史重新训练')
args, _ = parser.parse_known_args()

OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_FILE    = OUT_DIR / 'checkpoint.json'   # 断点元信息
HISTORY_FILE = OUT_DIR / 'history.json'
HSSP_CKPT    = OUT_DIR / 'resume_hssp.pt'
SCCC_CKPT    = OUT_DIR / 'resume_sccc.pt'

if args.reset:
    for f in [CKPT_FILE, HISTORY_FILE, HSSP_CKPT, SCCC_CKPT]:
        if f.exists(): f.unlink()
    print("[Reset] 已清除历史，从头开始")

# ── 初始化环境 + 模块 ────────────────────────────────────────────────────────
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cpu'

obs_list = [SimObstacle(-10.,4.,1.0), SimObstacle(-6.,-3.,0.8),
            SimObstacle(-3.,2.5,0.6), SimObstacle(5.,3.5,0.9),
            SimObstacle(9.,-3.,1.1)]
sim_cfg  = SimConfig(start_pos=START, goal_pos=GOAL,
                     max_steps=300, obstacles=obs_list)
env      = AmphibiousSimEnv(sim_cfg, seed=SEED)
eval_env = AmphibiousSimEnv(sim_cfg, seed=SEED+1)

map_cfg  = MapConfig(resolution=0.5, x_min=-18, x_max=18, y_min=-12, y_max=12)
env_info = EnvironmentInfo(shoreline_x=0., slope_width=2.5)
cd_grp   = CDGlobalReachabilityPlanner(map_cfg, env_info)
cd_grp.build()

hssp_cfg = HSSPConfig(state_dim=25, hidden_size=256, n_steps=256,
                      batch_size=64, n_epochs=6, lr=3e-4,
                      gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                      lambda_sw=0.05)
sccc_cfg = SCCCConfig(state_dim=25, action_dim=2, hidden_size=256,
                      lr=3e-4, gamma=0.99, tau=0.005,
                      batch_size=128, buffer_size=100_000,
                      learning_starts=200, gradient_steps=1, kappa=1.0)

hssp = HierarchicalSafeSwitchingPolicy(hssp_cfg, device=DEVICE)
sccc = SafetyConstrainedContinuousController(sccc_cfg, device=DEVICE)

# ── 断点恢复 ─────────────────────────────────────────────────────────────────
start_step = 0
if CKPT_FILE.exists() and not args.reset:
    with open(CKPT_FILE) as f:
        ckpt_meta = json.load(f)
    start_step = ckpt_meta.get('total_steps', 0)
    if HSSP_CKPT.exists(): hssp.load(str(HSSP_CKPT))
    if SCCC_CKPT.exists(): sccc.load(str(SCCC_CKPT))
    print(f"[Resume] 从 step {start_step:,} 继续训练")
else:
    ckpt_meta = {}

# ── 历史恢复 ─────────────────────────────────────────────────────────────────
HIST_KEYS = ['steps','ep_r','SSI','sw_cnt',
             'SR','CR','APL','EC',
             'lL','lH','lsw','alpha']
if HISTORY_FILE.exists() and not args.reset:
    with open(HISTORY_FILE) as f:
        hist = json.load(f)
    # 补充可能缺失的 key
    for k in HIST_KEYS:
        if k not in hist:
            hist[k] = []
else:
    hist = {k: [] for k in HIST_KEYS}

recent_r   = deque(maxlen=50)
recent_res = deque(maxlen=50)
traj_buf   = deque(maxlen=5)

# ── 评估函数 ─────────────────────────────────────────────────────────────────
def run_eval(n_ep: int) -> dict:
    n_suc = n_col = 0
    paths = []; energies = []
    for _ in range(n_ep):
        s = eval_env.reset(randomize=False)
        hssp.reset_episode()
        _, wps = cd_grp.plan((float(s[0]), float(s[1])),
                              (float(GOAL[0]), float(GOAL[1])),
                              waypoint_spacing=2.)
        wi = 0; pp = pe = 0.; done = False; info = {}; prev = s[:2].copy()
        for t in range(300):
            wp = wps[min(wi, len(wps)-1)]; wn = np.array(wp[:2])
            we = encode_waypoint(s[:2], wn); dom = int(s[-1])
            if np.linalg.norm(s[:2]-wn) < 1.5 and wi < len(wps)-1:
                wi += 1
            if t == 0 or hssp.should_reselect(s[:2], wn, dom):
                opt, lp, val, probs = hssp.select_option(
                    s, we, dom, deterministic=True)
            else:
                opt = hssp.current_option
                with torch.no_grad():
                    d, v = hssp.policy.get_dist(
                        torch.FloatTensor(s).unsqueeze(0),
                        torch.FloatTensor(we).unsqueeze(0))
                lp    = float(d.log_prob(torch.tensor([opt])).item())
                val   = float(v.item())
                probs = d.probs.squeeze(0).cpu().numpy()
            sa, _ = sccc.select_action(s, opt, deterministic=True)
            ns2, _, done, info = eval_env.step(sa)
            pp += float(np.linalg.norm(ns2[:2]-prev))
            pe += float(np.sum(sa**2))
            prev = ns2[:2].copy(); s = ns2
            if done: break
        r = info.get('result', 'timeout')
        if r == 'success':   n_suc += 1
        if r == 'collision': n_col += 1
        paths.append(pp); energies.append(pe)
    N = n_ep
    return dict(SR=round(n_suc/N,4), CR=round(n_col/N,4),
                APL=round(float(np.mean(paths)),2),
                EC=round(float(np.mean(energies)),2))

# ── 图表生成 ──────────────────────────────────────────────────────────────────
OC = ['#1565C0', '#FF8F00', '#2E7D32']

def plot_and_save(total_steps: int, n_ep: int, elapsed: float, em: dict):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':0.3,'grid.linestyle':'--'})

    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor('#F8F9FA')
    gs  = GridSpec(3, 4, figure=fig, hspace=0.44, wspace=0.32)

    sa = np.array(hist['steps'])
    ra = np.array(hist['ep_r'])

    # ── (0,0) Reward ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0,0])
    ax.plot(sa, ra, color='#90CAF9', lw=0.7, alpha=0.5)
    w = min(30, len(ra))
    if w > 1:
        sm = np.convolve(ra, np.ones(w)/w, mode='valid')
        ax.plot(sa[w-1:], sm, color='#1565C0', lw=2.5, label=f'MA-{w}')
    ax.set_xlabel('Steps'); ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Reward', fontweight='bold')
    ax.legend(fontsize=8)

    # ── (0,1) SR / CR ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0,1])
    if hist['SR']:
        es = np.linspace(EVAL_FREQ, total_steps, len(hist['SR']))
        ax.plot(es, hist['SR'], 'o-', color='#2E7D32',
                lw=2.5, ms=8, label='SR ↑')
        ax.plot(es, hist['CR'], 's--', color='#C62828',
                lw=2.5, ms=8, label='CR ↓')
        for x, y in zip(es, hist['SR']):
            ax.annotate(f'{y:.2f}', (x,y), xytext=(0,7),
                        textcoords='offset points', ha='center',
                        fontsize=9, color='#2E7D32', fontweight='bold')
        for x, y in zip(es, hist['CR']):
            ax.annotate(f'{y:.2f}', (x,y), xytext=(0,-14),
                        textcoords='offset points', ha='center',
                        fontsize=9, color='#C62828', fontweight='bold')
    ax.set_xlabel('Steps'); ax.set_ylabel('Rate')
    ax.set_title('Success / Collision Rate', fontweight='bold')
    ax.legend(fontsize=9); ax.set_ylim(-0.05, 1.05)

    # ── (0,2) SSI ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0,2])
    sia = np.array(hist['SSI'])
    ax.plot(sa, sia, color='#90CAF9', lw=0.7, alpha=0.5)
    if len(sia) > 10:
        sm2 = np.convolve(sia, np.ones(10)/10, mode='valid')
        ax.plot(sa[9:], sm2, color='#00838F', lw=2.5, label='MA-10')
    ax.axhline(0.9, color='#1565C0', ls='--', lw=1.5, alpha=0.7,
               label='Target 0.9')
    ax.set_xlabel('Steps'); ax.set_ylabel('SSI (Eq.29)')
    ax.set_title('Switching Stability Index', fontweight='bold')
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    # ── (0,3) Switch histogram ───────────────────────────────────
    ax = fig.add_subplot(gs[0,3])
    sw = np.array(hist['sw_cnt'])
    if len(sw):
        bins = range(0, max(int(sw.max())+2, 6))
        ax.hist(sw, bins=bins, color='#1565C0', alpha=0.8,
                edgecolor='white', rwidth=0.8)
        ax.axvline(sw.mean(), color='#C62828', lw=2.5, ls='--',
                   label=f'Mean={sw.mean():.1f}')
    ax.set_xlabel('Mode Switches/Episode'); ax.set_ylabel('Count')
    ax.set_title('Switch Distribution', fontweight='bold')
    ax.legend(fontsize=8)

    # ── (1,0) SAC Actor Loss ────────────────────────────────────
    ax = fig.add_subplot(gs[1,0])
    if hist['lL']:
        ll = np.array(hist['lL'])
        ax.plot(np.arange(len(ll)), ll, color='#E65100',
                lw=1.5, alpha=0.85, label='Actor L_L (Eq.20)')
    ax.set_xlabel('SAC Updates'); ax.set_ylabel('Loss')
    ax.set_title('SAC Actor Loss (Eq.20)', fontweight='bold')
    ax.legend(fontsize=8)

    # ── (1,1) PPO Losses ────────────────────────────────────────
    ax = fig.add_subplot(gs[1,1])
    if hist['lH']:
        lh  = np.array(hist['lH'])
        lsw = np.array(hist['lsw'])
        x   = np.arange(len(lh))
        ax.plot(x, lh,  color='#6A1B9A', lw=2.0,
                label='Policy L_H (Eq.16)')
        ax.plot(x, lsw, color='#E65100', lw=1.5, ls='--',
                label='L_sw (Eq.15)')
    ax.set_xlabel('PPO Updates'); ax.set_ylabel('Loss')
    ax.set_title('PPO Losses (Eq.15+16)', fontweight='bold')
    ax.legend(fontsize=8)

    # ── (1,2) APL / EC ──────────────────────────────────────────
    ax = fig.add_subplot(gs[1,2])
    if hist['APL']:
        es = np.linspace(EVAL_FREQ, total_steps, len(hist['APL']))
        ax.plot(es, hist['APL'], 'o-', color='#1565C0',
                lw=2.5, ms=8, label='APL (m)')
        ax2 = ax.twinx()
        ax2.plot(es, hist['EC'], 's--', color='#E65100',
                 lw=2.0, ms=8, label='EC')
        ax2.set_ylabel('Energy EC', fontsize=9, color='#E65100')
        l1, n1 = ax.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, n1+n2, fontsize=8)
    ax.set_xlabel('Steps'); ax.set_ylabel('APL (m)')
    ax.set_title('Path Length & Energy (Eq.26,28)', fontweight='bold')

    # ── (1,3) Entropy coeff α ───────────────────────────────────
    ax = fig.add_subplot(gs[1,3])
    if hist['alpha']:
        al = np.array(hist['alpha'])
        ax.plot(np.arange(len(al)), al, color='#2E7D32', lw=1.8)
        ax.axhline(1.0, color='gray', ls=':', lw=1, alpha=0.6,
                   label='Init=1.0')
    ax.set_xlabel('SAC Updates'); ax.set_ylabel('α')
    ax.set_title('Auto Entropy Coeff α', fontweight='bold')
    ax.legend(fontsize=8)

    # ── (2, 0-1) Trajectories ───────────────────────────────────
    ax = fig.add_subplot(gs[2, :2])
    ax.axvspan(-18,-2.5, alpha=0.08, color='dodgerblue')
    ax.axvspan(-2.5, 2.5, alpha=0.08, color='orange')
    ax.axvspan( 2.5, 18,  alpha=0.08, color='forestgreen')
    ax.axvline(-2.5, color='cyan', lw=2, ls='--', alpha=0.8)
    ax.axvline( 2.5, color='cyan', lw=2, ls='--', alpha=0.8)
    ax.text(-10, 5.5, 'Water Domain',  fontsize=9,
            color='steelblue', fontweight='bold', ha='center')
    ax.text(  0, 5.5, 'Transition',    fontsize=9,
            color='darkorange', fontweight='bold', ha='center')
    ax.text(  9, 5.5, 'Land Domain',   fontsize=9,
            color='sienna', fontweight='bold', ha='center')
    for obs in obs_list:
        c = plt.Circle((obs.x, obs.y), obs.radius,
                       color='#B71C1C', alpha=0.4, zorder=3)
        ax.add_patch(c)
    col_r = {'success':'#2E7D32','collision':'#C62828','timeout':'#FF8F00'}
    for i, td in enumerate(traj_buf):
        traj = np.array(td['traj'])
        opts = td['options']
        r    = td['result']
        al   = 0.30 + 0.14*i
        lw   = 1.0  + 0.50*i
        for j in range(min(len(traj)-1, len(opts))):
            opt = opts[j] if j < len(opts) else 0
            ax.plot(traj[j:j+2,0], traj[j:j+2,1],
                    color=OC[opt], lw=lw, alpha=al,
                    solid_capstyle='round')
        c_end = col_r.get(r, '#455A64')
        ax.plot(traj[-1,0], traj[-1,1], 'o',
                color=c_end, ms=10,
                markeredgecolor='white', markeredgewidth=1.5,
                zorder=5)
    ax.plot(*START[:2], 'g^', ms=14, zorder=6, label='Start',
            markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(*GOAL, 'r*', ms=16, zorder=6, label='Goal',
            markeredgecolor='white', markeredgewidth=1.5)
    patches  = [mpatches.Patch(color=c, label=n)
                for n, c in zip(['Water(o=0)','Transition(o=1)','Land(o=2)'], OC)]
    patches += [mpatches.Patch(color=c, label=n) for n, c in col_r.items()]
    ax.legend(handles=patches, fontsize=8, loc='lower right',
              ncol=2, framealpha=0.9)
    ax.set_xlim(-14,14); ax.set_ylim(-6.5,6.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Recent Trajectories  [Color = Motion Option from π_H (HSSP)]',
                 fontweight='bold')

    # ── (2,2) Option pie ────────────────────────────────────────
    ax = fig.add_subplot(gs[2,2])
    all_opts = []
    for td in traj_buf:
        all_opts.extend(td['options'])
    if all_opts:
        cnts = [all_opts.count(i) for i in range(3)]
        tot  = sum(cnts) + 1e-9
        weds, _ = ax.pie(
            [c/tot for c in cnts], colors=OC,
            startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2.5))
        ax.legend(weds,
                  [f'{n}\n{c/tot*100:.0f}%'
                   for n, c in zip(['Water','Transition','Land'], cnts)],
                  fontsize=9, loc='lower center',
                  bbox_to_anchor=(0.5,-0.06))
    ax.set_title('Option Distribution\n(π_H output)', fontweight='bold')

    # ── (2,3) Radar ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[2,3], polar=True)
    ssi_f = hist['SSI'][-1] if hist['SSI'] else 0
    cats  = ['SR','1-CR','SSI','CTS','1-SVR']
    vals  = [em['SR'], max(0,1-em['CR']), ssi_f,
             em['SR'], max(0,1-em['CR'])]
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist()
    vp = vals + [vals[0]]; ap = angles + [angles[0]]
    ax.plot(ap, vp, 'o-', color='#1565C0', lw=2.5, ms=8)
    ax.fill(ap, vp, color='#1565C0', alpha=0.22)
    ax.plot(ap, [0.8]*6, '--', color='gray', lw=1, alpha=0.4)
    ax.set_xticks(angles)
    ax.set_xticklabels(cats, size=9, fontweight='bold')
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.4)
    ax.set_title(f'Final Metrics\nSR={em["SR"]:.3f}  SSI={ssi_f:.3f}',
                 fontweight='bold', pad=18)

    fig.suptitle(
        f'CD-HSSRL End-to-End Training — Simulation Environment\n'
        f'Task: water_to_land  |  Steps: {total_steps:,} / {TOTAL_STEPS:,}  '
        f'|  Episodes: {n_ep}  |  Time: {elapsed:.0f}s  '
        f'|  Speed: {total_steps/max(elapsed,1):.0f} steps/s',
        fontsize=12, fontweight='bold', y=1.01)

    out = OUT_DIR / 'training_result.png'
    plt.savefig(out, dpi=175, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    return out

# ── 主训练循环 ────────────────────────────────────────────────────────────────
total_steps = start_step
n_ep        = ckpt_meta.get('n_episodes', 0)
ppo_steps   = 0
t0          = time.time() - ckpt_meta.get('elapsed', 0)

remaining = TOTAL_STEPS - total_steps
if remaining <= 0:
    print(f"[Done] 已完成 {total_steps:,} 步，无需继续训练。")
    print("运行 `python3 e2e_resumable.py --reset` 重新开始。")
    em = run_eval(N_EVAL_EP)
    plot_and_save(total_steps, n_ep, time.time()-t0, em)
    sys.exit(0)

print(f"[Train] 还需 {remaining:,} 步  (共 {TOTAL_STEPS:,})")
print("="*55)

while total_steps < TOTAL_STEPS:

    # ── Episode 初始化 ────────────────────────────────────────
    s = env.reset(randomize=True)
    hssp.reset_episode(); n_ep += 1
    _, wps = cd_grp.plan(
        (float(s[0]), float(s[1])),
        (float(GOAL[0]), float(GOAL[1])),
        waypoint_spacing=2.)
    wi    = 0
    ep_r  = 0.
    traj  = [s[:2].copy()]
    opts  = []
    info  = {'result': 'timeout'}

    for step in range(300):
        wp  = wps[min(wi, len(wps)-1)]; wn = np.array(wp[:2])
        we  = encode_waypoint(s[:2], wn); dom = int(s[-1])
        if np.linalg.norm(s[:2]-wn) < 1.5 and wi < len(wps)-1:
            wi += 1

        need = (step == 0 or hssp.should_reselect(s[:2], wn, dom))
        if need:
            opt, lp, val, probs = hssp.select_option(s, we, dom)
        else:
            opt = hssp.current_option
            with torch.no_grad():
                d, v = hssp.policy.get_dist(
                    torch.FloatTensor(s).unsqueeze(0),
                    torch.FloatTensor(we).unsqueeze(0))
            lp    = float(d.log_prob(torch.tensor([opt])).item())
            val   = float(v.item())
            probs = d.probs.squeeze(0).cpu().numpy()

        sa, _ = sccc.select_action(s, opt)
        ns2, br, done, info = env.step(sa)
        r_safe, _ = sccc.compute_safe_reward(br, s)

        sccc.store(s, opt, sa, r_safe, ns2, int(ns2[-1]), done)
        hssp.store(s, we, opt, lp, r_safe, val, done, probs)

        sm = sccc.update()
        if sm:
            hist['lL'].append(float(sm.get('actor_loss', 0)))
            hist['alpha'].append(float(sm.get('alpha', 0)))

        ppo_steps += 1
        if ppo_steps >= hssp_cfg.n_steps:
            with torch.no_grad():
                _, lv = hssp.policy(
                    torch.FloatTensor(ns2).unsqueeze(0),
                    torch.FloatTensor(we).unsqueeze(0))
            pm = hssp.update(float(lv.item())); ppo_steps = 0
            hist['lH'].append(float(pm.get('loss_policy', 0)))
            hist['lsw'].append(float(pm.get('loss_sw', 0)))

        ep_r += r_safe; total_steps += 1
        traj.append(ns2[:2].copy()); opts.append(opt); s = ns2
        if done:
            break

    # ── Episode 结束 ──────────────────────────────────────────
    res = info.get('result', 'timeout')
    ssi = hssp.switching_stability_index()
    recent_r.append(ep_r); recent_res.append(res)
    traj_buf.append({'traj': traj, 'options': opts, 'result': res})
    hist['steps'].append(total_steps)
    hist['ep_r'].append(ep_r)
    hist['SSI'].append(ssi)
    hist['sw_cnt'].append(hssp.switch_count)

    # ── 控制台日志（每 10 个 episode）────────────────────────
    if n_ep % 10 == 0:
        sr  = sum(1 for r in recent_res if r=='success') / len(recent_res)
        cr  = sum(1 for r in recent_res if r=='collision') / len(recent_res)
        fps = total_steps / max(time.time()-t0, 1)
        ll  = (f" lL={np.mean(hist['lL'][-20:]):.3f}"
               if hist['lL'] else "")
        lh  = (f" lH={np.mean(hist['lH'][-5:]):.4f}"
               if hist['lH'] else "")
        print(f"  [Step {total_steps:6,}] ep={n_ep:4d}  "
              f"R={np.mean(recent_r):7.1f}  "
              f"SR={sr:.3f}  CR={cr:.3f}  SSI={ssi:.3f}  "
              f"α={sccc.updater.alpha:.4f}{ll}{lh}  {fps:.0f}fps")

    # ── 定期评估 ─────────────────────────────────────────────
    if total_steps % EVAL_FREQ == 0:
        em = run_eval(N_EVAL_EP)
        hist['SR'].append(em['SR']); hist['CR'].append(em['CR'])
        hist['APL'].append(em['APL']); hist['EC'].append(em['EC'])
        elapsed = time.time() - t0
        print(f"\n  [EVAL @ {total_steps:,}]  "
              f"SR={em['SR']:.4f}  CR={em['CR']:.4f}  "
              f"APL={em['APL']:.1f}  EC={em['EC']:.1f}\n")

    # ── 定期存档 ─────────────────────────────────────────────
    if total_steps % SAVE_FREQ == 0:
        elapsed = time.time() - t0
        # 保存模型
        hssp.save(str(HSSP_CKPT))
        sccc.save(str(SCCC_CKPT))
        # 保存历史
        with open(HISTORY_FILE, 'w') as f:
            json.dump(hist, f)
        # 保存断点元信息
        with open(CKPT_FILE, 'w') as f:
            json.dump({'total_steps': total_steps,
                       'n_episodes':  n_ep,
                       'elapsed':     elapsed}, f)
        # 更新图表
        em_plot = run_eval(N_EVAL_EP) if total_steps % EVAL_FREQ != 0 \
                  else {'SR': hist['SR'][-1] if hist['SR'] else 0,
                        'CR': hist['CR'][-1] if hist['CR'] else 0,
                        'APL': hist['APL'][-1] if hist['APL'] else 0,
                        'EC': hist['EC'][-1] if hist['EC'] else 0}
        plot_and_save(total_steps, n_ep, elapsed, em_plot)
        print(f"  [Save] step={total_steps:,}  "
              f"模型 + 历史 + 图表已更新")

# ── 最终处理 ──────────────────────────────────────────────────────────────────
elapsed = time.time() - t0
em_final = run_eval(N_EVAL_EP)
hist['SR'].append(em_final['SR']); hist['CR'].append(em_final['CR'])
hist['APL'].append(em_final['APL']); hist['EC'].append(em_final['EC'])

hssp.save(str(OUT_DIR/'final_hssp.pt'))
sccc.save(str(OUT_DIR/'final_sccc.pt'))
with open(HISTORY_FILE, 'w') as f: json.dump(hist, f)
out_chart = plot_and_save(total_steps, n_ep, elapsed, em_final)

print("\n" + "="*55)
print("训练完成！")
print(f"  总步数   : {total_steps:,}")
print(f"  总 Episode: {n_ep}")
print(f"  用时     : {elapsed:.1f}s  ({total_steps/max(elapsed,1):.0f} steps/s)")
print(f"\n  最终指标 ({N_EVAL_EP} episodes):")
print(f"  SR   = {em_final['SR']:.4f}")
print(f"  CR   = {em_final['CR']:.4f}")
print(f"  APL  = {em_final['APL']:.2f} m")
print(f"  EC   = {em_final['EC']:.2f}")
print(f"  SSI  = {hist['SSI'][-1] if hist['SSI'] else 0:.4f}")
print(f"\n  图表 : {out_chart}")
print(f"  模型 : {OUT_DIR}/final_*.pt")
print("="*55)
