"""
CD-GRP 单元测试 + 可视化验证
运行: python3 test_cd_grp.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import time

from cd_grp import (
    MapConfig, EnvironmentInfo, Obstacle,
    CostLayerBuilder, CostMapFusion, IncrementalAstar,
    PathPostProcessor, CDGlobalReachabilityPlanner,
)


# ─────────────────────────────────────────────────────────────────────────────
# 构建测试场景
# ─────────────────────────────────────────────────────────────────────────────

def make_scenario():
    cfg = MapConfig(
        resolution = 0.4,
        x_min=-18.0, x_max=18.0,
        y_min=-12.0, y_max=12.0,
        alpha=0.30, beta=0.20, delta=0.20, eta=0.30,
    )
    env = EnvironmentInfo(
        shoreline_x  = 0.0,
        slope_width  = 2.5,
        slope_angle  = 15.0,
        obstacles=[
            # 水域障碍（浮标、暗礁）
            Obstacle(-12.0,  4.0, 1.0),
            Obstacle( -8.0, -4.5, 0.8),
            Obstacle( -5.0,  2.5, 0.7),
            Obstacle( -3.0, -2.0, 0.6),
            # 陆地障碍（岩石、建筑）
            Obstacle(  4.0,  3.5, 0.9),
            Obstacle(  8.0, -3.0, 1.1),
            Obstacle( 12.0,  5.0, 0.8),
        ],
    )
    return cfg, env


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — 四层代价地图验证  (Eq.5–8)
# ─────────────────────────────────────────────────────────────────────────────

def test_cost_layers(cfg, env):
    print("\n[Test 1] 构建四层代价地图 (Eq.5–8)")
    builder = CostLayerBuilder(cfg, env)

    t0 = time.perf_counter()
    D = builder.build_D()
    S = builder.build_S()
    F = builder.build_F()
    O = builder.build_O()
    elapsed = time.perf_counter() - t0

    print(f"  地图尺寸  : {cfg.nx} × {cfg.ny}")
    print(f"  D(x,y)   : min={D.min():.3f}  max={D.max():.3f}  mean={D.mean():.3f}")
    print(f"  S(x,y)   : min={S.min():.3f}  max={S.max():.3f}  mean={S.mean():.3f}")
    print(f"  F(x,y)   : min={F.min():.3f}  max={F.max():.3f}  mean={F.mean():.3f}")
    print(f"  O(x,y)   : min={O.min():.3f}  max={O.max():.3f}  mean={O.mean():.3f}")
    print(f"  构建耗时  : {elapsed*1000:.1f} ms")

    assert D.shape == (cfg.nx, cfg.ny), "D shape error"
    assert 0 <= D.min() and D.max() <= 1.0, "D range error"
    assert 0 <= S.min() and S.max() <= 1.0, "S range error"
    assert 0 <= F.min() and F.max() <= 1.0, "F range error"
    assert 0 <= O.min() and O.max() <= 1.0, "O range error"
    print("  ✓ 所有断言通过")
    return D, S, F, O


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — 代价地图融合  (Eq.9)
# ─────────────────────────────────────────────────────────────────────────────

def test_fusion(cfg, D, S, F, O):
    print("\n[Test 2] 代价地图融合 G = αD + βS + δF + ηO (Eq.9)")
    fusion = CostMapFusion(cfg)

    t0 = time.perf_counter()
    G = fusion.fuse(D, S, F, O)
    elapsed = time.perf_counter() - t0

    print(f"  α={cfg.alpha}  β={cfg.beta}  δ={cfg.delta}  η={cfg.eta}")
    print(f"  G(x,y)   : min={G.min():.3f}  max={G.max():.3f}  mean={G.mean():.3f}")
    print(f"  融合耗时  : {elapsed*1000:.2f} ms")

    assert G.shape == (cfg.nx, cfg.ny)
    assert 0 <= G.min() and G.max() <= 1.0
    print("  ✓ 所有断言通过")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — A* 路径搜索  (Eq.10–11)
# ─────────────────────────────────────────────────────────────────────────────

def test_astar(cfg, G):
    print("\n[Test 3] 增量 A* 路径搜索 (Eq.10–11)")
    astar = IncrementalAstar(cfg)
    post  = PathPostProcessor()

    start = (-15.0, -2.0)
    goal  = ( 14.0,  2.0)

    # 第一次搜索（冷启动）
    t0 = time.perf_counter()
    raw_path = astar.search(G, start, goal)
    t_cold = time.perf_counter() - t0

    smooth = post.smooth(raw_path)
    waypoints = post.extract_waypoints(smooth, waypoint_spacing=2.0)

    # 路径总长
    def path_len(p):
        return sum(np.sqrt((p[i][0]-p[i-1][0])**2+(p[i][1]-p[i-1][1])**2)
                   for i in range(1,len(p)))

    raw_len    = path_len(raw_path)
    smooth_len = path_len(smooth)

    print(f"  起点 → 终点 : {start} → {goal}")
    print(f"  原始路径    : {len(raw_path)} 格点  总长 {raw_len:.1f} m")
    print(f"  平滑路径    : {len(smooth)} 点   总长 {smooth_len:.1f} m")
    print(f"  航点序列 W  : {len(waypoints)} 个航点 (Eq.12)")
    print(f"  冷启动耗时  : {t_cold*1000:.1f} ms")

    # 第二次搜索（热启动，起点微移）
    start2 = (-14.5, -2.0)
    t0 = time.perf_counter()
    _ = astar.search(G, start2, goal)
    t_warm = time.perf_counter() - t0
    print(f"  热启动耗时  : {t_warm*1000:.1f} ms  (加速比 {t_cold/max(t_warm,1e-6):.1f}x)")

    assert len(raw_path) >= 2
    assert len(waypoints) >= 2
    print("  ✓ 所有断言通过")
    return raw_path, smooth, waypoints


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — 增量重规划（动态障碍物）
# ─────────────────────────────────────────────────────────────────────────────

def test_incremental_replan(cfg, env):
    print("\n[Test 4] 增量重规划（新增动态障碍物）")
    planner = CDGlobalReachabilityPlanner(cfg, env)
    start = (-15.0, 0.0)
    goal  = ( 14.0, 0.0)

    # 初始规划
    planner.build()
    _, wp1 = planner.plan(start, goal)

    # 模拟 LiDAR 检测到新障碍物
    new_obs = [Obstacle(-1.0, 0.5, 0.8)]    # 在过渡区中间出现新障碍
    t0 = time.perf_counter()
    G_new, changed = planner.update_obstacles(new_obs)
    _, wp2 = planner.plan(start, goal, changed_cells=changed)
    t_incr = time.perf_counter() - t0

    print(f"  初始航点数  : {len(wp1)}")
    print(f"  重规划航点数: {len(wp2)}")
    print(f"  变化栅格数  : {len(changed)}")
    print(f"  增量重规划耗时: {t_incr*1000:.1f} ms")
    print("  ✓ 增量重规划完成")
    return wp1, wp2, new_obs


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def visualize(cfg, env, D, S, F, O, G, raw_path, smooth_path, waypoints,
              wp_replan, new_obs):
    print("\n[Viz] 生成论文图表...")

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('#F8F9FA')
    gs = GridSpec(2, 4, figure=fig, hspace=0.38, wspace=0.30)

    EXTENT = [cfg.x_min, cfg.x_max, cfg.y_min, cfg.y_max]
    shore  = env.shoreline_x

    # ── 共用绘图参数 ──────────────────────────────────────────────────
    def draw_shore(ax):
        ax.axvline(shore, color='cyan', lw=1.8, ls='--', alpha=0.8, label='Shoreline')
        ax.axvspan(cfg.x_min, shore, alpha=0.04, color='dodgerblue')
        ax.axvspan(shore, cfg.x_max, alpha=0.04, color='sandybrown')

    def draw_obstacles(ax, extra=None):
        all_obs = env.obstacles + (extra or [])
        for obs in all_obs:
            c = mpatches.Circle((obs.x, obs.y), obs.radius,
                                 color='red', alpha=0.4, zorder=4)
            ax.add_patch(c)

    cmaps = ['Blues_r', 'YlOrRd', 'Greens', 'Reds']
    layers = [D.T, S.T, F.T, O.T]
    titles = [
        'D(x,y) — Water Depth Risk\n(Eq.5)',
        'S(x,y) — Shoreline Slope Cost\n(Eq.6)',
        'F(x,y) — Terrain Friction Cost\n(Eq.7)',
        'O(x,y) — Obstacle Occupancy\n(Eq.8)',
    ]
    weights = ['α = 0.30', 'β = 0.20', 'δ = 0.20', 'η = 0.30']

    # ── Row 0: 四层代价地图 ──────────────────────────────────────────
    for col, (layer, cmap, title, w) in enumerate(
            zip(layers, cmaps, titles, weights)):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(layer, cmap=cmap, origin='lower',
                       extent=EXTENT, aspect='auto', vmin=0, vmax=1)
        draw_shore(ax)
        draw_obstacles(ax)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9) if col == 0 else None
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        ax.text(0.97, 0.96, w, transform=ax.transAxes,
                ha='right', va='top', fontsize=9, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.25', fc='#333', alpha=0.65))
        ax.tick_params(labelsize=8)
        ax.grid(False)

    # ── Row 1 col 0-1: 融合代价地图 G ───────────────────────────────
    ax_g = fig.add_subplot(gs[1, :2])
    cmap_g = LinearSegmentedColormap.from_list(
        'cost', ['#1a6b1a', '#f0e020', '#cc2222'])
    im_g = ax_g.imshow(G.T, cmap=cmap_g, origin='lower',
                        extent=EXTENT, aspect='auto', vmin=0, vmax=1)
    draw_shore(ax_g)
    draw_obstacles(ax_g)

    # 绘制路径
    if len(raw_path) > 1:
        rx, ry = zip(*raw_path)
        ax_g.plot(rx, ry, color='#CCCCCC', lw=1.0, alpha=0.6,
                  label='Raw A* path', zorder=5)
    if len(smooth_path) > 1:
        sx, sy = zip(*smooth_path)
        ax_g.plot(sx, sy, color='white', lw=2.2, label='Smooth path', zorder=6)
    if len(waypoints) > 1:
        wx, wy = zip(*waypoints)
        ax_g.scatter(wx, wy, c='#FFD700', s=60, zorder=8,
                     edgecolors='black', linewidths=0.7,
                     label=f'Waypoints W (K={len(waypoints)})')

    # 起终点
    ax_g.plot(*raw_path[0],  'go', ms=12, zorder=9, label='Start')
    ax_g.plot(*raw_path[-1], 'r*', ms=14, zorder=9, label='Goal')

    cb_g = plt.colorbar(im_g, ax=ax_g, fraction=0.025, pad=0.02)
    cb_g.set_label('Cost G(x,y)', fontsize=9)
    ax_g.set_title(
        'G(x,y) = αD + βS + δF + ηO   (Eq.9)\n'
        'A* Optimal Path P* (Eq.10) & Waypoint Sequence W (Eq.12)',
        fontsize=10, fontweight='bold', pad=4)
    ax_g.set_xlabel('X (m)', fontsize=9)
    ax_g.set_ylabel('Y (m)', fontsize=9)
    ax_g.legend(fontsize=8, loc='lower right', framealpha=0.85)
    ax_g.tick_params(labelsize=8)
    ax_g.grid(False)

    # 域标注
    ax_g.text(cfg.x_min + 0.8, cfg.y_max - 1.0, 'Water Domain (Sw)',
              fontsize=9, color='dodgerblue', fontweight='bold', alpha=0.9)
    ax_g.text(shore + 0.3,     cfg.y_max - 1.0, 'Transition (St)',
              fontsize=8, color='cyan',       fontweight='bold', alpha=0.9)
    ax_g.text(shore + 3.0,     cfg.y_max - 1.0, 'Land Domain (Sl)',
              fontsize=9, color='sandybrown', fontweight='bold', alpha=0.9)

    # ── Row 1 col 2: 增量重规划对比 ─────────────────────────────────
    ax_rp = fig.add_subplot(gs[1, 2])
    ax_rp.imshow(G.T, cmap=cmap_g, origin='lower',
                 extent=EXTENT, aspect='auto', vmin=0, vmax=1, alpha=0.75)
    draw_shore(ax_rp)
    draw_obstacles(ax_rp)

    if len(waypoints) > 1:
        wx, wy = zip(*waypoints)
        ax_rp.plot(wx, wy, 'w--', lw=1.8, label='Initial W', zorder=5)
    if len(wp_replan) > 1:
        rx2, ry2 = zip(*wp_replan)
        ax_rp.plot(rx2, ry2, color='#FFD700', lw=2.2,
                   label='Replanned W\'', zorder=6)

    for obs in new_obs:
        c = mpatches.Circle((obs.x, obs.y), obs.radius,
                             color='magenta', alpha=0.7, zorder=7,
                             label='New obstacle')
        ax_rp.add_patch(c)

    ax_rp.plot(*raw_path[0],  'go', ms=10, zorder=8)
    ax_rp.plot(*raw_path[-1], 'r*', ms=12, zorder=8)
    ax_rp.set_title('Incremental Replanning\n(Dynamic Obstacle Update)', fontsize=10,
                    fontweight='bold', pad=4)
    ax_rp.set_xlabel('X (m)', fontsize=9)
    ax_rp.legend(fontsize=8, loc='lower right', framealpha=0.85)
    ax_rp.tick_params(labelsize=8)
    ax_rp.grid(False)

    # ── Row 1 col 3: 路径代价分析 ───────────────────────────────────
    ax_cost = fig.add_subplot(gs[1, 3])
    if len(smooth_path) > 1:
        pts = np.array(smooth_path)
        # 查询路径上每点的 G 值
        costs = []
        dists = [0.0]
        for i, (px, py) in enumerate(smooth_path):
            gx, gy = cfg.world_to_grid(px, py)
            costs.append(G[gx, gy])
            if i > 0:
                dx = smooth_path[i][0] - smooth_path[i-1][0]
                dy = smooth_path[i][1] - smooth_path[i-1][1]
                dists.append(dists[-1] + np.sqrt(dx**2+dy**2))

        ax_cost.fill_between(dists, costs, alpha=0.3, color='#1565C0')
        ax_cost.plot(dists, costs, color='#1565C0', lw=2.0)

        # 标注过渡区
        shore_dist = None
        for i, (px, _) in enumerate(smooth_path):
            if abs(px - shore) < 1.0 and shore_dist is None:
                shore_dist = dists[i]
        if shore_dist:
            ax_cost.axvline(shore_dist, color='cyan', lw=1.5, ls='--',
                            label='Shoreline crossing')

        # 标注障碍物附近
        ax_cost.set_xlabel('Path Distance (m)', fontsize=9)
        ax_cost.set_ylabel('Cost G(x,y)', fontsize=9)
        ax_cost.set_ylim(0, 1.05)
        ax_cost.set_title('Path Cost Profile\n(Σ G along P*)', fontsize=10,
                          fontweight='bold', pad=4)
        ax_cost.legend(fontsize=8)
        ax_cost.tick_params(labelsize=8)

        # 标注总代价
        total_cost = sum(costs)
        ax_cost.text(0.97, 0.97, f'Total cost: {total_cost:.1f}',
                     transform=ax_cost.transAxes, ha='right', va='top',
                     fontsize=9, color='#1565C0',
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    # ── 总标题 ────────────────────────────────────────────────────────
    fig.suptitle(
        'CD-GRP: Cross-Domain Global Reachability Planner\n'
        'Section 3.3, Eq.(5)–(12)',
        fontsize=13, fontweight='bold', y=1.01)

    out = '/mnt/user-data/outputs/cd_grp_result.png'
    plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [✓] 图表已保存: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CD-GRP 单元测试")
    print("=" * 60)

    cfg, env = make_scenario()

    D, S, F, O = test_cost_layers(cfg, env)
    G          = test_fusion(cfg, D, S, F, O)
    raw, smooth, waypoints = test_astar(cfg, G)
    wp1, wp2, new_obs = test_incremental_replan(cfg, env)

    visualize(cfg, env, D, S, F, O, G,
              raw, smooth, waypoints, wp2, new_obs)

    print("\n" + "=" * 60)
    print("全部测试通过 ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
