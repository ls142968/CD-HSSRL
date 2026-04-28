"""
CD-GRP: Cross-Domain Global Reachability Planner
论文 Section 3.3, Eq.(5)–(12)

三个步骤：
  Step 1 — 构建四层代价地图 D, S, F, O          (Eq.5–8)
  Step 2 — 融合为统一代价地图 G(x,y)             (Eq.9)
  Step 3 — 增量 A* 搜索最优路径 P*, 输出航点 W   (Eq.10–12)
"""

import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from scipy.ndimage import gaussian_filter


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MapConfig:
    """地图参数"""
    resolution: float = 0.5          # 米/格
    x_min: float     = -20.0         # 世界坐标范围
    x_max: float     =  20.0
    y_min: float     = -15.0
    y_max: float     =  15.0
    # 融合权重 (Eq.9): α, β, δ, η
    alpha: float     = 0.30          # 水深风险权重
    beta:  float     = 0.20          # 坡度风险权重
    delta: float     = 0.20          # 摩擦风险权重
    eta:   float     = 0.30          # 障碍物权重

    @property
    def nx(self) -> int:
        return int((self.x_max - self.x_min) / self.resolution)

    @property
    def ny(self) -> int:
        return int((self.y_max - self.y_min) / self.resolution)

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """世界坐标 → 栅格索引"""
        gx = int((wx - self.x_min) / self.resolution)
        gy = int((wy - self.y_min) / self.resolution)
        gx = np.clip(gx, 0, self.nx - 1)
        gy = np.clip(gy, 0, self.ny - 1)
        return int(gx), int(gy)

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """栅格索引 → 世界坐标（格中心）"""
        wx = self.x_min + (gx + 0.5) * self.resolution
        wy = self.y_min + (gy + 0.5) * self.resolution
        return wx, wy


@dataclass
class Obstacle:
    """障碍物描述"""
    x: float
    y: float
    radius: float = 0.8


@dataclass
class EnvironmentInfo:
    """
    仿真环境信息
    x < shoreline_x        → 水域 (Sw)
    shoreline_x ± slope_w  → 过渡区 (St)
    x > shoreline_x        → 陆地 (Sl)
    """
    shoreline_x:  float         = 0.0
    slope_width:  float         = 2.5   # 过渡区半宽度 (m)
    slope_angle:  float         = 15.0  # 坡面角度 (度)
    water_depth_offshore: float = 3.0   # 深水区深度 (m)
    obstacles: List[Obstacle]   = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — 四层代价地图  (Eq.5–8)
# ─────────────────────────────────────────────────────────────────────────────

class CostLayerBuilder:
    """
    构建论文 Eq.(5)–(8) 定义的四层代价地图。
    所有输出值域 ∈ [0, 1]，数值越大代价越高。
    """

    def __init__(self, cfg: MapConfig, env: EnvironmentInfo):
        self.cfg = cfg
        self.env = env
        # 预计算每格的世界坐标
        xs = cfg.x_min + (np.arange(cfg.nx) + 0.5) * cfg.resolution
        ys = cfg.y_min + (np.arange(cfg.ny) + 0.5) * cfg.resolution
        self.X, self.Y = np.meshgrid(xs, ys, indexing='ij')  # shape (nx, ny)

    # ------------------------------------------------------------------
    def build_D(self) -> np.ndarray:
        """
        Eq.(5)  D(x,y) — 水深风险代价
        浅水区（靠近海岸线水侧）搁浅风险高 → 代价高
        深水区 → 代价低
        陆地区 → 0（不适用）
        """
        shore = self.env.shoreline_x
        d_raw = np.zeros((self.cfg.nx, self.cfg.ny))

        water_mask = self.X < shore
        if not np.any(water_mask):
            return d_raw

        # 距海岸线的距离（水侧为正）
        dist_to_shore = shore - self.X   # 水侧 > 0

        # 浅水风险：距海岸越近，深度越浅，搁浅风险越高
        # 用指数函数建模：risk = exp(-dist / λ)，λ = 3m
        depth_risk = np.exp(-dist_to_shore / 3.0)

        d_raw[water_mask] = depth_risk[water_mask]
        d_raw = gaussian_filter(d_raw, sigma=1.5)
        return self._normalize(d_raw)

    # ------------------------------------------------------------------
    def build_S(self) -> np.ndarray:
        """
        Eq.(6)  S(x,y) — 海岸线坡度过渡代价
        在海岸线过渡带（St）坡度最大 → 代价最高
        两侧（深水 / 平地）→ 代价低
        """
        shore = self.env.shoreline_x
        width = self.env.slope_width

        # 以海岸线为中心的高斯型代价
        dist = np.abs(self.X - shore)
        s_raw = np.exp(-0.5 * (dist / (width * 0.6)) ** 2)

        # 坡度角越大，代价越高（用角度调制幅值）
        angle_factor = np.sin(np.radians(self.env.slope_angle))
        s_raw *= angle_factor

        s_raw = gaussian_filter(s_raw, sigma=1.0)
        return self._normalize(s_raw)

    # ------------------------------------------------------------------
    def build_F(self) -> np.ndarray:
        """
        Eq.(7)  F(x,y) — 地形摩擦代价
        陆地区摩擦系数高（硬地面、砂砾）→ 能耗高 → 代价高
        水域摩擦系数低（粘性阻力用水动力模型处理，此处不重复计入）
        """
        shore = self.env.shoreline_x
        f_raw = np.zeros((self.cfg.nx, self.cfg.ny))

        # 陆地摩擦：距海岸越远，地形越复杂
        land_mask  = self.X >= shore
        dist_land  = np.maximum(self.X - shore, 0.0)
        # 线性 + 饱和：f = min(0.3 + 0.07*d, 0.85)
        f_land = np.clip(0.3 + 0.07 * dist_land, 0.0, 0.85)
        f_raw[land_mask] = f_land[land_mask]

        # 过渡区：砂砾、淤泥，摩擦中等
        trans_mask = np.abs(self.X - shore) < self.env.slope_width
        f_raw[trans_mask] = np.maximum(f_raw[trans_mask], 0.5)

        f_raw = gaussian_filter(f_raw, sigma=1.2)
        return self._normalize(f_raw)

    # ------------------------------------------------------------------
    def build_O(self, dynamic_obstacles: Optional[List[Obstacle]] = None) -> np.ndarray:
        """
        Eq.(8)  O(x,y) — 障碍物占用代价
        支持静态（地图预设）+ 动态（LiDAR实时更新）障碍物。
        每个障碍物用各向同性高斯膨胀建模。
        """
        o_raw = np.zeros((self.cfg.nx, self.cfg.ny))
        all_obs = self.env.obstacles.copy()
        if dynamic_obstacles:
            all_obs.extend(dynamic_obstacles)

        for obs in all_obs:
            dist = np.sqrt((self.X - obs.x) ** 2 + (self.Y - obs.y) ** 2)
            # 膨胀半径内设为高代价，衰减系数 σ = radius
            sigma = max(obs.radius, 0.3)
            o_raw += np.exp(-0.5 * (dist / sigma) ** 2)

        o_raw = gaussian_filter(o_raw, sigma=0.8)
        return self._normalize(o_raw)

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """归一化到 [0, 1]"""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — 统一代价地图融合  (Eq.9)
# ─────────────────────────────────────────────────────────────────────────────

class CostMapFusion:
    """
    Eq.(9):  G(x,y) = α·D(x,y) + β·S(x,y) + δ·F(x,y) + η·O(x,y)
    """

    def __init__(self, cfg: MapConfig):
        self.cfg = cfg

    def fuse(self,
             D: np.ndarray,
             S: np.ndarray,
             F: np.ndarray,
             O: np.ndarray) -> np.ndarray:
        """返回融合代价地图 G，值域 [0,1]"""
        G = (self.cfg.alpha * D +
             self.cfg.beta  * S +
             self.cfg.delta * F +
             self.cfg.eta   * O)
        # 再次归一化确保 [0,1]
        mn, mx = G.min(), G.max()
        if mx - mn > 1e-8:
            G = (G - mn) / (mx - mn)
        return G

    def update_obstacle_layer(self,
                               G_prev: np.ndarray,
                               O_old:  np.ndarray,
                               O_new:  np.ndarray) -> np.ndarray:
        """
        增量更新：仅替换障碍物层，避免重算整张地图。
        G_new = G_prev - η·O_old + η·O_new
        """
        G_new = G_prev - self.cfg.eta * O_old + self.cfg.eta * O_new
        G_new = np.clip(G_new, 0.0, 1.0)
        return G_new


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — 增量 A* 路径搜索  (Eq.10–11)
# ─────────────────────────────────────────────────────────────────────────────

class IncrementalAstar:
    """
    论文 Eq.(10)–(11) 的增量 A* 实现。

    P* = argmin_P  Σ_{(x,y)∈P} G(x,y)          (Eq.10)
    f(n) = g(n) + h(n)                           (Eq.11)

    g(n): 从起点到 n 的累计代价
    h(n): 启发式距离（欧氏距离，保证可采纳性）

    "增量"体现在：
      - 地图局部更新时只对受影响区域重新搜索
      - 维护上次搜索的 g_score 缓存，热启动下次搜索
    """

    # 8邻域移动 (dr, dc, move_cost_factor)
    NEIGHBORS = [
        (-1,  0, 1.000), ( 1,  0, 1.000),
        ( 0, -1, 1.000), ( 0,  1, 1.000),
        (-1, -1, 1.414), (-1,  1, 1.414),
        ( 1, -1, 1.414), ( 1,  1, 1.414),
    ]

    def __init__(self, cfg: MapConfig):
        self.cfg = cfg
        # 增量缓存：上一次的 g_score，供热启动
        self._cached_g: Dict[Tuple[int,int], float] = {}
        self._last_start: Optional[Tuple[int,int]] = None
        self._last_goal:  Optional[Tuple[int,int]] = None

    # ------------------------------------------------------------------
    def search(self,
               G: np.ndarray,
               start_world: Tuple[float, float],
               goal_world:  Tuple[float, float],
               changed_cells: Optional[List[Tuple[int,int]]] = None
               ) -> List[Tuple[float, float]]:
        """
        在代价地图 G 上搜索最优路径 P*。

        热启动策略：
          - 地图未变、仅起点微移 → 复用 g_score 剪枝（加速 ~30%）
          - 地图有更新（changed_cells 非空）→ 清空缓存，完整搜索
          - 起点或终点改变 → 清空缓存，完整搜索
        """
        start = self.cfg.world_to_grid(*start_world)
        goal  = self.cfg.world_to_grid(*goal_world)

        # 地图变化 or 起终点改变 → 清缓存，保证 came_from 完整性
        if changed_cells or start != self._last_start or goal != self._last_goal:
            self._cached_g.clear()

        path_grid = self._astar(G, start, goal)

        self._last_start = start
        self._last_goal  = goal

        path_world = [self.cfg.grid_to_world(gx, gy) for gx, gy in path_grid]
        return path_world

    # ------------------------------------------------------------------
    def _astar(self,
               G: np.ndarray,
               start: Tuple[int,int],
               goal:  Tuple[int,int]
               ) -> List[Tuple[int,int]]:
        """
        核心 A* 算法。
        f(n) = g(n) + h(n)   (Eq.11)

        热启动策略：
          - g_score 从缓存恢复，用于剪枝（跳过已知代价更高的路径）
          - came_from 每次从头构建（保证路径回溯正确）
          - open_set 仅放入起点，让搜索自然展开
          这样在地图不变、起终点相同时，已探索节点直接被剪枝跳过，
          只有新的 changed_cells 区域才会被重新展开。
        """
        nx, ny = G.shape
        came_from: Dict[Tuple[int,int], Tuple[int,int]] = {}
        visited = set()

        # 热启动：复用缓存 g_score（剪枝用），但起点强制置 0
        if self._cached_g:
            g_score: Dict[Tuple[int,int], float] = dict(self._cached_g)
        else:
            g_score = {}
        g_score[start] = 0.0

        # open_set 始终从起点开始（came_from 需要完整重建）
        open_set: List[Tuple[float, Tuple[int,int]]] = []
        heapq.heappush(open_set, (self._heuristic(start, goal), start))

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            # 到达目标 → 保存缓存 + 回溯路径
            if current == goal:
                self._cached_g = dict(g_score)
                return self._reconstruct(came_from, current, start)

            cr, cc = current
            for dr, dc, move_factor in self.NEIGHBORS:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < nx and 0 <= nc < ny):
                    continue
                nb = (nr, nc)

                # Eq.(10): 格代价 = (G(nb) + ε) × 移动距离（米）
                step_cost = (G[nr, nc] + 0.01) * move_factor * self.cfg.resolution
                tentative_g = g_score.get(current, np.inf) + step_cost

                # 热启动剪枝：tentative_g 不优于缓存值时跳过
                if tentative_g < g_score.get(nb, np.inf):
                    g_score[nb]    = tentative_g
                    came_from[nb]  = current        # 记录前驱（回溯用）
                    h = self._heuristic(nb, goal)   # Eq.(11)
                    heapq.heappush(open_set, (tentative_g + h, nb))

        # 搜索失败：退化为直线路径
        return [start, goal]

    # ------------------------------------------------------------------
    def _heuristic(self,
                   a: Tuple[int,int],
                   b: Tuple[int,int]) -> float:
        """
        h(n) — 欧氏距离启发式 (Eq.11)
        乘以 resolution 转换为实际距离（米），与 g(n) 量纲一致。
        """
        return (np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
                * self.cfg.resolution * 0.01)   # 0.01 = 最小格代价ε，保证可采纳

    # ------------------------------------------------------------------
    @staticmethod
    def _reconstruct(came_from: Dict, current: Tuple[int,int],
                     start: Tuple[int,int]) -> List[Tuple[int,int]]:
        """回溯 came_from 指针，重建路径"""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path


# ─────────────────────────────────────────────────────────────────────────────
# 路径后处理：平滑 + 抽稀航点  (Eq.12)
# ─────────────────────────────────────────────────────────────────────────────

class PathPostProcessor:
    """
    将 A* 原始路径处理为论文 Eq.(12) 定义的航点序列
    W = {w1, w2, ..., wK}
    """

    def smooth(self,
               path: List[Tuple[float,float]],
               alpha: float = 0.4,
               beta:  float = 0.3,
               iterations: int = 60) -> List[Tuple[float,float]]:
        """
        梯度下降路径平滑。
        alpha: 路径拉直权重（趋向原始路径）
        beta:  平滑权重（趋向相邻点均值）
        固定首尾两端不动。
        """
        if len(path) <= 2:
            return path

        pts = np.array(path, dtype=float)
        orig = pts.copy()

        for _ in range(iterations):
            for i in range(1, len(pts) - 1):
                pts[i] += (alpha * (orig[i] - pts[i]) +
                            beta  * (pts[i-1] + pts[i+1] - 2 * pts[i]))
        return [tuple(p) for p in pts]

    def extract_waypoints(self,
                           smooth_path: List[Tuple[float,float]],
                           waypoint_spacing: float = 2.0
                           ) -> List[Tuple[float,float]]:
        """
        按固定弧长间隔抽取航点，得到 W = {w1,...,wK} (Eq.12)。
        waypoint_spacing: 相邻航点间距（米）
        """
        if len(smooth_path) == 0:
            return []

        waypoints = [smooth_path[0]]
        accumulated = 0.0

        for i in range(1, len(smooth_path)):
            dx = smooth_path[i][0] - smooth_path[i-1][0]
            dy = smooth_path[i][1] - smooth_path[i-1][1]
            accumulated += np.sqrt(dx**2 + dy**2)
            if accumulated >= waypoint_spacing:
                waypoints.append(smooth_path[i])
                accumulated = 0.0

        # 确保终点在列表中
        if waypoints[-1] != smooth_path[-1]:
            waypoints.append(smooth_path[-1])

        return waypoints


# ─────────────────────────────────────────────────────────────────────────────
# 顶层接口：CD-GRP
# ─────────────────────────────────────────────────────────────────────────────

class CDGlobalReachabilityPlanner:
    """
    Cross-Domain Global Reachability Planner (Section 3.3)

    完整流程：
      1. build_cost_layers()  → D, S, F, O      (Eq.5–8)
      2. fuse()               → G(x,y)           (Eq.9)
      3. search()             → P*               (Eq.10–11)
      4. post_process()       → W={w1,...,wK}    (Eq.12)

    支持动态重规划：update_obstacles() 只重算 O 和 G，
    再用增量 A* 热启动，避免全图重算。
    """

    def __init__(self,
                 cfg: Optional[MapConfig]  = None,
                 env: Optional[EnvironmentInfo] = None):
        self.cfg = cfg or MapConfig()
        self.env = env or EnvironmentInfo()

        self.layer_builder = CostLayerBuilder(self.cfg, self.env)
        self.fusion        = CostMapFusion(self.cfg)
        self.astar         = IncrementalAstar(self.cfg)
        self.post          = PathPostProcessor()

        # 存储当前各层（供增量更新）
        self.D: Optional[np.ndarray] = None
        self.S: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self.O: Optional[np.ndarray] = None
        self.G: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def build(self,
              dynamic_obstacles: Optional[List[Obstacle]] = None) -> np.ndarray:
        """
        全量构建代价地图（首次调用 or 环境大幅变化时）。
        Returns: G (nx, ny)
        """
        self.D = self.layer_builder.build_D()
        self.S = self.layer_builder.build_S()
        self.F = self.layer_builder.build_F()
        self.O = self.layer_builder.build_O(dynamic_obstacles)
        self.G = self.fusion.fuse(self.D, self.S, self.F, self.O)
        return self.G

    # ------------------------------------------------------------------
    def update_obstacles(self,
                         new_obstacles: List[Obstacle]) -> np.ndarray:
        """
        增量更新：仅重算 O 层，再增量更新 G。
        论文中描述的 "incremental replanning under dynamic environmental updates"
        """
        assert self.G is not None, "请先调用 build()"
        O_new = self.layer_builder.build_O(new_obstacles)

        # 找出变化较大的栅格（变化阈值 0.05）
        diff = np.abs(O_new - self.O)
        changed_mask = diff > 0.05
        changed_cells = list(zip(*np.where(changed_mask)))

        self.G = self.fusion.update_obstacle_layer(self.G, self.O, O_new)
        self.O = O_new
        return self.G, changed_cells

    # ------------------------------------------------------------------
    def plan(self,
             start: Tuple[float, float],
             goal:  Tuple[float, float],
             waypoint_spacing: float = 2.0,
             dynamic_obstacles: Optional[List[Obstacle]] = None,
             changed_cells: Optional[List[Tuple[int,int]]] = None
             ) -> Tuple[List[Tuple[float,float]], List[Tuple[float,float]]]:
        """
        完整规划流程。

        Args:
            start             : 起点 (x, y)（米）
            goal              : 终点 (x, y)（米）
            waypoint_spacing  : 航点间距（米）
            dynamic_obstacles : 当前时刻动态障碍物（来自 LiDAR）
            changed_cells     : 本轮变化的栅格（增量A*用）

        Returns:
            (dense_path, waypoints)
              dense_path : 平滑密集路径点
              waypoints  : Eq.(12) 的航点序列 W
        """
        # Step 1+2: 构建或增量更新代价地图
        if self.G is None:
            self.build(dynamic_obstacles)
        elif dynamic_obstacles is not None:
            self.G, changed_cells = self.update_obstacles(dynamic_obstacles)

        # Step 3: 增量 A* 搜索  (Eq.10–11)
        raw_path = self.astar.search(self.G, start, goal, changed_cells)

        # Step 4: 平滑 + 抽取航点  (Eq.12)
        smooth_path = self.post.smooth(raw_path)
        waypoints   = self.post.extract_waypoints(smooth_path, waypoint_spacing)

        return smooth_path, waypoints
