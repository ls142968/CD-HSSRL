"""
amphibious_sim_env.py — 两栖导航仿真环境
对标论文 Section 4.1 描述的 Gazebo 仿真配置，
在无 ROS 情况下提供等价的物理行为。

物理模型：
  水域  — UUV Fossen 水动力模型（线性+二次阻尼）
  过渡区 — 坡面接触力 + 阻尼过渡
  陆地  — 刚体差速驱动 + 滑动摩擦

"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict


# ─────────────────────────────────────────────────────────────────────────────
# 物理常数（对应论文 Section 4.1）
# ─────────────────────────────────────────────────────────────────────────────

RHO_WATER  = 1025.0       # kg/m³ 海水密度
GRAVITY    = 9.81         # m/s²
ROBOT_MASS = 10.0         # kg（来自 ky_3.xacro inertial mass）
ROBOT_VOL  = 0.0155       # m³（来自 ky_3.xacro UUV plugin volume）

# Fossen 水动力参数（来自 ky_3.xacro）
ADDED_MASS   = np.diag([10.5, 24.7, 28.57])        # 附加质量矩阵
LINEAR_DAMP  = np.diag([12.03, 20.22, 16.18])      # 线性阻尼
QUAD_DAMP    = np.diag([28.18, 29.66, 46.99])       # 二次阻尼

# 陆地摩擦（来自 ky_3.xacro gazebo reference mu1=1.2）
MU_LAND       = 1.2
MU_TRANSITION = 0.5

# 控制参数
DT            = 0.05    # 控制步长 20Hz
MAX_LIN_VEL   = 1.5     # m/s
MAX_ANG_VEL   = 2.0     # rad/s
MAX_THRUST     = 80.0   # N（单侧最大推力）

# 域边界
SHORE_X       = 0.0     # 海岸线 x 坐标
SLOPE_HALF_W  = 2.5     # 过渡区半宽 m
WATER_Z_THR   = -0.05   # z < 此值视为水中
LAND_Z_THR    =  0.05   # z > 此值视为陆地


# ─────────────────────────────────────────────────────────────────────────────
# 障碍物
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimObstacle:
    x:      float
    y:      float
    radius: float = 0.8
    is_dynamic: bool = False     # 是否动态障碍（MVTD 场景用）
    vx:     float = 0.0
    vy:     float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 仿真环境配置
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    # 任务设置
    task:         str   = 'water_to_land'
    max_steps:    int   = 500
    goal_radius:  float = 1.0
    control_freq: float = 20.0

    # 起点 / 终点
    start_pos: np.ndarray = field(
        default_factory=lambda: np.array([-12.0, 0.0, -0.3]))
    goal_pos:  np.ndarray = field(
        default_factory=lambda: np.array([ 12.0, 0.0]))

    # 奖励系数
    r_goal:      float =  100.0
    r_collision: float =  -50.0
    r_timeout:   float =  -10.0
    r_progress:  float =    1.0
    r_time:      float =   -0.05

    # 物理噪声
    pos_noise:    float = 0.01   # m
    vel_noise:    float = 0.02   # m/s
    sensor_noise: float = 0.05   # m（超声波）

    # 水流扰动（鲁棒性实验用）
    current_vel:  float = 0.0    # m/s
    current_dir:  float = 0.0    # rad

    # 障碍物
    obstacles: List[SimObstacle] = field(default_factory=lambda: [
        SimObstacle(-10.0,  4.0, 1.0),
        SimObstacle( -6.0, -3.0, 0.8),
        SimObstacle( -3.0,  2.5, 0.6),
        SimObstacle(  5.0,  3.5, 0.9),
        SimObstacle(  9.0, -3.0, 1.1),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 物理引擎
# ─────────────────────────────────────────────────────────────────────────────

class AmphibiousPhysics:
    """
    两栖机器人物理模型。

    水域：Fossen 水动力方程
      M_A * v̇ = F_thrust + F_buoyancy + F_gravity
                - D_lin*v - D_quad*|v|*v + F_current

    陆地：刚体差速运动学
      ẋ = v*cos(yaw), ẏ = v*sin(yaw), ψ̇ = omega

    过渡区：线性插值 + 坡面法向力
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

        # 有效质量（含附加质量）
        self._M_eff = ROBOT_MASS * np.eye(3) + ADDED_MASS   # (3,3)

    # ------------------------------------------------------------------
    def buoyancy_force(self, z: float) -> float:
        """
        浮力（z轴方向）。
        完全浸入时 F_b = ρ·g·V_body。
        过渡区线性过渡。
        """
        submerge_ratio = np.clip((-z) / 0.3, 0.0, 1.0)  # 0.3m 为机器人高度
        return RHO_WATER * GRAVITY * ROBOT_VOL * submerge_ratio

    def terrain_height(self, x: float) -> float:
        """
        地形高度：水域=-0.5m，坡面线性过渡，陆地=0m。
        """
        if x < SHORE_X - SLOPE_HALF_W:
            return -0.5
        elif x < SHORE_X + SLOPE_HALF_W:
            t = (x - (SHORE_X - SLOPE_HALF_W)) / (2 * SLOPE_HALF_W)
            return -0.5 + 0.5 * t    # 从 -0.5 线性增到 0
        else:
            return 0.0

    def domain_label(self, x: float, z: float) -> int:
        """0=水 1=过渡 2=陆。"""
        if z < WATER_Z_THR:   return 0
        if z > LAND_Z_THR:    return 2
        return 1

    def step_water(self, pos: np.ndarray, vel: np.ndarray,
                   yaw: float, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        水中一步：Fossen 水动力模型。
        action = [vx_norm, wz_norm]
        Returns: (new_vel_body, acc_body, new_yaw)
        """
        v_cmd  = action[0] * MAX_LIN_VEL
        wz_cmd = action[1] * MAX_ANG_VEL

        # 推力（前进 + 转向通过差动实现）
        F_x = v_cmd  * MAX_THRUST / MAX_LIN_VEL
        M_z = wz_cmd * 5.0    # 偏航力矩系数

        # 当前速度（机体坐标系）
        vb = np.array([vel[0]*np.cos(yaw)+vel[1]*np.sin(yaw),
                        -vel[0]*np.sin(yaw)+vel[1]*np.cos(yaw),
                        vel[2]])

        # 阻力 F_drag = -(D_lin + D_quad*|v|)*v
        v3  = vb[:3]
        drag = -(LINEAR_DAMP @ v3 + QUAD_DAMP @ (np.abs(v3) * v3))

        # 水流扰动
        cur_x = self.cfg.current_vel * np.cos(self.cfg.current_dir)
        cur_y = self.cfg.current_vel * np.sin(self.cfg.current_dir)

        # 加速度（牛顿第二定律，简化为 2D + 垂直）
        acc_x = (F_x + drag[0]) / (ROBOT_MASS + ADDED_MASS[0,0])
        acc_y = drag[1]         / (ROBOT_MASS + ADDED_MASS[1,1])

        # 更新机体速度（一阶积分）
        new_vbx = np.clip(vb[0] + acc_x * DT, -MAX_LIN_VEL, MAX_LIN_VEL)
        new_vby = np.clip(vb[1] + acc_y * DT, -MAX_LIN_VEL, MAX_LIN_VEL)

        # 转换回世界坐标系
        new_vx  = new_vbx * np.cos(yaw) - new_vby * np.sin(yaw) + cur_x
        new_vy  = new_vbx * np.sin(yaw) + new_vby * np.cos(yaw) + cur_y
        new_yaw = yaw + wz_cmd * DT

        acc_body = np.array([acc_x, acc_y, -GRAVITY + 
                             self.buoyancy_force(pos[2]) / (ROBOT_MASS + ADDED_MASS[2,2])])
        return np.array([new_vx, new_vy, 0.0]), acc_body, new_yaw

    def step_land(self, vel: np.ndarray, yaw: float,
                  action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        陆地一步：刚体差速运动学 + 摩擦。
        """
        v_cmd  = action[0] * MAX_LIN_VEL
        wz_cmd = action[1] * MAX_ANG_VEL

        # 摩擦减速
        fric_coef = MU_LAND * GRAVITY * DT
        cur_spd   = np.linalg.norm(vel[:2])
        if cur_spd > 1e-3:
            fric_acc = -fric_coef * vel[:2] / cur_spd
        else:
            fric_acc = np.zeros(2)

        # 期望速度（差速运动学）
        target_vx = v_cmd * np.cos(yaw)
        target_vy = v_cmd * np.sin(yaw)

        # 一阶跟踪（20Hz）
        alpha  = 0.7   # 响应系数
        new_vx = alpha * target_vx + (1-alpha) * vel[0] + fric_acc[0] * DT
        new_vy = alpha * target_vy + (1-alpha) * vel[1] + fric_acc[1] * DT
        new_yaw = yaw + wz_cmd * DT

        acc_body = np.array([(new_vx-vel[0])/DT, (new_vy-vel[1])/DT, 0.0])
        return np.array([new_vx, new_vy, 0.0]), acc_body, new_yaw

    def step_transition(self, pos: np.ndarray, vel: np.ndarray,
                        yaw: float, action: np.ndarray, domain: int
                        ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        过渡区：水动力与陆地运动学的加权插值。
        """
        t = np.clip((pos[0] - (SHORE_X-SLOPE_HALF_W)) / (2*SLOPE_HALF_W), 0, 1)
        vw, aw, yw = self.step_water(pos, vel, yaw, action * 0.6)
        vl, al, yl = self.step_land(vel, yaw, action * 0.6)
        new_vel = (1-t) * vw + t * vl
        new_acc = (1-t) * aw + t * al
        new_yaw = (1-t) * yw + t * yl
        return new_vel, new_acc, new_yaw


# ─────────────────────────────────────────────────────────────────────────────
# 传感器模型
# ─────────────────────────────────────────────────────────────────────────────

class SensorModel:
    """
    模拟 ky_3 五路传感器输出（与 ROS 接口对齐）：
      1. GPS         — 绝对位置（等矩形投影到局部 x,y）
      2. IMU         — 线加速度 + 角速度（含高斯噪声）
      3. LiDAR       — 360°扫描压缩为 8扇区最小距离（障碍物感知）
      4. 超声波(4路) — 向下发射，感知水底距离
      5. 深度传感器  — 压力计换算机器人当前在水中深度
    """
    US_ANGLES     = np.radians([-75, -25, 25, 75])   # 超声波 4路角度
    US_MAX_RNG    = 5.0    # m
    LIDAR_MAX_RNG = 10.0   # m
    N_SECTORS     = 8      # LiDAR 扇区数

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def gps(self, pos: np.ndarray) -> np.ndarray:
        """
        GPS 传感器：机器人位置加 GPS 噪声（水平±0.5m，垂直±1m）。
        返回 (x, y, z)，单位 m。
        """
        noise = self.rng.normal(0, [0.5, 0.5, 1.0])
        return pos + noise.astype(np.float32)

    def imu(self, acc_body: np.ndarray, yaw_rate: float) -> Tuple[np.ndarray, float]:
        """
        IMU 传感器：线加速度 + 角速度，加高斯噪声。
        """
        acc_noise = self.rng.normal(0, 0.05, 3)
        gyro_noise = self.rng.normal(0, 0.01)
        return acc_body + acc_noise, yaw_rate + gyro_noise

    def lidar(self, pos: np.ndarray, yaw: float,
              obstacles) -> np.ndarray:
        """
        LiDAR 传感器：模拟 360° 扫描，压缩为 8扇区（每扇区 45°）最小距离。
        用于障碍物感知（水中浮标、陆地岩石等）。
        返回 shape (8,)，单位 m。
        """
        sectors = np.full(self.N_SECTORS, self.LIDAR_MAX_RNG, dtype=np.float32)
        for obs in obstacles:
            dist = np.sqrt((pos[0]-obs.x)**2 + (pos[1]-obs.y)**2)
            if dist > self.LIDAR_MAX_RNG + obs.radius:
                continue
            # 计算障碍物相对角度
            angle_to_obs = np.arctan2(obs.y - pos[1], obs.x - pos[0]) - yaw
            angle_to_obs = (angle_to_obs + np.pi) % (2*np.pi) - np.pi
            # 找对应扇区
            sector_idx = int((angle_to_obs + np.pi) / (2*np.pi) * self.N_SECTORS)
            sector_idx = np.clip(sector_idx, 0, self.N_SECTORS-1)
            surface_dist = max(0.1, dist - obs.radius)
            if surface_dist < sectors[sector_idx]:
                sectors[sector_idx] = surface_dist
        # 加测距噪声
        noise = self.rng.normal(0, 0.08, self.N_SECTORS)
        sectors = np.clip(sectors + noise, 0.1, self.LIDAR_MAX_RNG)
        return sectors

    def pressure(self, z: float) -> float:
        """
        深度传感器（压力计）：机器人当前在水中的深度。
        depth = max(0, -z)，z < 0 表示在水面以下。
        返回单位 m，> 0 表示在水面以下。
        """
        depth = max(0.0, -z)
        return float(depth + self.rng.normal(0, 0.01))

    def ultrasonic(self, pos: np.ndarray, yaw: float,
                   obstacles) -> np.ndarray:
        """
        超声波传感器（4路）：向下发射，感知水底距离。
        角度：-75°, -25°, +25°, +75°（机体坐标系）。
        水底深度 ≈ |z| + 水底高程（仿真中水底在 z=-0.5m）。
        也兼顾水下障碍物检测。
        返回 shape (4,)，单位 m。
        """
        readings = np.full(4, self.US_MAX_RNG)
        for i, ang in enumerate(self.US_ANGLES):
            world_ang = yaw + ang
            dx = np.cos(world_ang)
            dy = np.sin(world_ang)
            min_dist = self.US_MAX_RNG
            for obs in obstacles:
                # 射线与圆的交点
                ox = obs.x - pos[0]
                oy = obs.y - pos[1]
                proj = ox*dx + oy*dy
                if proj < 0:
                    continue
                perp_sq = ox**2 + oy**2 - proj**2
                if perp_sq > obs.radius**2:
                    continue
                dist = proj - np.sqrt(max(0, obs.radius**2 - perp_sq))
                if 0.1 <= dist < min_dist:
                    min_dist = dist
            noise = self.rng.normal(0, self.cfg.sensor_noise)
            readings[i] = np.clip(min_dist + noise, 0.1, self.US_MAX_RNG)
        return readings


# ─────────────────────────────────────────────────────────────────────────────
# 主仿真环境
# ─────────────────────────────────────────────────────────────────────────────

class AmphibiousSimEnv:
    """
    两栖导航仿真环境（Gym 风格）。
    完全替代 Gazebo，用于 HSSP + SCCC 端到端训练。

    状态向量：17维（与 KY3GazeboEnv 完全一致）
    动作空间：[v_lin, v_ang] ∈ [-1,1]²
    """

    STATE_DIM  = 25
    ACTION_DIM = 2

    def __init__(self, cfg: SimConfig, seed: int = 0):
        self.cfg    = cfg
        self.rng    = np.random.default_rng(seed)
        self.physics = AmphibiousPhysics(cfg, self.rng)
        self.sensors = SensorModel(cfg, self.rng)
        self.dt      = DT

        # 动态障碍物
        self._obstacles = list(cfg.obstacles)

        # Episode 状态
        self._pos  = np.zeros(3)
        self._vel  = np.zeros(3)
        self._yaw  = 0.0
        self._acc  = np.zeros(3)
        self._step = 0
        self._prev_dist = 0.0
        self._prev_pos  = np.zeros(2)

        # Episode 统计
        self.n_episodes  = 0
        self.n_success   = 0
        self.n_collision = 0
        self._ep_path    = 0.0
        self._ep_energy  = 0.0
        self._ep_reward  = 0.0
        self._ep_lengths: List[int]   = []
        self._ep_paths:   List[float] = []
        self._ep_energies:List[float] = []
        self._ep_rewards: List[float] = []
        self._switch_counts: List[int] = []

    # ------------------------------------------------------------------
    def _build_state(self) -> np.ndarray:
        """
        构建 25 维状态向量（与 KY3ROSInterface.build_state 对齐）：
          [0:3]   GPS (x,y,z)         — 绝对定位（含噪声）
          [3:6]   IMU 线加速度 (ax,ay,az) — 运动状态
          [6:9]   IMU 角速度  (wx,wy,wz) — 姿态变化率
          [9]     偏航角 yaw
          [10]    目标距离 d_goal
          [11:19] LiDAR 8扇区最小距离   — 障碍物感知
          [19:23] 超声波 4路            — 水底深度感知
          [23]    深度传感器 water_depth — 机器人在水中深度
          [24]    域标签 domain (0/1/2)
        """
        x, y, z = self._pos
        domain   = self.physics.domain_label(x, z)

        # 1. GPS（含噪声）
        gps_pos  = self.sensors.gps(self._pos)

        # 2. IMU：线加速度 + 角速度
        acc_noisy, gyro_noisy = self.sensors.imu(self._acc, self._vel[1])
        # IMU 角速度（简化：使用速度分量模拟）
        gyro = np.array([0.0, 0.0, gyro_noisy], dtype=np.float32)

        # 3. LiDAR（8扇区，障碍物感知）
        lidar_sectors = self.sensors.lidar(self._pos, self._yaw, self._obstacles)

        # 4. 超声波（4路，水底距离感知）
        us = self.sensors.ultrasonic(self._pos, self._yaw, self._obstacles)

        # 5. 深度传感器（机器人在水中的深度）
        water_depth = self.sensors.pressure(z)

        d_goal = float(np.linalg.norm(self._pos[:2] - self.cfg.goal_pos))

        return np.array([
            gps_pos[0], gps_pos[1], gps_pos[2],      # [0:3]   GPS
            acc_noisy[0], acc_noisy[1], acc_noisy[2], # [3:6]   IMU accel
            gyro[0], gyro[1], gyro[2],                # [6:9]   IMU gyro
            self._yaw,                                 # [9]     yaw
            d_goal,                                    # [10]    d_goal
            *lidar_sectors,                            # [11:19] LiDAR 8扇区
            us[0], us[1], us[2], us[3],               # [19:23] 超声波
            water_depth,                               # [23]    深度传感器
            float(domain),                             # [24]    域标签
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, randomize: bool = True) -> np.ndarray:
        """
        重置 Episode。
        randomize=True 时对起点施加小随机扰动（数据增强）。
        """
        start = self.cfg.start_pos.copy()
        if randomize:
            start[:2] += self.rng.normal(0, 0.3, 2)

        self._pos  = start.astype(np.float32)
        self._vel  = np.zeros(3, np.float32)
        self._yaw  = float(self.rng.uniform(-0.1, 0.1))
        self._acc  = np.zeros(3, np.float32)
        self._step = 0

        self._prev_dist = float(np.linalg.norm(start[:2] - self.cfg.goal_pos))
        self._prev_pos  = start[:2].copy()
        self._ep_path   = 0.0
        self._ep_energy = 0.0
        self._ep_reward = 0.0

        # 更新动态障碍物位置
        self._update_dynamic_obstacles()

        return self._build_state()

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步控制。
        Returns: (next_state, reward, done, info)
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        x, y, z = self._pos
        domain  = self.physics.domain_label(x, z)

        # ── 物理更新 ─────────────────────────────────────────────
        if domain == 0:
            new_vel, self._acc, new_yaw = self.physics.step_water(
                self._pos, self._vel, self._yaw, action)
        elif domain == 2:
            new_vel, self._acc, new_yaw = self.physics.step_land(
                self._vel, self._yaw, action)
        else:
            new_vel, self._acc, new_yaw = self.physics.step_transition(
                self._pos, self._vel, self._yaw, action, domain)

        # ── 位置更新 ─────────────────────────────────────────────
        noise_pos   = self.rng.normal(0, self.cfg.pos_noise, 3)
        new_pos     = self._pos.copy()
        new_pos[0] += new_vel[0] * DT + noise_pos[0]
        new_pos[1] += new_vel[1] * DT + noise_pos[1]

        # z 跟随地形（不能低于地面）
        terrain_z   = self.physics.terrain_height(new_pos[0])
        buoy_z      = z + (-GRAVITY + self.physics.buoyancy_force(z) /
                           ROBOT_MASS) * DT * (1 if domain == 0 else 0)
        new_pos[2]  = max(terrain_z, buoy_z + self.rng.normal(0, 0.01))

        # 边界裁剪
        new_pos[0] = np.clip(new_pos[0], -20.0, 20.0)
        new_pos[1] = np.clip(new_pos[1], -15.0, 15.0)

        self._pos = new_pos.astype(np.float32)
        self._vel = (new_vel + self.rng.normal(0, self.cfg.vel_noise, 3)).astype(np.float32)
        self._yaw = float(new_yaw % (2 * np.pi))

        # ── 动态障碍物更新 ───────────────────────────────────────
        self._update_dynamic_obstacles()

        self._step += 1

        # ── 碰撞检测 ─────────────────────────────────────────────
        collision = self._check_collision()

        # ── 终止条件 ─────────────────────────────────────────────
        pos2d   = self._pos[:2]
        d_goal  = float(np.linalg.norm(pos2d - self.cfg.goal_pos))
        reached = d_goal < self.cfg.goal_radius
        timeout = self._step >= self.cfg.max_steps

        # ── 奖励 ─────────────────────────────────────────────────
        progress = self._prev_dist - d_goal
        if reached:
            reward = self.cfg.r_goal
            done   = True
            result = 'success'
        elif collision:
            reward = self.cfg.r_collision
            done   = True
            result = 'collision'
        elif timeout:
            reward = self.cfg.r_timeout
            done   = True
            result = 'timeout'
        else:
            reward = self.cfg.r_progress * progress + self.cfg.r_time
            done   = False
            result = 'running'

        # ── Episode 统计 ─────────────────────────────────────────
        self._ep_reward += reward
        self._ep_energy += float(np.sum(action**2))
        self._ep_path   += float(np.linalg.norm(pos2d - self._prev_pos))
        self._prev_dist  = d_goal
        self._prev_pos   = pos2d.copy()

        if done:
            self.n_episodes += 1
            if result == 'success':  self.n_success   += 1
            if result == 'collision':self.n_collision  += 1
            self._ep_lengths.append(self._step)
            self._ep_paths.append(self._ep_path)
            self._ep_energies.append(self._ep_energy)
            self._ep_rewards.append(self._ep_reward)

        next_state = self._build_state()
        info = {
            'result': result, 'domain': domain,
            'dist_to_goal': d_goal, 'step': self._step,
            'collision': collision,
            'ep_reward': self._ep_reward,
            'ep_path':   self._ep_path,
            'ep_energy': self._ep_energy,
        }
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    def _check_collision(self) -> bool:
        """检测与障碍物的碰撞（机器人半径 0.35m）."""
        robot_r = 0.35
        for obs in self._obstacles:
            dist = np.sqrt((self._pos[0]-obs.x)**2 + (self._pos[1]-obs.y)**2)
            if dist < robot_r + obs.radius:
                return True
        return False

    def _update_dynamic_obstacles(self):
        """更新动态障碍物位置。"""
        for obs in self._obstacles:
            if obs.is_dynamic:
                obs.x += obs.vx * DT
                obs.y += obs.vy * DT
                # 边界反弹
                if abs(obs.x) > 14: obs.vx *= -1
                if abs(obs.y) > 10: obs.vy *= -1

    # ------------------------------------------------------------------
    def get_metrics(self) -> dict:
        """计算论文 Section 4.6 评估指标 (Eq.23-28)。"""
        N = self.n_episodes
        if N == 0:
            return {}
        return {
            'SR':  round(self.n_success  / N, 4),
            'CR':  round(self.n_collision / N, 4),
            'APL': round(np.mean(self._ep_paths),    2) if self._ep_paths    else 0.0,
            'EC':  round(np.mean(self._ep_energies), 2) if self._ep_energies else 0.0,
            'N_episodes': N,
        }

    # ------------------------------------------------------------------
    def set_current(self, vel: float, direction: float = 0.0):
        """设置水流扰动（鲁棒性实验）。"""
        self.cfg.current_vel = vel
        self.cfg.current_dir = direction
        self.physics.cfg = self.cfg

    def add_dynamic_obstacle(self, x, y, radius=0.6, vx=0.2, vy=0.1):
        """添加动态障碍物（MVTD 场景）。"""
        self._obstacles.append(SimObstacle(x,y,radius,True,vx,vy))
