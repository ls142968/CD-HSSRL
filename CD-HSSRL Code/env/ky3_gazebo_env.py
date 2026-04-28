#!/usr/bin/env python3
"""
ky3_gazebo_env.py
基于 ky_3 + Gazebo 的两栖导航 Gym 环境

状态向量 s_t (dim=17):
  [0:3]  位置       (x, y, z)
  [3:6]  线速度     (vx, vy, vz)
  [6]    偏航角     yaw
  [7]    到目标距离  d_goal
  [8:12] 超声波距离  (us1, us2, us3, us4)
  [12:15] 加速度    (ax, ay, az)
  [15]   水深       depth
  [16]   域标签     domain (0/1/2)

动作空间 a_t (dim=2):
  [0] vx_norm  前进速度 ∈ [-1, 1]
  [1] wz_norm  偏航角速度 ∈ [-1, 1]

奖励:
  到达目标:  +100
  碰撞:      -50
  超时:      -10
  进度:      +1.0 * Δdist（靠近目标）
  时间惩罚:  -0.05/step
"""

import numpy as np
import rospy

from ky3_ros_interface import KY3ROSInterface


# ─────────────────────────────────────────────────────────────────────────────
# 任务配置
# ─────────────────────────────────────────────────────────────────────────────

# 论文场景：从水中出发，穿越海岸线，到达陆地目标
# 根据 demo_cross_river.world 的实际坐标调整
TASK_CONFIGS = {
    'water_to_land': {
        'start': {'x': -8.0, 'y': 0.0, 'z': 0.5},   # 水中起点
        'goal':  np.array([8.0, 0.0]),                 # 陆地终点 (x,y)
    },
    'land_to_water': {
        'start': {'x': 8.0, 'y': 0.0, 'z': 0.1},    # 陆地起点
        'goal':  np.array([-8.0, 0.0]),                # 水中终点
    },
    'multi_transition': {
        'start': {'x': -8.0, 'y': 0.0, 'z': 0.5},   # 水中→陆地→水中
        'goal':  np.array([-6.0, 5.0]),
    },
}

STATE_DIM  = 25
ACTION_DIM = 2


class KY3GazeboEnv:
    """
    ky_3 两栖机器人 Gazebo 训练环境（Gym 风格接口）。
    """

    def __init__(self,
                 task:          str   = 'water_to_land',
                 max_steps:     int   = 500,
                 goal_radius:   float = 1.0,
                 control_freq:  float = 20.0,
                 node_init:     bool  = True):
        """
        Args:
            task        : 任务名称 ('water_to_land' / 'land_to_water' / 'multi_transition')
            max_steps   : 单 episode 最大步数
            goal_radius : 到达目标的判定半径 (m)
            control_freq: 控制频率 (Hz)
            node_init   : 是否在此处初始化 ROS 节点
        """
        self.task_cfg   = TASK_CONFIGS[task]
        self.max_steps  = max_steps
        self.goal       = self.task_cfg['goal']
        self.goal_radius= goal_radius
        self.dt         = 1.0 / control_freq

        # 奖励系数
        self.r_goal      =  100.0
        self.r_collision =  -50.0
        self.r_timeout   =  -10.0
        self.r_progress  =    1.0
        self.r_time      =   -0.05

        # ROS 接口
        self.ros = KY3ROSInterface(node_already_init=not node_init)
        self._rate = rospy.Rate(control_freq)

        # Episode 状态
        self._step_count  = 0
        self._prev_dist   = None
        self._prev_pos    = None

        # 统计（跨 episode）
        self.n_episodes   = 0
        self.n_success    = 0
        self.n_collision  = 0
        self._ep_rewards  = []
        self._ep_lengths  = []
        self._ep_paths    = []
        self._ep_energies = []

        rospy.loginfo(f"[KY3Env] 任务: {task}  目标: {self.goal}")

    # ──────────────────────────────────────────────────────────────
    # 状态构建
    # ──────────────────────────────────────────────────────────────

    def _build_state(self) -> np.ndarray:
        """
        从 RobotState 构建论文定义的 17 维状态向量。
        """
        snap = self.ros.get_state()
        pos  = snap['pos']    # [x, y, z]
        vel  = snap['vel']    # [vx, vy, vz]
        rpy  = snap['rpy']    # [roll, pitch, yaw]
        acc  = snap['imu_accel'] if 'imu_accel' in snap else np.zeros(3)
        us   = snap['ultrasonic']  # [u1, u2, u3, u4]
        depth= snap['depth']
        domain = float(self.ros.get_domain_label())

        d_goal = float(np.linalg.norm(pos[:2] - self.goal))

        state = np.array([
            pos[0], pos[1], pos[2],   # [0:3]  位置
            vel[0], vel[1], vel[2],   # [3:6]  线速度
            rpy[2],                   # [6]    偏航角
            d_goal,                   # [7]    到目标距离
            us[0], us[1], us[2], us[3],  # [8:12] 超声波
            acc[0], acc[1], acc[2],   # [12:15] 加速度
            depth,                    # [15]   水深
            domain,                   # [16]   域标签
        ], dtype=np.float32)

        return state

    # ──────────────────────────────────────────────────────────────
    # Gym 接口
    # ──────────────────────────────────────────────────────────────

    def reset(self, task: str = None) -> np.ndarray:
        """
        复位机器人到任务起点，返回初始状态。
        """
        if task is not None and task in TASK_CONFIGS:
            self.task_cfg = TASK_CONFIGS[task]
            self.goal     = self.task_cfg['goal']

        start = self.task_cfg['start']
        self.ros.reset(
            x=start['x'], y=start['y'], z=start['z'])

        self._step_count  = 0
        self._ep_reward   = 0.0
        self._ep_path     = 0.0
        self._ep_energy   = 0.0

        state = self._build_state()
        self._prev_dist = float(np.linalg.norm(
            state[:2] - self.goal))
        self._prev_pos  = state[:2].copy()

        return state

    def step(self, action: np.ndarray
             ) -> tuple:
        """
        执行一步控制。

        Args:
            action: [vx_norm, wz_norm] ∈ [-1, 1]²

        Returns:
            (next_state, reward, done, info)
        """
        domain_label = self.ros.get_domain_label()

        # 发送动作
        self.ros.apply_action(action, domain_label)

        # 等待一个控制周期
        self._rate.sleep()

        # 读取新状态
        next_state   = self._build_state()
        snap         = self.ros.get_state()
        collision    = snap['collision']
        pos          = next_state[:2]
        dist         = float(np.linalg.norm(pos - self.goal))

        # ── 奖励计算 ──────────────────────────────────────────────
        reached = dist < self.goal_radius
        timeout = self._step_count >= self.max_steps - 1

        if reached:
            reward = self.r_goal
            done   = True
            result = 'success'
        elif collision:
            reward = self.r_collision
            done   = True
            result = 'collision'
        elif timeout:
            reward = self.r_timeout
            done   = True
            result = 'timeout'
        else:
            progress = self._prev_dist - dist          # 靠近目标为正
            reward   = self.r_progress * progress + self.r_time
            done     = False
            result   = 'running'

        self._prev_dist = dist

        # ── 统计 ──────────────────────────────────────────────────
        self._ep_reward += reward
        self._ep_energy += float(np.sum(action ** 2))
        if self._prev_pos is not None:
            self._ep_path += float(np.linalg.norm(pos - self._prev_pos))
        self._prev_pos = pos.copy()
        self._step_count += 1

        info = {
            'result':       result,
            'domain':       int(domain_label),
            'dist_to_goal': dist,
            'step':         self._step_count,
        }

        if done:
            self.n_episodes += 1
            if result == 'success':  self.n_success   += 1
            if result == 'collision':self.n_collision  += 1
            self._ep_rewards.append(self._ep_reward)
            self._ep_lengths.append(self._step_count)
            self._ep_paths.append(self._ep_path)
            self._ep_energies.append(self._ep_energy)

            info['ep_reward'] = self._ep_reward
            info['ep_length'] = self._step_count
            info['ep_path']   = self._ep_path
            info['ep_energy'] = self._ep_energy

        return next_state, reward, done, info

    # ──────────────────────────────────────────────────────────────
    # 评估指标  (Eq.23-28)
    # ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """
        计算论文 Section 4.6 的评估指标。
        SR (Eq.23), CR (Eq.24), APL (Eq.26), EC (Eq.28)
        """
        N = self.n_episodes
        if N == 0:
            return {}
        return {
            'SR':  round(self.n_success   / N, 4),   # Eq.23
            'CR':  round(self.n_collision  / N, 4),   # Eq.24
            'APL': round(np.mean(self._ep_paths),   2) if self._ep_paths   else 0.0,  # Eq.26
            'EC':  round(np.mean(self._ep_energies),2) if self._ep_energies else 0.0, # Eq.28
            'N':   N,
        }

    def close(self):
        self.ros.stop()
