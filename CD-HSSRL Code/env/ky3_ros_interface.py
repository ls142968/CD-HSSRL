#!/usr/bin/env python3
"""
ky3_ros_interface.py — ky_3 五传感器 ROS 接口

传感器配置（实际 ky_3.xacro）：
  1. GPS              /ky3_origin/gps          NavSatFix    绝对定位
  2. IMU              /ky3_origin/imu          Imu          姿态+加速度
  3. LiDAR            /ky3_origin/scan         LaserScan    障碍物感知
  4. 超声波(4路)      /ultrasonic1~4           Range        水底深度感知
  5. 深度传感器       /ky3_origin/pressure      FluidPressure 水中深度

状态向量 s_t  dim=25:
  [0:3]   GPS (x,y,z)
  [3:6]   IMU 线加速度 (ax,ay,az)
  [6:9]   IMU 角速度  (wx,wy,wz)
  [9]     偏航角 yaw
  [10]    目标距离 d_goal
  [11:19] LiDAR 8扇区最小距离
  [19:23] 超声波 4路 (水底距离)
  [23]    深度传感器 water_depth (机器人在水中深度)
  [24]    域标签 domain (0水/1过渡/2陆)
"""

import rospy
import numpy as np
from threading import Lock
from std_msgs.msg      import Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg   import Imu, NavSatFix, LaserScan, Range, FluidPressure
from nav_msgs.msg      import Odometry
from gazebo_msgs.msg   import ModelState
from gazebo_msgs.srv   import SetModelState, GetModelState
import tf.transformations as tft

NS           = "ky3_origin"
N_THRUST     = 8
STATE_DIM    = 25
P_ATM        = 101325.0
RHO_G        = 1028.0 * 9.81
WATER_DEPTH_THRESH = 0.05
LAND_HEIGHT_THRESH = 0.05
N_LIDAR_SECTORS    = 8
LIDAR_MAX_RANGE    = 10.0
US_MAX_RANGE       = 5.0
N_ULTRASONIC       = 4
MAX_LIN_VEL  = 1.5
MAX_ANG_VEL  = 2.0
MAX_THRUST   = 80.0


class RobotState:
    def __init__(self):
        self._lock = Lock()
        # GPS
        self.gps_x = self.gps_y = self.gps_z = 0.0
        self.gps_valid = False
        self._gps_ref_lat = self._gps_ref_lon = None
        # IMU
        self.imu_accel = np.zeros(3, np.float32)
        self.imu_gyro  = np.zeros(3, np.float32)
        # LiDAR 8 sectors
        self.lidar_sectors   = np.ones(N_LIDAR_SECTORS, np.float32) * LIDAR_MAX_RANGE
        self.min_obstacle_dist = LIDAR_MAX_RANGE
        # Ultrasonic 4ch
        self.ultrasonic = np.ones(N_ULTRASONIC, np.float32) * US_MAX_RANGE
        # Depth sensor
        self.water_depth = 0.0
        # GT (Gazebo ground truth)
        self.gt_x = self.gt_y = self.gt_z = 0.0
        self.gt_yaw = 0.0
        self.collision = False
        self.last_update = 0.0

    def update_gps(self, msg):
        if msg.status.status < 0: return
        R = 6371000.0
        with self._lock:
            if not self.gps_valid:
                self._gps_ref_lat = msg.latitude
                self._gps_ref_lon = msg.longitude
                self.gps_valid = True
            dlat = np.radians(msg.latitude  - self._gps_ref_lat)
            dlon = np.radians(msg.longitude - self._gps_ref_lon)
            self.gps_x = dlon * R * np.cos(np.radians(self._gps_ref_lat))
            self.gps_y = dlat * R
            self.gps_z = msg.altitude
            self.last_update = rospy.get_time()

    def update_imu(self, msg):
        q = msg.orientation
        rpy = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        with self._lock:
            self.imu_accel = np.array([msg.linear_acceleration.x,
                                       msg.linear_acceleration.y,
                                       msg.linear_acceleration.z], np.float32)
            self.imu_gyro  = np.array([msg.angular_velocity.x,
                                       msg.angular_velocity.y,
                                       msg.angular_velocity.z], np.float32)
            self.gt_yaw = rpy[2]

    def update_lidar(self, msg):
        r = np.array(msg.ranges, np.float32)
        r = np.where(np.isinf(r)|np.isnan(r), LIDAR_MAX_RANGE, r)
        r = np.clip(r, 0., LIDAR_MAX_RANGE)
        n = len(r); sz = max(1, n // N_LIDAR_SECTORS)
        secs = np.array([np.min(r[i*sz:min((i+1)*sz,n)]) for i in range(N_LIDAR_SECTORS)], np.float32)
        with self._lock:
            self.lidar_sectors    = secs
            self.min_obstacle_dist = float(np.min(secs))
            self.collision         = bool(self.min_obstacle_dist < 0.4)

    def update_ultrasonic(self, idx, msg):
        with self._lock:
            self.ultrasonic[idx] = float(np.clip(msg.range, msg.min_range, msg.max_range))

    def update_pressure(self, msg):
        depth = max(0.0, (msg.fluid_pressure - P_ATM) / RHO_G)
        with self._lock:
            self.water_depth = depth

    def update_gt(self, msg):
        q = msg.pose.pose.orientation
        yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        with self._lock:
            p = msg.pose.pose.position
            self.gt_x = p.x; self.gt_y = p.y; self.gt_z = p.z
            self.gt_yaw = yaw
            self.last_update = rospy.get_time()

    def domain_label(self):
        with self._lock:
            if self.water_depth > WATER_DEPTH_THRESH: return 0
            if self.gt_z        > LAND_HEIGHT_THRESH:  return 2
            return 1

    def snapshot(self):
        with self._lock:
            return {
                "gps":        np.array([self.gps_x, self.gps_y, self.gps_z], np.float32),
                "imu_accel":  self.imu_accel.copy(),
                "imu_gyro":   self.imu_gyro.copy(),
                "yaw":        float(self.gt_yaw),
                "lidar":      self.lidar_sectors.copy(),
                "ultrasonic": self.ultrasonic.copy(),
                "water_depth":self.water_depth,
                "gt_pos":     np.array([self.gt_x, self.gt_y, self.gt_z], np.float32),
                "collision":  self.collision,
                "min_obs":    self.min_obstacle_dist,
                "last_update":self.last_update,
            }

    def is_alive(self):
        return (rospy.get_time() - self.last_update) < 1.0


class ThrusterAllocator:
    def __init__(self):
        s = np.sin(np.radians(45))
        self._B = np.array([[s,s,0.3],[s,-s,-0.3],[s,s,-0.3],[s,-s,0.3],
                             [0,1,.5],[0,-1,-.5],[0,1,.5],[0,-1,-.5]])
    def allocate(self, vx, vy, wz):
        t = self._B @ np.array([vx,vy,wz])
        mx = np.abs(t).max()
        if mx>1: t/=mx
        return np.clip(t*MAX_THRUST, -MAX_THRUST, MAX_THRUST)


class KY3ROSInterface:
    def __init__(self, node_already_init=False):
        if not node_already_init:
            rospy.init_node("cd_hssrl_env", anonymous=False)
        self.state     = RobotState()
        self.allocator = ThrusterAllocator()
        self._thrust_pubs = [rospy.Publisher(f"/{NS}/thrusters/{i}/input",
                                              Float64, queue_size=1)
                             for i in range(N_THRUST)]
        self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        # 1. GPS
        rospy.Subscriber(f"/{NS}/gps", NavSatFix, self.state.update_gps, queue_size=1)
        # 2. IMU
        rospy.Subscriber(f"/{NS}/imu", Imu, self.state.update_imu, queue_size=1)
        # 3. LiDAR
        rospy.Subscriber(f"/{NS}/scan", LaserScan, self.state.update_lidar, queue_size=1)
        # 4. 超声波
        for i in range(1, N_ULTRASONIC+1):
            rospy.Subscriber(f"/ultrasonic{i}", Range,
                             lambda msg,idx=i-1: self.state.update_ultrasonic(idx,msg),
                             queue_size=1)
        # 5. 深度传感器
        rospy.Subscriber(f"/{NS}/pressure", FluidPressure,
                         self.state.update_pressure, queue_size=1)
        # GT
        rospy.Subscriber(f"/{NS}/pose_gt", Odometry, self.state.update_gt, queue_size=1)
        rospy.wait_for_service("/gazebo/set_model_state", timeout=10.)
        self._set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        rospy.sleep(0.5)
        rospy.loginfo("[KY3] 五传感器接口就绪 (GPS/IMU/LiDAR/US/Depth)")

    def build_state(self, goal_pos: np.ndarray) -> np.ndarray:
        """构建 25 维状态向量。"""
        s  = self.state.snapshot()
        gps = s["gps"]
        # 用 GT 位置作为实际坐标（训练时）
        gt  = s["gt_pos"]
        d_goal = float(np.linalg.norm(gt[:2] - goal_pos[:2]))
        domain = self.state.domain_label()
        return np.array([
            gt[0], gt[1], gt[2],         # [0:3]  位置 (GT，更精确)
            s["imu_accel"][0],
            s["imu_accel"][1],
            s["imu_accel"][2],            # [3:6]  IMU 加速度
            s["imu_gyro"][0],
            s["imu_gyro"][1],
            s["imu_gyro"][2],             # [6:9]  IMU 角速度
            s["yaw"],                     # [9]    偏航角
            d_goal,                       # [10]   目标距离
            *s["lidar"],                  # [11:19] LiDAR 8扇区
            *s["ultrasonic"],             # [19:23] 超声波 4路 (水底深度)
            s["water_depth"],             # [23]   深度传感器 (水中深度)
            float(domain),                # [24]   域标签
        ], dtype=np.float32)

    def get_domain_label(self): return self.state.domain_label()
    def is_alive(self):         return self.state.is_alive()
    def get_collision(self):    return self.state.snapshot()["collision"]

    def apply_action(self, action, domain):
        vx = float(np.clip(action[0], -1., 1.))
        wz = float(np.clip(action[1], -1., 1.))
        if   domain == 0: self._water(vx, 0., wz)
        elif domain == 1: self._water(vx*.5, 0., wz*.5); self._land(vx*.5, wz*.5)
        else:             self._land(vx, wz)

    def _water(self, vx, vy, wz):
        t = self.allocator.allocate(vx, vy, wz)
        for i,p in enumerate(self._thrust_pubs): p.publish(Float64(data=float(t[i])))

    def _land(self, vx, wz):
        cmd=Twist(); cmd.linear.x=vx*MAX_LIN_VEL; cmd.angular.z=wz*MAX_ANG_VEL
        self._cmd_vel_pub.publish(cmd)

    def stop(self):
        for p in self._thrust_pubs: p.publish(Float64(data=0.))
        self._cmd_vel_pub.publish(Twist())

    def reset(self, x=0., y=0., z=1., yaw=0.):
        self.stop()
        ms=ModelState(); ms.model_name=NS
        ms.pose.position.x=x; ms.pose.position.y=y; ms.pose.position.z=z
        q=tft.quaternion_from_euler(0.,0.,yaw)
        ms.pose.orientation.x=q[0]; ms.pose.orientation.y=q[1]
        ms.pose.orientation.z=q[2]; ms.pose.orientation.w=q[3]
        ms.reference_frame="world"
        try: self._set_state(ms); rospy.sleep(0.3)
        except: pass
