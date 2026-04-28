#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Joy
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from geometry_msgs.msg import Twist

class AmphibiousTeleop8Thrusters:
    def __init__(self):
        # Thruster publishers (uuv_thruster_manager will listen to /thrusters/<id>/input)
        self.thruster_pubs = []
        for i in range(8):
            pub = rospy.Publisher("/ky3_origin/thrusters/{}/input".format(i), FloatStamped, queue_size=10)
            self.thruster_pubs.append(pub)

        # 陆地模式下发布 /cmd_vel
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Joy 订阅 
        rospy.Subscriber("/joy", Joy, self.joy_callback)

        # 模式切换: 0=陆地, 1=水下
        self.mode = 0

    def joy_callback(self, msg):
        # RB键 → 陆地模式, LB 键 → 水下模式
        if msg.buttons[5] == 1:
            self.mode = 0
            rospy.loginfo("Switched to LAND mode")
        elif msg.buttons[4] == 1:
            self.mode = 1
            rospy.loginfo("Switched to WATER mode")

        if self.mode == 0:
            # 陆地模式，用 /cmd_vel 控制轮子
            twist = Twist()
            twist.linear.x = msg.axes[1] * 1.0     # 左摇杆上下 → 前进/后退
            twist.angular.z = msg.axes[3] * 10.0    # 右摇杆左右 → 转向
            self.cmd_vel_pub.publish(twist)

        elif self.mode == 1:
            # 水下模式，用 8 个推进器
            forward = msg.axes[1] * 20.0   # 左摇杆上下 → 前进/后退
            strafe  = msg.axes[0] * 20.0   # 左摇杆左右 → 横移
            yaw     = -1 * msg.axes[3] * 10.0   # 右摇杆左右 → 偏航
            heave   = 0 #msg.axes[4] * 20.0   # 右摇杆上下 → 上浮/下潜

            # Thruster 分配 (示例，可按自己布置改)
            thrust = [0.0] * 8
            thrust[0] = heave     # 左前水平
            thrust[1] = heave      # 右前水平
            thrust[2] = heave             # 左横移
            thrust[3] = heave            # 右横移
            thrust[4] = forward + yaw               # 垂直左前
            thrust[5] = forward - yaw               # 垂直右前
            thrust[6] = forward + yaw               # 垂直左后
            thrust[7] = forward - yaw               # 垂直右后

            # 发布
            for i in range(8):
                msg_out = FloatStamped()
                msg.header.stamp = rospy.Time.now()
                msg_out.data = thrust[i]
                self.thruster_pubs[i].publish(msg_out)

if __name__ == "__main__":
    rospy.init_node("robot_control")
    AmphibiousTeleop8Thrusters()
    rospy.spin()
