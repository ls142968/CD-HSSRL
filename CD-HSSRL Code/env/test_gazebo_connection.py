#!/usr/bin/env python3
"""
test_gazebo_connection.py
Gazebo 连接测试脚本

使用方法：
  终端1: roslaunch ky_3 gazebo.launch
  终端2: python3 test_gazebo_connection.py

测试内容：
  1. 话题连接（传感器是否在发布）
  2. 服务连接（Gazebo 服务是否可用）
  3. 控制指令（推进器 + cmd_vel）
  4. 状态读取（state vector 是否正常）
  5. 复位功能
"""

import sys
import os
import time

import rospy
import numpy as np

# 把 env 目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def check_topics():
    """检查所有需要的话题是否在发布。"""
    print_section("Step 1: 话题检查")

    import rostopic
    expected = [
        '/ky3_origin/pose_gt',
        '/ky3_origin/imu',
        '/ky3_origin/pressure',
        '/ultrasonic1',
        '/ultrasonic2',
        '/ultrasonic3',
        '/ultrasonic4',
        '/cmd_vel',
    ]
    # 推进器话题
    for i in range(8):
        expected.append(f'/ky3_origin/thrusters/{i}/input')

    all_topics = [t for t, _ in rospy.get_published_topics()]
    ok_count   = 0
    for t in expected:
        found = t in all_topics
        mark  = '✓' if found else '✗'
        print(f"  {mark}  {t}")
        if found:
            ok_count += 1

    print(f"\n  结果: {ok_count}/{len(expected)} 话题正常")
    return ok_count == len(expected)


def check_services():
    """检查 Gazebo 服务是否可用。"""
    print_section("Step 2: Gazebo 服务检查")
    import rosservice

    services = [
        '/gazebo/set_model_state',
        '/gazebo/get_model_state',
        '/gazebo/reset_simulation',
    ]
    ok = True
    for s in services:
        try:
            rospy.wait_for_service(s, timeout=3.0)
            print(f"  ✓  {s}")
        except rospy.ROSException:
            print(f"  ✗  {s}  (超时)")
            ok = False
    return ok


def test_state_reading(ros_if):
    """测试状态读取。"""
    print_section("Step 3: 状态读取测试")

    rospy.sleep(0.5)   # 等待订阅者填充数据
    snap = ros_if.get_state()

    pos    = snap['pos']
    vel    = snap['vel']
    depth  = snap['depth']
    us     = snap['ultrasonic']
    domain = ros_if.get_domain_label()

    domain_name = ['Water', 'Transition', 'Land'][domain]

    print(f"  位置      : x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}  (m)")
    print(f"  线速度    : vx={vel[0]:.3f}  vy={vel[1]:.3f}  vz={vel[2]:.3f}  (m/s)")
    print(f"  水深      : {depth:.3f} m")
    print(f"  超声波    : {us}")
    print(f"  域标签    : {domain} ({domain_name})")
    print(f"  数据新鲜度: last_update={snap['last_update']:.2f}s  alive={ros_if.is_alive()}")

    # 构建 env 并测试 state vector
    from ky3_gazebo_env import KY3GazeboEnv
    env   = KY3GazeboEnv(node_init=False)
    state = env._build_state()
    print(f"\n  State vector (dim={len(state)}):")
    labels = ['x','y','z','vx','vy','vz','yaw','d_goal',
              'us1','us2','us3','us4','ax','ay','az','depth','domain']
    for i, (l, v) in enumerate(zip(labels, state)):
        print(f"    [{i:2d}] {l:8s} = {v:.4f}")

    return env


def test_control(ros_if):
    """测试控制指令发送。"""
    print_section("Step 4: 控制指令测试")

    domain = ros_if.get_domain_label()
    domain_name = ['Water', 'Transition', 'Land'][domain]
    print(f"  当前域: {domain_name}")

    # 发送小推力
    print("  发送前进指令 [vx=0.3, wz=0.0] (1秒)...")
    t_start = time.time()
    while time.time() - t_start < 1.0:
        ros_if.apply_action(np.array([0.3, 0.0]), domain)
        time.sleep(0.05)

    snap_after = ros_if.get_state()
    print(f"  1秒后速度: vx={snap_after['vel'][0]:.3f}  vy={snap_after['vel'][1]:.3f}")

    # 停止
    ros_if.stop()
    print("  停止指令已发送 ✓")


def test_reset(ros_if):
    """测试复位功能。"""
    print_section("Step 5: 复位测试")

    # 记录当前位置
    snap_before = ros_if.get_state()
    print(f"  复位前: ({snap_before['pos'][0]:.2f}, "
          f"{snap_before['pos'][1]:.2f}, {snap_before['pos'][2]:.2f})")

    # 执行复位
    ros_if.reset(x=-8.0, y=0.0, z=0.5)
    rospy.sleep(0.5)

    snap_after = ros_if.get_state()
    print(f"  复位后: ({snap_after['pos'][0]:.2f}, "
          f"{snap_after['pos'][1]:.2f}, {snap_after['pos'][2]:.2f})")

    # 验证复位精度
    err = np.linalg.norm(snap_after['pos'] - np.array([-8.0, 0.0, 0.5]))
    print(f"  位置误差: {err:.3f} m  {'✓' if err < 0.5 else '✗'}")


def test_full_episode(env):
    """测试完整 episode 循环。"""
    print_section("Step 6: 完整 Episode 测试 (10步)")

    state = env.reset()
    print(f"  初始状态 d_goal={state[7]:.2f}m  domain={int(state[16])}")

    for step in range(10):
        # 随机动作测试
        action = np.array([0.3, 0.0])   # 直线前进
        next_state, reward, done, info = env.step(action)
        print(f"  步{step+1:2d}: domain={info['domain']}  "
              f"d_goal={info['dist_to_goal']:.2f}m  "
              f"reward={reward:.2f}  result={info['result']}")
        if done:
            print(f"  Episode 结束: {info['result']}")
            break

    metrics = env.get_metrics()
    print(f"\n  评估指标: {metrics}")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rospy.init_node('cd_hssrl_connection_test', anonymous=False)
    print("\n" + "="*55)
    print("  CD-HSSRL ← ky_3 Gazebo 连接测试")
    print("="*55)
    print("  请确保已运行: roslaunch ky_3 gazebo.launch")

    # Step 1: 话题检查
    topics_ok = check_topics()
    if not topics_ok:
        print("\n  ⚠ 部分话题未找到，请检查 Gazebo 是否已启动")

    # Step 2: 服务检查
    services_ok = check_services()
    if not services_ok:
        print("\n  ⚠ Gazebo 服务不可用，退出")
        return

    # 初始化接口
    print_section("初始化 KY3ROSInterface...")
    from ky3_ros_interface import KY3ROSInterface
    ros_if = KY3ROSInterface(node_already_init=True)

    # Step 3: 状态读取
    env = test_state_reading(ros_if)

    # Step 4: 控制测试
    test_control(ros_if)

    # Step 5: 复位测试
    test_reset(ros_if)

    # Step 6: Episode 测试
    test_full_episode(env)

    print("\n" + "="*55)
    print("  所有测试完成 ✓")
    print("="*55)
    print("\n  下一步: python3 train_cd_hssrl.py")


if __name__ == '__main__':
    main()
