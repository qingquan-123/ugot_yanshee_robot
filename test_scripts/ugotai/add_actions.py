file_path = r"C:\Users\16231\Desktop\ros\gS\ugotai\zncc.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 在 arm_reset 后面加基础动作函数
old_arm_end = '''    print("机器复位完成")


def align_with_apriltag(got):'''

new_arm_end = '''    print("机器复位完成")


def action_forward(got, distance_cm, speed=15):
    """前进指定距离"""
    print(f"  前进 {distance_cm}cm")
    got.mecanum_move_speed_times(0, speed, distance_cm, 1)
    time.sleep(0.5)


def action_backward(got, distance_cm, speed=15):
    """后退指定距离"""
    print(f"  后退 {distance_cm}cm")
    got.mecanum_move_speed_times(1, speed, distance_cm, 1)
    time.sleep(0.5)


def action_turn(got, angle, speed=60):
    """原地转向，正数左转，负数右转"""
    direction = "左转" if angle > 0 else "右转"
    abs_angle = abs(angle)
    print(f"  {direction} {abs_angle}°")
    if angle > 0:
        got.mecanum_turn_speed_times(2, speed, abs_angle, 2)
    else:
        got.mecanum_turn_speed_times(3, speed, abs_angle, 2)
    time.sleep(0.5)


def align_with_apriltag(got):'''

content = content.replace(old_arm_end, new_arm_end)

# 2. 修改 main 函数，支持智能仓储完整流程
old_main = '''if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python zncc.py <target_color> <target_store_id> <need_return>")
        print("示例: python zncc.py 绿色 5 1")
        print("  need_return: 1=7步流程(需要回到取货区), 0=6步流程(不需要返回)")
        sys.exit(1)
    target_color = sys.argv[1]
    target_store_id = int(sys.argv[2])
    need_return = int(sys.argv[3])
    print(f"=== zncc.py 启动 ===")
    print(f"目标颜色: {target_color}")
    print(f"目标存储区: {target_store_id}号")
    print(f"是否需要返回: {'是(7步)' if need_return else '否(6步)'}")
    got = ugot.UGOT()
    got.initialize('10.233.104.242')
    print("机器人连接成功")
    if need_return:
        success, grab_flag, place_flag = warehouse_full_7step(got, target_color, target_store_id)
    else:
        success, grab_flag, place_flag = warehouse_full_6step(got, target_color, target_store_id)
    print(f"\\n执行结果: success={success}, grab_flag={grab_flag}, place_flag={place_flag}")
    sys.exit(0 if success else 1)'''

new_main = '''if __name__ == "__main__":
    print(f"=== zncc.py 启动 ===")

    # 智能仓储完整流程：python zncc.py smart_warehouse <前进cm> <颜色> <放置号> <后退cm> <转向角>
    if len(sys.argv) >= 2 and sys.argv[1] == "smart_warehouse":
        forward_cm = int(sys.argv[2])
        target_color = sys.argv[3]
        target_store_id = int(sys.argv[4])
        backward_cm = int(sys.argv[5])
        turn_angle = int(sys.argv[6])

        print(f"模式: 智能仓储完整流程")
        print(f"  前进: {forward_cm}cm")
        print(f"  抓取颜色: {target_color}")
        print(f"  放置位置: {target_store_id}号")
        print(f"  后退: {backward_cm}cm")
        print(f"  转向: {turn_angle}°")

        got = ugot.UGOT()
        got.initialize('10.233.104.242')
        print("机器人连接成功")

        # 1. 前进 Zcm
        action_forward(got, forward_cm)

        # 2. 抓取并放置（7步流程，需要回到取货区）
        success, grab_flag, place_flag = warehouse_full_7step(got, target_color, target_store_id)
        if not success:
            print("仓储流程失败")
            sys.exit(1)

        # 3. 后退 Lcm
        action_backward(got, backward_cm)

        # 4. 转向
        if turn_angle != 0:
            action_turn(got, turn_angle)

        print(f"\\n智能仓储任务完成: grab_flag={grab_flag}, place_flag={place_flag}")
        sys.exit(0)

    # 原模式：python zncc.py <target_color> <target_store_id> <need_return>
    else:
        if len(sys.argv) < 4:
            print("用法:")
            print("  智能仓储: python zncc.py smart_warehouse <前进cm> <颜色> <放置号> <后退cm> <转向角>")
            print("  基础模式: python zncc.py <target_color> <target_store_id> <need_return>")
            print("示例:")
            print("  python zncc.py smart_warehouse 0 绿色 6 30 90")
            print("  python zncc.py 绿色 5 1")
            sys.exit(1)
        target_color = sys.argv[1]
        target_store_id = int(sys.argv[2])
        need_return = int(sys.argv[3])
        print(f"模式: 基础仓储流程")
        print(f"目标颜色: {target_color}")
        print(f"目标存储区: {target_store_id}号")
        print(f"是否需要返回: {'是(7步)' if need_return else '否(6步)'}")
        got = ugot.UGOT()
        got.initialize('10.233.104.242')
        print("机器人连接成功")
        if need_return:
            success, grab_flag, place_flag = warehouse_full_7step(got, target_color, target_store_id)
        else:
            success, grab_flag, place_flag = warehouse_full_6step(got, target_color, target_store_id)
        print(f"\\n执行结果: success={success}, grab_flag={grab_flag}, place_flag={place_flag}")
        sys.exit(0 if success else 1)'''

content = content.replace(old_main, new_main)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("OK")
