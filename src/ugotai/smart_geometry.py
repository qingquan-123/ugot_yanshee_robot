# -*- coding: utf-8 -*-
"""
smart_geometry.py - 智能解算完整流程（完全独立，不依赖 ugot 目录下其他模块）
接收任意数量的动作参数，依次执行几何运动。
命令行用法：
  python smart_geometry.py <动作1> <动作2> <动作3> ...
示例：
  python smart_geometry.py "polygon:4,50,cw" "turn:90,left" "polygon:4,50,cw,backward" "move:30,left" "arm:3"
  python smart_geometry.py "polygon:4,30,cw" "move:30,right" "polygon:5,30,ccw" "arm:3" "rotate:360"

支持的动作类型：
  polygon:<边数>,<边长>,<方向>[,<模式>]   走多边形（边数3-8，边长≤50cm，方向cw=顺时针/ccw=逆时针，模式forward=前进/backward=倒退）
  move:<距离>,<方向>                        平移（方向left/right/forward/backward）
  turn:<角度>,<方向>                        原地转向（角度≤360，方向left/right）
  rotate:<角度>                             旋转（用于360度转圈）
  arm:<时长秒>                              机械臂自定义动作（时长≥3秒）
"""

import sys
import time
from ugot import ugot

# ============================================================
#  配置常量
# ============================================================
ROBOT_IP = "10.233.104.242"

# 运动速度
POLYGON_MOVE_SPEED = 20       # 多边形走边速度
POLYGON_TURN_SPEED = 60       # 多边形转向速度
MOVE_SPEED = 15                # 平移速度
TURN_SPEED = 60                # 转向速度
ROTATE_SPEED = 80              # 旋转速度

# 约束范围
MAX_SIDES = 8                   # 最多8边形
MIN_SIDES = 3                   # 最少3边形
MAX_SIDE_LENGTH = 50            # 最大边长50cm
MAX_TURN_ANGLE = 360            # 最大转向角度
MIN_ARM_DURATION = 3            # 机械臂最少3秒


# ============================================================
#  动作执行函数
# ============================================================
def exec_polygon(got, params):
    """
    走多边形
    params: [边数, 边长, 方向(cw/ccw), 模式(forward/backward可选)]
    原理：多边形外角和=360°，每次转 360÷边数 度，走一条边转一次
    """
    sides = int(params[0])
    side_length = int(params[1])
    direction = params[2].lower()  # cw or ccw
    mode = params[3].lower() if len(params) > 3 else "forward"

    # 约束检查
    sides = max(MIN_SIDES, min(MAX_SIDES, sides))
    side_length = max(1, min(MAX_SIDE_LENGTH, side_length))

    # 每次转向角度 = 360 / 边数
    turn_angle = int(round(360.0 / sides))

    is_backward = (mode == "backward")
    is_clockwise = (direction == "cw")

    print(f"  走{sides}边形 | 边长{side_length}cm | "
          f"{'顺时针' if is_clockwise else '逆时针'} | "
          f"{'倒退' if is_backward else '前进'} | "
          f"每转{turn_angle}°")

    for i in range(sides):
        # 走一条边
        if is_backward:
            got.mecanum_move_speed_times(1, POLYGON_MOVE_SPEED, side_length, 1)
        else:
            got.mecanum_move_speed_times(0, POLYGON_MOVE_SPEED, side_length, 1)
        time.sleep(0.2)

        # 转向（最后一条边走完不需要再转）
        if i < sides - 1:
            if is_clockwise:
                got.mecanum_turn_speed_times(3, POLYGON_TURN_SPEED, turn_angle, 2)  # 右转
            else:
                got.mecanum_turn_speed_times(2, POLYGON_TURN_SPEED, turn_angle, 2)  # 左转
            time.sleep(0.3)

    print(f"  多边形完成")
    return True


def exec_move(got, params):
    """
    平移
    params: [距离, 方向(left/right/forward/backward)]
    """
    distance = int(params[0])
    direction = params[1].lower()

    print(f"  平移{distance}cm | 方向:{direction}")

    if direction == "left":
        got.mecanum_translate_speed_times(-90, MOVE_SPEED, distance, 1)
    elif direction == "right":
        got.mecanum_translate_speed_times(90, MOVE_SPEED, distance, 1)
    elif direction == "forward":
        got.mecanum_move_speed_times(0, MOVE_SPEED, distance, 1)
    elif direction == "backward":
        got.mecanum_move_speed_times(1, MOVE_SPEED, distance, 1)
    else:
        print(f"  未知平移方向: {direction}")
        return False

    time.sleep(0.3)
    return True


def exec_turn(got, params):
    """
    原地转向
    params: [角度, 方向(left/right)]
    """
    angle = int(params[0])
    direction = params[1].lower()

    angle = max(0, min(MAX_TURN_ANGLE, angle))

    print(f"  转向{angle}° | 方向:{direction}")

    if direction == "left":
        got.mecanum_turn_speed_times(2, TURN_SPEED, angle, 2)
    elif direction == "right":
        got.mecanum_turn_speed_times(3, TURN_SPEED, angle, 2)
    else:
        print(f"  未知转向方向: {direction}")
        return False

    time.sleep(0.3)
    return True


def exec_rotate(got, params):
    """
    旋转（用于360度转圈，和turn类似但速度更快）
    params: [角度]
    """
    angle = int(params[0])
    angle = max(0, min(MAX_TURN_ANGLE, angle))

    print(f"  旋转{angle}°")

    # 默认右转（顺时针转圈）
    got.mecanum_turn_speed_times(3, ROTATE_SPEED, angle, 2)
    time.sleep(0.3)
    return True


def exec_arm(got, params):
    """
    机械臂自定义动作（至少3秒）
    params: [时长秒]
    动作：关节2上下运动 + 夹爪开合交替
    """
    duration = int(params[0])
    duration = max(MIN_ARM_DURATION, duration)

    print(f"  机械臂动作 | 时长:{duration}秒")

    start_time = time.time()
    cycle = 0
    while time.time() - start_time < duration:
        cycle += 1
        print(f"    机械臂循环 {cycle}")

        # 关节2抬起
        got.mechanical_single_joint_control(2, 0, 100)
        time.sleep(0.5)
        # 关节2放下
        got.mechanical_single_joint_control(2, 60, 100)
        time.sleep(0.5)
        # 夹爪张开
        got.mechanical_clamp_release()
        time.sleep(0.3)
        # 夹爪闭合
        got.mechanical_clamp_close()
        time.sleep(0.3)

    # 机械臂复位
    print("  机械臂复位")
    got.mechanical_single_joint_control(2, 80, 100)
    got.mechanical_single_joint_control(3, -80, 100)
    time.sleep(0.5)
    return True


# ============================================================
#  动作分发表
# ============================================================
ACTION_HANDLERS = {
    "polygon": exec_polygon,
    "poly": exec_polygon,
    "move": exec_move,
    "turn": exec_turn,
    "rotate": exec_rotate,
    "arm": exec_arm,
}


def parse_and_exec(got, action_str):
    """
    解析并执行单个动作
    格式：类型:参数1,参数2,...  或  func_name(arg1, arg2, ...)
    """
    action_str = action_str.strip()
    if not action_str:
        return False

    # 自动转换括号格式 func(a, b, c) -> func:a,b,c
    if "(" in action_str and action_str.endswith(")"):
        open_idx = action_str.index("(")
        func_name = action_str[:open_idx].strip()
        args_str = action_str[open_idx+1:-1].strip()
        args_list = [a.strip().strip("'").strip('"') for a in args_str.split(",") if a.strip()]
        action_str = func_name + ":" + ",".join(args_list)
        print(f"  [格式转换] {action_str}")

    if ":" not in action_str:
        print(f"  [错误] 动作格式错误（缺少冒号）: {action_str}")
        print(f"  正确格式: 类型:参数1,参数2,...  例如: polygon:4,50,cw")
        return False

    action_type, params_str = action_str.split(":", 1)
    action_type = action_type.strip().lower()
    params = [p.strip() for p in params_str.split(",") if p.strip()]

    if action_type not in ACTION_HANDLERS:
        print(f"  [错误] 未知动作类型: {action_type}")
        print(f"  支持的类型: {', '.join(ACTION_HANDLERS.keys())}")
        return False

    try:
        return ACTION_HANDLERS[action_type](got, params)
    except Exception as e:
        print(f"  [错误] 动作执行失败 [{action_str}]: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
#  主流程
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print(" smart_geometry.py - 智能解算")
        print("=" * 60)
        print("用法: python smart_geometry.py \"<动作1>;<动作2>;<动作3>;...\"")
        print()
        print("示例1（例1）:")
        print('  python smart_geometry.py "polygon:4,50,cw;turn:90,left;polygon:4,50,cw,backward;move:30,left;arm:3"')
        print()
        print("示例2（例2）:")
        print('  python smart_geometry.py "polygon:4,30,cw;move:30,right;polygon:5,30,ccw;arm:3;rotate:360"')
        print()
        print("支持的动作（动作之间用分号分隔）:")
        print("  polygon:<边数>,<边长>,<cw|ccw>[,<forward|backward>]  走多边形（边3-8，边长≤50）")
        print("  move:<距离>,<left|right|forward|backward>              平移")
        print("  turn:<角度>,<left|right>                                原地转向（≤360°）")
        print("  rotate:<角度>                                            旋转（用于转圈）")
        print("  arm:<时长秒>                                              机械臂动作（≥3秒）")
        sys.exit(1)

    # 动作序列：一个字符串，用分号分隔
    action_seq = sys.argv[1]
    actions = [a.strip() for a in action_seq.split(";") if a.strip()]

    print("=" * 60)
    print(" smart_geometry.py - 智能解算启动")
    print("=" * 60)
    print(f"动作序列（共{len(actions)}个）:")
    for i, a in enumerate(actions):
        print(f"  {i + 1}. {a}")

    # 初始化机器人
    got = ugot.UGOT()
    got.initialize(ROBOT_IP)
    print("机器人连接成功")

    # 依次执行每个动作
    success_count = 0
    for i, action_str in enumerate(actions):
        print(f"\n{'─' * 50}")
        print(f" 执行第 {i + 1}/{len(actions)} 个动作: {action_str}")
        print(f"{'─' * 50}")
        if parse_and_exec(got, action_str):
            success_count += 1
        time.sleep(0.3)

    # 总结
    print(f"\n{'=' * 60}")
    print(f" 智能解算完成：{success_count}/{len(actions)} 个动作成功")
    print(f"{'=' * 60}")

    sys.exit(0 if success_count == len(actions) else 1)


if __name__ == "__main__":
    main()
