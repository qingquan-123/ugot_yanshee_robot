import time


trun_time = 0
max_rotate_speed = 20#40
max_forward_speed = 30#45
base_forward_speed = 20#35
_pid_turn = None


def _init_pid(u):
    global _pid_turn
    if _pid_turn is None:
        _pid_turn = u.create_pid_controller()
        _pid_turn.set_pid(0.12, 0.001, 0.08)


def quick_align_after_turn(u, enable_forward_slide=True):
    """
    分级步长渐进矫正，解决来回震荡问题
    :param enable_forward_slide: 是否启用结尾向前滑行，近路口场景设为False
    """
    print("开始转向后车身矫正...")
    u.mecanum_stop()
    time.sleep(0.15)

    MAX_ALIGN_ROUNDS = 4
    align_round = 0

    line_info = u.get_single_track_total_info()
    offset, line_type, x, y = line_info
    print(f"矫正前状态 - 偏移量: {offset}, 线路类型: {line_type}")

    if line_type == 0:
        print("未检测到线路，快速回正找线...")
        u.mecanum_move_xyz(0, 0, 10)
        time.sleep(0.2)
        u.mecanum_stop()
        time.sleep(0.1)
        line_info = u.get_single_track_total_info()
        offset, line_type, x, y = line_info

    while line_type in (1, 2) and align_round < MAX_ALIGN_ROUNDS:
        align_round += 1
        print(f"--- 第 {align_round} 轮矫正 ---")

        if abs(offset) <= 5:
            print("偏移量已达标，结束静态矫正")
            break

        # 分级横移，避免矫枉过正
        if abs(offset) > 40:
            step_speed = 7
            print(f"大偏差横移矫正，步速: {step_speed}")
        elif abs(offset) > 20:
            step_speed = 5
            print(f"中偏差横移矫正，步速: {step_speed}")
        else:
            step_speed = 0
            print("小偏差，进入旋转微调")

        if step_speed > 0:
            if offset > 0:
                u.mecanum_translate_speed_times(-90, step_speed, 1, 1)
            else:
                u.mecanum_translate_speed_times(90, step_speed, 1, 1)
            time.sleep(0.25)
        else:
            try:
                rotate_speed = round(_pid_turn.update(offset))
                rotate_speed = int(max(min(rotate_speed, 8), -8))
                print(f"旋转角度微调: {rotate_speed}")
                u.mecanum_move_xyz(0, 0, -rotate_speed)
                time.sleep(0.2)
                u.mecanum_stop()
                time.sleep(0.12)
            except Exception as e:
                print(f"旋转微调异常: {e}")
                break

        line_info = u.get_single_track_total_info()
        offset, line_type, x, y = line_info
        print(f"本轮后状态 - 偏移量: {offset}, 线路类型: {line_type}")

    # ========== 修改：根据参数决定是否执行前滑 ==========
    if enable_forward_slide:
        print("静态矫正完成，低速滑行动态回正...")
        u.mecanum_move_xyz(0, 10, 0)
        time.sleep(0.5)
        u.mecanum_stop()
        time.sleep(0.1)
    else:
        print("静态矫正完成（近路口模式，跳过前滑）")
        u.mecanum_stop()
        time.sleep(0.1)
    # ==================================================

    line_info = u.get_single_track_total_info()
    offset, line_type, x, y = line_info
    print(f"最终状态 - 偏移量: {offset}, 线路类型: {line_type}")

    print("111车身矫正完成")
    return True


def get_rotate_speed(u):
    line_info = u.get_single_track_total_info()
    print(f"巡线信息: 偏移量={line_info[0]}, 线路类型={line_info[1]}")
    offset, line_type, x, y = line_info
    if line_type == 0:
        return 0, line_type, x, y
    rotate_speed = round(_pid_turn.update(offset))
    rotate_speed = int(max(min(rotate_speed, max_rotate_speed), -max_rotate_speed))
    return rotate_speed, line_type, x, y


def pid_find_line(u):
    global trun_time
    _init_pid(u)
    time.sleep(0.5)
    last_rotate_speed = 0
    lost_line_count = 0
    crossroad_cd = 0

    # 官方弯道状态（保留辅助）
    curve_stable_frames = 0
    curve_end_frames = 0
    CURVE_THRESHOLD = 15
    CROSSROAD_WINDOW = 30
    last_offset = 0

    LOST_LINE_THRESHOLD = 30
    CROSSROAD_CD_FRAMES = 2

    # 起步加速缓冲
    start_frames = 0
    START_BUFFER_FRAMES = 30

    # ========== 优化：单帧触发路口警戒 ==========
    crossroad_alert = False      # 路口警戒状态
    alert_frames = 0             # 警戒已持续帧数
    ALERT_TOTAL_FRAMES = 35      # 警戒总时长
    ALERT_FORCE_TRIGGER = 25     # 警戒第25帧强制触发路口
    # 触发警戒的条件：单帧偏移超过该值 或 识别到弯道
    ALERT_OFFSET_THRESHOLD = 90
    # =========================================

    while True:
        if crossroad_cd > 0:
            crossroad_cd -= 1

        rotate_speed, line_type, raw_offset, y = get_rotate_speed(u)

        # 官方弯道状态更新
        if line_type == 2:
            curve_stable_frames += 1
            curve_end_frames = 0
        else:
            if curve_stable_frames >= CURVE_THRESHOLD:
                curve_end_frames += 1
            if curve_end_frames > CROSSROAD_WINDOW:
                curve_stable_frames = 0
                curve_end_frames = 0

        # ========== 路口警戒逻辑 开始 ==========
        # 进入警戒：识别到弯道 或 单帧大偏移，且冷却结束、未到3次路口
        if not crossroad_alert and crossroad_cd == 0 and trun_time < 3:
            if line_type == 2 or abs(raw_offset) > ALERT_OFFSET_THRESHOLD:
                crossroad_alert = True
                alert_frames = 0
                print(" 进入路口警戒状态，降速慢行")

        # 警戒计时
        if crossroad_alert:
            alert_frames += 1
            # 警戒结束，重置
            if alert_frames > ALERT_TOTAL_FRAMES:
                crossroad_alert = False
                alert_frames = 0
                print("路口警戒结束，恢复正常巡线")

        last_offset = rotate_speed

        if line_type in (1, 2, 3):
            lost_line_count = 0
            start_frames += 1

            trigger_crossroad = False
            trigger_reason = ""

            # 保险1：官方识别到十字路口
            if line_type == 3 and crossroad_cd == 0 and trun_time < 3:
                trigger_crossroad = True
                trigger_reason = "官方识别到十字路口"


            if trigger_crossroad:
                print(f" {trigger_reason}，当前转弯计数: {trun_time}")
                u.mecanum_stop()
                time.sleep(0.5)
                trun_time += 1
                curve_stable_frames = 0
                curve_end_frames = 0
                crossroad_alert = False
                alert_frames = 0

                if trun_time == 3:
                    print("第三次十字路口，直行后结束")
                    u.mecanum_move_speed_times(0, 15, 19, 1)
                    print("---------------------------------------------------1111111111111")
                    time.sleep(0.5)
                    break
                elif trun_time ==2:
                    print("第二次十字路口")
                    u.mecanum_move_speed_times(0, 50, 18, 1)
                    u.mecanum_turn_speed_times(3, 150, 91, 2)
                    print("转向完成，低速滑行回正...")
                    u.mecanum_move_xyz(0, 20, 0)
                    time.sleep(0.5)
                    u.mecanum_stop()
                elif trun_time == 1:
                    u.play_audio_tts("", 0, False)
                    u.mecanum_move_speed_times(0, 80, 19, 1)
                    u.mecanum_turn_speed_times(3, 150, 89, 2)

                    print("转向完成，低速滑行回正...")
                    u.mecanum_move_xyz(0, 20, 0)
                    time.sleep(0.3)
                    u.mecanum_stop()

                    if trun_time != 1:
                        print("开始车身矫正...")
                        try:
                            quick_align_after_turn(u)
                            print("矫正完成，继续巡线...")
                        except Exception as e:
                            print(f"矫正异常，直接进入巡线: {e}")
                    else:
                        print("第一次路口后，跳过车身微调，继续巡线...")



                crossroad_cd = CROSSROAD_CD_FRAMES
                start_frames = 0
                continue

            # ========== 巡线 ==========
            smooth_rotate = int(last_rotate_speed * 0.3 + rotate_speed * 0.7)
            last_rotate_speed = smooth_rotate

            # 基础调速
            is_curve = line_type == 2 or curve_end_frames > 0
            if is_curve:
                dynamic_speed = int(base_forward_speed * 0.55 - min(abs(rotate_speed) / 5, 3))
            else:
                dynamic_speed = int(base_forward_speed - min(abs(rotate_speed) / 5, 5))

            # 第2次转弯后，整体降速
            if trun_time >= 2:
                dynamic_speed = int(dynamic_speed * 0.85)

            # 起步缓冲限速
            if start_frames < START_BUFFER_FRAMES:
                speed_limit = 15 + int((dynamic_speed - 15) * (start_frames / START_BUFFER_FRAMES))
                dynamic_speed = min(dynamic_speed, speed_limit)

            # 大偏移强制降速
            if abs(raw_offset) > 80:
                dynamic_speed = min(dynamic_speed, 18)
            elif abs(raw_offset) > 50:
                dynamic_speed = min(dynamic_speed, 22)

            # 警戒状态强制锁速15
            if crossroad_alert:
                dynamic_speed = min(dynamic_speed, 15)

            # 识别到路口直接降到12
            if line_type == 3:
                dynamic_speed = min(dynamic_speed, 12)

            print(f"巡线中 - 偏移: {rotate_speed}, 旋转: {smooth_rotate}, 速度: {dynamic_speed}")
            u.mecanum_move_xyz(0, dynamic_speed, -smooth_rotate)

        # 丢线
        else:
            lost_line_count += 1
            start_frames = 0
            print(f"连续丢帧计数: {lost_line_count}/{LOST_LINE_THRESHOLD}")

            if lost_line_count < LOST_LINE_THRESHOLD:
                print("临时丢帧，原地停车等待识别")
                u.mecanum_stop()
            else:
                print("连续25帧未识别到线路，判定为丢线，原地小角度旋转找线")
                u.mecanum_move_xyz(0, 0, 10)
            time.sleep(0.2)

        time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════════
#  新增：从存储区返回中转区
# ═══════════════════════════════════════════════════════════════════
def return_to_transit_zone(u, target_store):
    """
    PID巡线返回中转区
    ─────────────────────────────────
    A 号存储区 → 中转区：经过 3 个十字路口
    B 号存储区 → 中转区：经过 4 个十字路口
    到达中转区后函数直接 return
    """
    LOCAL_BASE_SPEED = 22    # 基础前进速度（直线基准，转向时自动减速）
    LOCAL_MAX_SPEED = 27     # 前进速度上限
    LOCAL_MAX_ROTATE = 30    # 转向速度上限
    # 独立创建PID控制器，不依赖外部PID实例
    u.mecanum_turn_speed_times(2, 90, 173, 2)
    u.mecanum_translate_speed_times(90, 50, 8, 1)
    pid_return = u.create_pid_controller()
    pid_return.set_pid(0.12, 0.0015, 0.15)

    # 常量配置
    LOST_LINE_THRESHOLD = 8
    RECOVER_FRAMES = 15
    START_BUFFER_FRAMES = 30
    CROSSROAD_CD_FRAMES = 2

    # 状态变量初始化
    lost_line_count = 0
    recover_stable_count = 0
    last_rotate_speed = 0
    turn_count = 0
    crossroad_confirm = 0
    crossroad_cd = 0
    last_line_type = 0
    start_frames = 0

    finding_line = False
    find_direction = 1  # 1顺时针 -1逆时针
    find_step_count = 0

    if target_store == "A":
        print("从A号存储区返回中转区")
    else:
        print("从B号存储区返回中转区")

    while True:
        # 十字路口冷却计数递减，防止同一路口重复触发
        if crossroad_cd > 0:
            crossroad_cd -= 1
            u.mecanum_stop()
            time.sleep(0.1)

        try:
            line_info = u.get_single_track_total_info()
            offset, line_type, x, y = line_info
        except Exception as e:
            print(f"巡线识别出错: {e}，停车重试")
            u.mecanum_stop()
            time.sleep(0.2)
            continue

        # 线路类型变化时打印提示
        if line_type != last_line_type:
            if line_type == 3:
                print("检测到十字路口")
            elif line_type == 4:
                print("检测到T字路口")
            last_line_type = line_type

        # ── 正常巡线（直线/弯道/T字路口） ──
        if line_type in (1, 2, 4):
            lost_line_count = 0
            crossroad_confirm = 0
            finding_line = False
            find_step_count = 0
            start_frames += 1

            # 丢线恢复阶段限速
            if recover_stable_count < RECOVER_FRAMES:
                recover_stable_count += 1
                speed_limit = 15
            else:
                speed_limit = max_forward_speed

            # PID计算转向量 + 一阶平滑滤波
            rotate_speed = round(pid_return.update(offset))
            rotate_speed = max(min(rotate_speed, max_rotate_speed), -max_rotate_speed)
            smooth_rotate = int(last_rotate_speed * 0.4 + rotate_speed * 0.6)
            last_rotate_speed = smooth_rotate

            # 动态速度：弯道减速，转向量越大速度越低
            is_curve = line_type == 2
            if is_curve:
                dynamic_speed = int(base_forward_speed * 0.6 - min(abs(smooth_rotate) / 4, 5))
            else:
                dynamic_speed = int(base_forward_speed - min(abs(smooth_rotate) / 4, 6))

            # 起步缓冲：缓加速避免冲线
            if start_frames < START_BUFFER_FRAMES:
                ramp_speed = 12 + int((dynamic_speed - 12) * (start_frames / START_BUFFER_FRAMES))
                dynamic_speed = min(dynamic_speed, ramp_speed)

            # 速度上下限约束
            dynamic_speed = max(min(dynamic_speed, speed_limit), 5)
            u.mecanum_move_xyz(0, dynamic_speed, -smooth_rotate)

        # ── 十字路口处理 ──
        elif line_type == 3:
            # 冷却期内跳过，防止重复计数
            if crossroad_cd > 0:
                u.mecanum_move_xyz(0, 8, 0)
                time.sleep(0.05)
                continue

            crossroad_confirm += 1
            if crossroad_confirm < 1:
                u.mecanum_move_xyz(0, 8, 0)
                time.sleep(0.05)
                continue

            # 确认路口，停车执行动作
            u.mecanum_stop()
            time.sleep(0.3)
            turn_count += 1
            recover_stable_count = 0
            crossroad_confirm = 0
            crossroad_cd = CROSSROAD_CD_FRAMES
            start_frames = 0
            print(f"确认第 {turn_count} 个十字路口，执行对应动作")

            # ═══════════════════════════════════════
            #  A 区返回路径：共 3 个十字路口
            # ═══════════════════════════════════════
            if target_store == "A":
                if turn_count == 1:
                    print("第1个路口：直行后转向")
                    u.mecanum_move_speed_times(0, 35, 18, 1)
                    u.mecanum_turn_speed_times(2, 60, 90, 2)
                    time.sleep(0.5)

                elif turn_count == 2:
                    print("第2个路口：直行后转向")
                    u.mecanum_move_speed_times(0, 40, 19, 1)
                    u.mecanum_turn_speed_times(2, 60, 90, 2)
                    time.sleep(0.5)

                elif turn_count == 3:
                    print("第3个路口：到达中转区")
                    u.mecanum_move_speed_times(0, 35, 21, 1)
                    u.mecanum_stop()
                    time.sleep(0.5)
                    print("已返回中转区（A区路径）")
                    return

            # ═══════════════════════════════════════
            #  B 区返回路径：共 4 个十字路口
            # ═══════════════════════════════════════
            else:  # target_store == "B"
                if turn_count == 1:
                    print("第1个路口")
                    u.mecanum_move_speed_times(0, 35, 18, 1)
                    u.mecanum_turn_speed_times(2, 60, 95, 2)
                    time.sleep(0.5)
                    pass

                elif turn_count == 2:
                    print("第2个路口")
                    # u.mecanum_move_speed_times(0, 40, 57, 1)
                    # u.mecanum_turn_speed_times(2, 60, 91, 2)
                    # time.sleep(0.5)
                    u.mecanum_move_speed_times(0, 40, 19, 1)
                    time.sleep(0.5)
                    pass

                elif turn_count == 3:
                    print("第3个路口")
                    # u.mecanum_move_speed_times(0, 35, 21, 1)
                    # u.mecanum_stop()
                    # time.sleep(0.5)
                    # return
                    u.mecanum_move_speed_times(0, 40, 19, 1)
                    u.mecanum_turn_speed_times(2, 60, 90, 2)
                    time.sleep(0.5)
                    print(" 已返回中转区（B区路径）")
                    pass

                elif turn_count == 4:
                    print("第4个路口：到达中转区")
                    u.mecanum_move_speed_times(0, 35, 21, 1)
                    u.mecanum_stop()
                    time.sleep(0.5)
                    u.mecanum_stop()
                    print(" ---已返回中转区（B区路径）")
                    return

            time.sleep(0.4)
            u.mecanum_stop()
            time.sleep(0.1)

        # ── 丢线找回逻辑 ──
        else:
            recover_stable_count = 0
            start_frames = 0
            lost_line_count += 1

            if lost_line_count < LOST_LINE_THRESHOLD:
                # 临时丢线：低速前进，沿上次转向微调
                u.mecanum_stop()
                u.mecanum_move_xyz(0, 2, -int(last_rotate_speed * 0.2))
                print(f"临时丢线，低速维持找线，连续丢帧: {lost_line_count}")
                time.sleep(0.15)
            else:
                # 持续丢线：左右摆动主动找线
                if not finding_line:
                    finding_line = True
                    find_direction = 1
                    find_step_count = 0
                    print("持续丢线，启动主动旋转找线")

                u.mecanum_move_xyz(0, 0, 10 * find_direction)
                find_step_count += 1

                if find_step_count >= 10:
                    find_direction *= -1
                    find_step_count = 0

        time.sleep(0.1)

if __name__ == "__main__":
    from ugot import ugot

    try:
        trun_time = 0
        u = ugot.UGOT()
        u.initialize('10.233.104.242')
        u.load_models(["line_recognition"])
        u.set_track_recognition_line(0)
        print("开始平滑巡线...")
        pid_find_line(u)
    except KeyboardInterrupt:
        u.mecanum_stop()
        print("程序被用户中断")
    except Exception as e:
        u.mecanum_stop()
        print(f"发生错误: {e}")
    finally:
        print("巡线任务完成")
