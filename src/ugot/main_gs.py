# -*- coding: utf-8 -*-
import time
import threading
import socket
import json
from ugot import ugot
from pid_controller import pid_find_line, return_to_transit_zone
from grasp_vision import get_bbox, five_store
from apriltag_detect import wttag


class UGOTCompetition:
    def __init__(self, ip_address='10.233.104.242'):
        self.ip_address = ip_address
        self.got = ugot.UGOT()

        # ---- 两轮任务状态 ----
        self.task_list = []  # [(color, store), (color, store)]，按执行顺序
        self.current_round = 0  # 当前执行到第几轮（1 or 2）1
        self.target_color = None  # 当前轮目标颜色（兼容原有变量名）
        self.target_store = None  # 当前轮目标存储区

        # ---- UDP广播配置（替换原来机器人内置广播） ----
        self.broadcast_port = 9527
        self.broadcast_addr = '255.255.255.255'

        self.search_time = 0
        self.camera_opened = False
        self.camera_initialized = False
        self.current_zone_id = None
        self.arm_ready = threading.Event()

        self.got.initialize(self.ip_address)
        print("机器人连接成功")

        hw_thread = threading.Thread(target=self._light_init_thread, daemon=True)
        hw_thread.start()

        self.setup_robot()

        print("正在加载视觉模型...")
        self._load_vision_model()

        self.arm_ready.wait()
        print("全部就绪，开始执行任务")

    def broadcast_task_info(self):
        if len(self.task_list) < 2:
            print("任务数量不足，跳过广播")
            return False

        task_data = {
            "first_task": {
                "color": self.task_list[0][0],
                "storage_area": self.task_list[0][1]
            },
            "second_task": {
                "color": self.task_list[1][0],
                "storage_area": self.task_list[1][1]
            }
        }
        json_str = json.dumps(task_data, ensure_ascii=False).encode('utf-8')

        try:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            udp_socket.sendto(json_str, (self.broadcast_addr, self.broadcast_port))
            udp_socket.close()
            print(f"任务数据已广播：{task_data}")
            return True
        except Exception as e:
            print(f"广播发送失败：{e}")
            return False

    def _light_init_thread(self):
        """后台轻量初始化"""
        try:
            print("初始化机械臂...")
            self.got.mechanical_single_joint_control(1, 0, 250)
            time.sleep(0.6)
            self.got.mechanical_single_joint_control(2, 60, 100)
            time.sleep(0.6)
            self.got.mechanical_single_joint_control(3, 60, 100)
            time.sleep(0.6)
            self.got.mechanical_clamp_release()
            time.sleep(0.6)
            self.got.set_volume(100)
            self.got.open_camera()
            time.sleep(0.2)
            self.camera_opened = True
        except Exception as e:
            print(f"硬件初始化异常：{e}")
        finally:
            self.arm_ready.set()

    def _load_vision_model(self):
        """单独加载视觉大模型"""
        try:
            if not self.camera_opened:
                self.got.open_camera()
                time.sleep(0.2)
            self.got.load_models(['line_recognition'])
            time.sleep(0.2)
            self.got.set_track_recognition_line(0)
            self.camera_initialized = True
        except Exception as e:
            print(f"视觉模型加载异常：{e}")
            self.camera_initialized = False
        print("视觉系统就绪")
        time.sleep(1)


    def extract_task_info(self, task_message):
        color_set = {"红色", "蓝色", "绿色", "黄色", "白色", "黑色"}
        area_letters = {"A", "B"}

        extracted_colors = []
        for color in color_set:
            idx = 0
            while True:
                pos = task_message.find(color, idx)
                if pos == -1:
                    break
                extracted_colors.append((color, pos))
                idx = pos + len(color)
        extracted_colors.sort(key=lambda x: x[1])
        colors_ordered = [c for c, _ in extracted_colors]

        extracted_stores = []

        import re
        pattern = r'([A-Ba-b0-9])\s*号\s*(?:存储区|库区|仓库|存放区)'
        for m in re.finditer(pattern, task_message):
            char = m.group(1).upper()
            extracted_stores.append((char, m.start()))

        if not extracted_stores:
            has_store_keyword = any(k in task_message for k in
                                    ["存储", "放置", "区", "库区", "仓库", "存放"])
            if has_store_keyword:
                for char in area_letters:
                    if char in task_message:
                        extracted_stores.append((char, task_message.find(char)))

        extracted_stores.sort(key=lambda x: x[1])
        stores_ordered = [s for s, _ in extracted_stores]

        task_list = []
        pair_count = min(len(colors_ordered), len(stores_ordered))
        for i in range(pair_count):
            task_list.append((colors_ordered[i], stores_ordered[i]))

        return task_list

    def setup_robot(self):
        """语音识别：循环监听，识别两轮任务后统一播报 + UDP广播"""
        ASR_TIMEOUT = 15
        ASR_INTERVAL = 0.3

        print("\n语音识别已启动，请立刻说出指令")

        while True:
            task_msg = ""
            start_time = time.time()
            while time.time() - start_time < ASR_TIMEOUT:
                try:
                    result = self.got.start_audio_asr()
                    if result and result.strip():
                        task_msg = result.strip()
                        break
                except Exception as e:
                    print(f"语音识别异常: {e}")
                time.sleep(ASR_INTERVAL)

            print(f"识别内容：{task_msg}")

            self.task_list = self.extract_task_info(task_msg)

            if len(self.task_list) >= 2:
                (c1, s1), (c2, s2) = self.task_list[0], self.task_list[1]
                res = f"好的，第一次搬运{c1}放{s1}号，第二次搬运{c2}放{s2}号"
                print(res)
                self.got.play_audio_tts(res, 0, True)
                self.broadcast_task_info()
                break

            print("指令识别失败，请重新说出指令")
            self.got.play_audio_tts("请重试", 0, True)

    def find_line(self):
        if not self.camera_initialized:
            print("错误：摄像头未就绪，禁止巡线")
            return

        print("开始巡线...")
        try:
            pid_find_line(self.got)
        except Exception as e:
            print(f"巡线异常：{e}")
            return

        self.got.mechanical_clamp_release()
        time.sleep(0.15)
        print("巡线完成")

    def grasp_target(self, round_num=1):
        """
        抓取指定轮次的目标色块
        :param round_num: 第几次任务（1 or 2）
        """
        if round_num - 1 >= len(self.task_list):
            print(f"错误：第{round_num}轮任务不存在")
            return 0, False, None, {}, None

        self.current_round = round_num
        self.target_color, self.target_store = self.task_list[round_num - 1]
        print(f"\n===== 第{round_num}轮：抓取 {self.target_color} 色块 =====")

        try:
            _, self.search_time, flag, flag_diction, self.current_zone_id = \
                get_bbox(self.got, self.target_color)
            task_info = [self.target_color, self.target_store]
            return self.search_time, flag, task_info, flag_diction, self.current_zone_id
        except Exception as e:
            print(f"第{round_num}轮抓取流程异常：{e}")
            return 0, False, None, {}, None

    def store_to_platform(self, search_time, flag, task_info, flag_diction, zone_id=None):
        """
        将当前轮色块放置到对应存储区
        """
        print(f"第{self.current_round}轮：放置到 {self.target_store} 存储区")
        if self.target_store not in ("A", "B"):
            print("错误：未识别到有效目标仓库（A/B），终止入库流程")
            self.got.mecanum_stop()
            return
        five_store(self.got, self.target_store, self.current_round,
                   search_time, flag, task_info, flag_diction, zone_id)

    def cleanup(self):
        try:
            self.got.mecanum_stop()
            if self.camera_opened:
                self.got.close_camera()
        except Exception:
            pass


def main():
    print("=== 语音识别 ===")
    comp = None
    try:
        comp = UGOTCompetition('10.233.104.242')
        from grasp_vision import reset_line_vision

        comp.find_line()

        wttag()
        reset_line_vision(comp.got)

        res1 = comp.grasp_target(round_num=1)
        if res1[1]:
            comp.store_to_platform(*res1)

        reset_line_vision(comp.got)
        return_to_transit_zone(comp.got, comp.task_list[0][1])

        wttag()
        reset_line_vision(comp.got)


        res2 = comp.grasp_target(round_num=2)
        if res2[1]:
            comp.store_to_platform(*res2)

        comp.got.play_audio_tts("任务一已完成", 0, False)

        print("全部任务完成，程序退出")
    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        if comp:
            comp.cleanup()


if __name__ == "__main__":
    main()
