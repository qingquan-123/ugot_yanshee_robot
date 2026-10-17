# coding=UTF-8
import time
import json
import socket
import YanAPI

BROADCAST_PORT = 9527

def tts(content):
    try:
        YanAPI.sync_do_tts(content, True)
    except Exception as e:
        print("tts error:", str(e))

def receive_once():
    print("\n等待UGOT广播任务... 端口：" + str(BROADCAST_PORT))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", BROADCAST_PORT))
    try:
        raw_data, addr = sock.recvfrom(1024)
        payload = json.loads(raw_data.decode("utf-8"))
        print("收到数据：", payload)
        return payload
    except Exception as e:
        print("接收异常：", e)
        return None
    finally:
        sock.close()

def main():
    print("==== Yanshee 单次接收播报程序启动 ====")
    tts("准备就绪，等待任务")

    task = receive_once()
    if task is not None:
        first_task = task["first_task"]
        second_task = task["second_task"]

        first_color = first_task["color"]
        first_area = first_task["storage_area"].upper()
        second_color = second_task["color"]
        second_area = second_task["storage_area"].upper()

        msg1 = "我要去" + first_area + "号存储区，分拣" + first_color + "色块"
        print(msg1)
        tts(msg1)
        time.sleep(1)

        msg2 = "第二趟去" + second_area + "号存储区，分拣" + second_color + "色块"
        print(msg2)
        tts(msg2)

        print("播报完成，程序结束")
    else:
        print("没有收到有效数据，程序结束")

if __name__ == "__main__":
    main()