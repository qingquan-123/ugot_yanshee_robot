# -*- coding: utf-8 -*-
"""
ugotzncc 启动包装器
在 ugot 环境下运行本脚本，内部自动调用 tf-od 环境执行 ugotzncc.py
模仿 ACTION.py 中 smart_warehouse_task 调用 action_test.py 的方式
"""
import subprocess
import os
import sys


def run_ugotzncc():
    """用 tf-od 环境运行 ugotzncc.py"""
    tf_od_python = r"C:\Users\16231\anaconda3\envs\tf-od\python.exe"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "ugotzncc.py")

    if not os.path.exists(script_path):
        print(f"错误：找不到 ugotzncc.py：{script_path}")
        return False

    if not os.path.exists(tf_od_python):
        print(f"错误：找不到 tf-od 环境 python：{tf_od_python}")
        return False

    cmd = [tf_od_python, script_path]

    print("=" * 50)
    print("  ugotzncc 启动器（ugot 环境 → tf-od 环境）")
    print(f"  调用环境: {tf_od_python}")
    print(f"  目标脚本: {script_path}")
    print("=" * 50)
    print(f"\n执行命令: {' '.join(cmd)}")
    print("ugotzncc 执行中...（tf-od 环境，含 TensorFlow）\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            capture_output=False,
            text=True
        )
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("  ugotzncc 执行完成")
            print("=" * 50)
            return True
        else:
            print(f"\n" + "=" * 50)
            print(f"  ugotzncc 执行失败，返回码: {result.returncode}")
            print("=" * 50)
            return False
    except Exception as e:
        print(f"\n调用 tf-od 环境失败: {e}")
        return False


if __name__ == "__main__":
    success = run_ugotzncc()
    sys.exit(0 if success else 1)
