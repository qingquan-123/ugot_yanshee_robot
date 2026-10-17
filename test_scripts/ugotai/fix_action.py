file_path = r"C:\Users\16231\Desktop\ros\gS\ugotai\ACTION.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old = '''# ---- 1. 智能仓储 → zncc.py ----
def smart_warehouse_task(forward_dist, target_color, target_pos, backward_dist, rotate_angle=0):
    """
    智能仓储完整任务（调用 zncc.py）
    参数：
    - forward_dist: 前进距离（cm），透传给脚本
    - target_color: 目标颜色（红色/绿色/蓝色）
    - target_pos: 目标存储区编号（1-6）
    - backward_dist: 后退距离（cm），透传给脚本
    - rotate_angle: 旋转角度（度），>0 表示需要回到取货区（7步流程），=0 表示不需要（6步流程）
    """
    print("\\n" + "=" * 50)
    print(" 智能仓储任务启动（调用 zncc.py）")
    print(f" 前进{forward_dist}cm | 抓取{target_color} | 放置{target_pos}号 | 后退{backward_dist}cm | 旋转{rotate_angle}度")
    print("=" * 50)

    need_return = 1 if rotate_angle > 0 else 0
    print(f"执行模式: {'7步流程（需要回到取货区）' if need_return else '6步流程（不需要返回）'}")

    # zncc.py 参数：target_color target_store_id need_return
    return _run_script("zncc.py", [target_color, target_pos, need_return])'''

new = '''# ---- 1. 智能仓储 → zncc.py ----
def smart_warehouse_task(forward_dist, target_color, target_pos, backward_dist, rotate_angle=0):
    """
    智能仓储完整任务（调用 zncc.py）
    参数：
    - forward_dist: 前进距离（cm）
    - target_color: 目标颜色（红色/绿色/蓝色）
    - target_pos: 目标存储区编号（1-6）
    - backward_dist: 后退距离（cm）
    - rotate_angle: 转向角度（度），正=左转，负=右转
    """
    print("\\n" + "=" * 50)
    print(" 智能仓储任务启动（调用 zncc.py）")
    print(f" 前进{forward_dist}cm | 抓取{target_color} | 放置{target_pos}号 | 后退{backward_dist}cm | 转向{rotate_angle}度")
    print("=" * 50)

    # zncc.py smart_warehouse 模式参数：smart_warehouse 前进cm 颜色 放置号 后退cm 转向角
    return _run_script("zncc.py", ["smart_warehouse", str(forward_dist), target_color, str(target_pos), str(backward_dist), str(rotate_angle)])'''

content = content.replace(old, new)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("OK")
