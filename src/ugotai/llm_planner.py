import os
import json
import re
import ast
from api_config import *
import qianfan

os.environ["QIANFAN_ACCESS_KEY"] = QIANFAN_ACCESS_KEY
os.environ["QIANFAN_SECRET_KEY"] = QIANFAN_SECRET_KEY
MODEL = "DeepSeek-V4-Flash"  # 已开通可用的轻量模型
chat_comp = qianfan.ChatCompletion(model=MODEL)


_api_fail_count = 0
API_FAIL_LIMIT = 2


def safe_parse_json(json_str):
    """安全解析JSON，兼容多种格式"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(json_str)
        except:
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'```json|```', '', json_str).strip()
            try:
                return json.loads(json_str)
            except:
                actions = re.findall(r'\[.*?\]', json_str, re.DOTALL)
                if actions:
                    try:
                        action_list = ast.literal_eval(actions[0])
                        return {"action": action_list, "response": "开始执行动作"}
                    except:
                        pass
    return {"action": [], "response": "无法解析指令"}


def fallback_parse(question=""):
    """本地规则兜底：千帆 API 不可用时使用，保证比赛现场不依赖网络。
    返回格式与 agent_plan_qianfan 一致：(动作列表, 语音回复)"""
    q = question or ""
    actions = []
    response = "好的，开始执行"

    # 智能侦察
    if any(k in q for k in ["识别", "侦察", "侦探"]):
        mapping = [("文字", "文字"), ("手势", "手势"), ("交通", "交通标志"),
                   ("标签", "标签"), ("色块", "色块")]
        found = [name for kw, name in mapping if kw in q]
        if found:
            actions.append(f"smart_detective_task({found})")
            response = "开始侦察模式，按顺序识别播报"
            return actions, response

    # 智能陪伴
    if any(k in q for k in ["陪伴", "聊天", "对话", "表情", "声音", "呼吸灯", "音效", "模仿"]):
        emotion = None
        for e in ["愤怒", "微笑", "高兴", "疑惑", "爱心", "骄傲", "睡眠", "苏醒"]:
            if e in q:
                emotion = e
                break
        sounds = [s for s in ["大象", "老虎", "救护车", "警车", "狗", "猫", "鸡"] if s in q]
        light = None
        for c in ["红色", "绿色", "蓝色"]:
            if c in q and "灯" in q:
                light = c
                break
        if emotion or sounds or light:
            args = ["2",
                    f"'{emotion}'" if emotion else "None",
                    str(sounds) if sounds else "[]",
                    f"'{light}'" if light else "None",
                    "3"]
            actions.append(f"smart_companion_task({', '.join(args)})")
            response = "开始陪伴模式"
            return actions, response

    # 智能仓储
    if any(k in q for k in ["搬运", "仓储", "抓取", "拿", "放置", "存储", "仓库", "色块"]):
        color = None
        for c in ["红色", "绿色", "蓝色"]:
            if c in q:
                color = c
                break
        if color:
            pos = 3
            import re
            pos_match = re.search(r'(\d+)\s*号', q)
            if pos_match:
                pos = int(pos_match.group(1))
            backward = 10
            back_match = re.search(r'后退\s*(\d+)', q)
            if back_match:
                backward = int(back_match.group(1))
            rotate = 0
            left_match = re.search(r'左转\s*(\d+)', q)
            right_match = re.search(r'右转\s*(\d+)', q)
            if left_match:
                rotate = int(left_match.group(1))
            elif right_match:
                rotate = -int(right_match.group(1))
            pre_rotate = 0
            circle_match = re.search(r'转\s*(\d+)\s*度.*圈|圈.*转\s*(\d+)\s*度', q)
            if circle_match:
                pre_rotate = int(circle_match.group(1) or circle_match.group(2) or 360)
            elif "圈" in q or "转圈" in q:
                pre_rotate = 360

            actions.append(f"smart_warehouse_task('{color}', {pos}, {backward}, {rotate}, {pre_rotate})")
            response = "开始仓储搬运任务"
            return actions, response

    # 智能解算
    if any(k in q for k in ["四边形", "三角形", "多边形", "解算", "几何", "边长", "算数"]):
        actions.append('geometry_task("polygon:4,30,cw;move:20,left;move:20,right;arm:3")')
        response = "开始执行几何运动"
        return actions, response


    if "前进" in q or "向前" in q:
        actions.append("move_forward(20)")
        response = "好的，向前前进20厘米"
    elif "后退" in q:
        actions.append("move_backward(15)")
        response = "好的，向后退15厘米"
    elif "左转" in q:
        actions.append("turn_left(90)")
        response = "好的，向左转90度"
    elif "右转" in q:
        actions.append("turn_right(90)")
        response = "好的，向右转90度"

    if not actions:
        response = "我暂时无法理解这个指令，请换个说法试试"

    return actions, response


def agent_plan_qianfan(question=""):
    global _api_fail_count
    if _api_fail_count >= API_FAIL_LIMIT:
        print(f" 千帆API已连续失败{_api_fail_count}次，本次直接使用本地规则兜底")
        return fallback_parse(question)

    PROMPT = f"""
    请将用户指令解析为机器人可执行的动作序列，严格返回JSON格式，禁止输出其他内容。
    返回格式：{{"action": ["动作1", "动作2"], "response": "对用户的语音回复"}}

    用户指令：{question}

    【严格规则】
    1. 字符串参数必须用英文单引号包围
    2. 只能使用下方列出的函数
    3. 数字参数不加引号
    4. 陪伴模式（chat_mode/smart_companion_task）如果用户未指定对话次数，默认 2 次对话
    5. 呼吸灯（set_breathing_light）如果用户未指定时长，默认 3 秒

    【可用函数清单】
    === 智能仓储 ===
    - smart_warehouse_task(前置转圈, 前进距离, '目标颜色', 目标位置, 后退距离, 旋转角度, 是否返回)
      【最高优先级】涉及物块搬运的任务，只输出这一个函数！绝对不要单独输出 geometry_task、move_forward、grab_color_block、place_block、move_backward、turn_left、rotate 等任何其他动作，所有动作都已包含在 smart_warehouse_task 内部。
      参数说明：
        - 前置转圈：原地转圈角度，正数右转，负数左转，0=不转圈。如360=先原地转一圈。未指定默认0
        - 前进距离：厘米，未指定默认0
        - 目标颜色：红色/蓝色/绿色
        - 目标位置：1-6号工位
        - 后退距离：cm，未指定默认10
        - 旋转角度：搬运完成后转向，正数左转，负数右转，0不旋转，未指定默认0
        - 是否返回：1=7步流程(放置后回到取货区)，0=6步流程(放置后不返回)。用户说"返回/回到/回起点"用1，否则默认0
      例1（前进20cm，搬运红色到3号，后退10cm，左转90度，不返回）：smart_warehouse_task(0, 20, '红色', 3, 10, 90, 0)
      例2（搬运蓝色到4号，后退30cm，右转90度，不返回）：smart_warehouse_task(0, 0, '蓝色', 4, 30, -90, 0)
      例3（先转一圈，再搬运绿色到5号，后退30cm，右转90度，不返回）：smart_warehouse_task(360, 0, '绿色', 5, 30, -90, 0)
      例4（搬运红色到3号，后退10cm，左转90度，需要返回取货区）：smart_warehouse_task(0, 20, '红色', 3, 10, 90, 1)
    - grab_color_block('颜色')
    - place_block(位置号)

    === 智能陪伴 ===
    - smart_companion_task(对话次数, '表情', '声音', '灯光颜色', 灯光时长, 是否重复回答)
    - chat_mode(次数, 是否重复回答)
    - chat_once('机器人主动问的问题')
    - show_emotion('表情名')
    - imitate_sound('音效名')
    - set_breathing_light('颜色', 持续秒数)

    【陪伴模式函数选择规则】
    - 用户说"聊聊天/陪伴模式/对话"等自由聊天 → 用 chat_mode(2, False)
    - 场景化单次询问（如"有人摔倒了询问情况""问问对方还好吗"）→ 用 chat_once('符合场景的问题')，如 chat_once('请问你还好吗？需要帮助吗？')
    - 完整陪伴任务（指定了对话次数+表情+音效+呼吸灯）→ 用 smart_companion_task
    - chat_once 只问一次，机器人主动提问，用户回答后机器人回应，回答文字会显示在屏幕上

    === 智能侦察 ===
    - smart_detective_task(['识别顺序列表'])
    - recognize_traffic_sign()
    - recognize_gesture()
    - recognize_words()
    - recognize_apriltag()
    - recognize_color()

    === 智能解算 ===
    - geometry_task("动作序列字符串")
    - move_arm_action(持续秒数)

    【geometry_task 动作格式】
    一个字符串参数，动作之间用分号(;)分隔，每个动作格式为 "类型:参数1,参数2,..."，支持以下类型：
    - polygon:<边数>,<边长>,<cw|ccw>[,<forward|backward>]  走多边形（边3-8，边长≤50cm，cw=顺时针，ccw=逆时针，forward=前进默认，backward=倒退）
    - move:<距离>,<left|right|forward|backward>              平移
    - turn:<角度>,<left|right>                                原地转向（角度≤360）
    - rotate:<角度>                                            旋转（用于360度转圈）
    - arm:<时长秒>                                              机械臂动作（时长≥3秒）

    【解算任务规则】
    - 走多边形用 polygon，不要用 move_forward + turn_left 拼
    - 平移用 move，转向用 turn，转圈用 rotate，机械臂用 arm
    - 动作按指令顺序排列，动作之间用分号(;)分隔
    - 整个动作序列放在一个字符串里，作为 geometry_task 的唯一参数
    - 例：顺时针四边形(50cm)→左转90°→倒退顺时针四边形(50cm)→左平移30cm→机械臂3秒
      → geometry_task("polygon:4,50,cw;turn:90,left;polygon:4,50,cw,backward;move:30,left;arm:3")

    === 基础动作 ===
    - move_forward(距离cm)
    - move_backward(距离cm)
    - turn_left(角度)
    - turn_right(角度)
    - move_left(距离cm)
    - move_right(距离cm)
    - display_text_on_screen('文字')
    - speak_and_display('文字')
    - clear_screen()

    【参数取值范围】
    - 颜色：'红色'/'绿色'/'蓝色'
    - 位置：1/2/3/4/5/6
    - 表情：'愤怒'/'微笑'/'高兴'/'疑惑'/'爱心'/'骄傲'/'睡眠'/'苏醒'
    - 音效：'大象'/'老虎'/'救护车'/'警车'/'狗'/'猫'/'鸡'等
    - 方向：'clockwise'顺时针 / 'counterclockwise'逆时针
    - 移动方向：'forward'前进 / 'backward'后退

    【比赛示例参考】
    1. 仓储："前进20cm搬运红色色块到3号位置，后退15cm回到取货区"
       返回：{{"action": ["smart_warehouse_task(20, '红色', 3, 15, 0)"], "response": "开始执行仓储搬运任务"}}

    2. 陪伴："开启2次对话，做愤怒表情，模仿大象和老虎声音，红色呼吸灯3秒"
       返回：{{"action": ["smart_companion_task(2, '愤怒', ['大象', '老虎'], '红色', 3, False)"], "response": "好的，进入陪伴模式"}}

    2b. 场景化陪伴："前面有人摔倒了，模仿救护车声音，询问情况，显示蓝色呼吸灯，微笑表情"
       返回：{{"action": ["imitate_sound('救护车')", "chat_once('请问你还好吗？需要帮助吗？')", "set_breathing_light('蓝色', 3)", "show_emotion('微笑')"], "response": "好的"}}

    3. 侦察："按文字、手势、交通标志、标签、色块的顺序识别播报"
       返回：{{"action": ["smart_detective_task(['文字', '手势', '交通标志', '标签', '色块'])"], "response": "开始侦探模式"}}

    4. 解算："顺时针走边长50cm的四边形，左转90度，倒退走边长50cm的四边形，左平移30cm，动动手3秒"
       返回：{{"action": ["geometry_task(\"polygon:4,50,cw;turn:90,left;polygon:4,50,cw,backward;move:30,left;arm:3\")"], "response": "好的"}}

    4b. 解算："顺时针走边长30cm的四边形，右平移30cm，逆时针走边长30cm的五边形，动动手，转个圈圈"
       返回：{{"action": ["geometry_task(\"polygon:4,30,cw;move:30,right;polygon:5,30,ccw;arm:3;rotate:360\")"], "response": "好的"}}
    """

    try:
        resp = chat_comp.do(
            messages=[{"role": "user", "content": PROMPT}],
            top_p=0.8,
            temperature=0.3,
            penalty_score=1.0,
            retry_count=0,
            request_timeout=15
        )
        result = resp["result"]
        print('【大模型原始输出】\n', result)

        json_match = re.search(r'\{[^{}]*(\{[^{}]*\})?[^{}]*\}', result)
        if json_match:
            action_plan_json = safe_parse_json(json_match.group())
        else:
            action_plan_json = {"action": [], "response": result}

        print('【解析后动作计划】\n', action_plan_json)
        _api_fail_count = 0
        return action_plan_json.get('action', []), action_plan_json.get('response', '好的，执行指令')

    except Exception as e:
        _api_fail_count += 1
        print(f" 大模型调用失败: {e}")
        print(f" 千帆API不可用（连续失败{_api_fail_count}/{API_FAIL_LIMIT}次），切换到本地规则兜底解析")
        return fallback_parse(question)
