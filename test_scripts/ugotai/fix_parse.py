import re

file_path = r"C:\Users\16231\Desktop\ros\gS\ugotai\znjs.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old = '''def parse_and_exec(got, action_str):
    """
    解析并执行单个动作
    格式：类型:参数1,参数2,...
    """
    action_str = action_str.strip()
    if not action_str:
        return False

    if ":" not in action_str:
        print(f"  [错误] 动作格式错误（缺少冒号）: {action_str}")
        print(f"  正确格式: 类型:参数1,参数2,...  例如: polygon:4,50,cw")
        return False

    action_type, params_str = action_str.split(":", 1)
    action_type = action_type.strip().lower()
    params = [p.strip() for p in params_str.split(",") if p.strip()]'''

new = '''def parse_and_exec(got, action_str):
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
    params = [p.strip() for p in params_str.split(",") if p.strip()]'''

if old in content:
    content = content.replace(old, new)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("OK")
else:
    print("NOT FOUND")
