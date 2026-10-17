# UGOT & Yanshee 机器人控制项目

面向睿抗（RoboCon）机器人竞赛的控制代码，覆盖 **UGOT** 与 **Yanshee** 两款机器人平台。

## 目录结构

```
gs/
├── src/
│   ├── ugot/          # UGOT 竞赛主程序
│   │   ├── main_gs.py         # 两轮任务调度主入口
│   │   ├── grasp_vision.py    # 抓取视觉（颜色识别、五号位）
│   │   ├── pid_controller.py  # PID 巡线与回中转区控制
│   │   └── apriltag_detect.py # AprilTag 识别
│   ├── ugotai/        # UGOT + AI（LLM 规划 / 语音指令）
│   │   ├── main.py             # 入口（含 SSL 兼容补丁）
│   │   ├── llm_planner.py      # 大模型任务规划
│   │   ├── action_scheduler.py # 动作队列调度
│   │   ├── smart_warehouse.py  # 智能仓储任务
│   │   ├── smart_companion.py  # 智能陪伴
│   │   ├── smart_geometry.py   # 几何识别任务
│   │   ├── smart_recon.py      # 侦察任务
│   │   ├── color.py            # 颜色识别
│   │   └── api_config.py       # 【不入库】百度智能云 API 配置
│   └── yanshee/       # Yanshee 机器人
│       ├── main.py             # 主控制程序
│       └── YanAPI.py           # Yanshee 通信 API 封装
├── models/            # 目标检测模型（TensorFlow Lite / frozen graph）
├── test_scripts/      # 分平台测试与调试脚本
└── .gitignore
```

## 环境依赖

- Python 3.8（UGOT 板载环境）
- 依赖安装包（如 TensorFlow aarch64 wheel）通过 `*.whl` 单独安装，不入库
- `certifi`、`aiohttp` 等（ugotai 需要）

## 使用说明

各平台入口：

```bash
# UGOT 竞赛主程序
python src/ugot/main_gs.py

# UGOT + AI（需先配置 API 密钥）
python src/ugotai/main.py

# Yanshee 机器人
python src/yanshee/main.py
```

## 安全说明

`src/ugotai/api_config.py` 含有真实的百度智能云密钥（AppBuilder Token、千帆 AK/SK），**已通过 .gitignore 排除，不会上传**。使用时请自行创建该文件，或在部署环境改用环境变量注入：

```python
# api_config.py 示例（本地自行创建，勿提交）
APPBUILDER_TOKEN = "bce-v3/你的token"
QIANFAN_ACCESS_KEY = "你的AK"
QIANFAN_SECRET_KEY = "你的SK"
```
