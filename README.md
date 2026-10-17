# UGOT & Yanshee 机器人控制项目

面向睿抗（RoboCon）机器人竞赛的机器人智能控制系统，覆盖 **UGOT** 与 **Yanshee** 双平台，打通视觉感知 → 运动控制 → 大模型任务规划完整链路。

## 项目亮点

- **双平台机器人控制**：一套代码体系同时驱动 UGOT（竞赛轮式机器人）与 Yanshee（人形机器人），覆盖抓取搬运、智能仓储、人机协同等竞赛任务
- **多层视觉感知**：OpenCV HSV 颜色识别、AprilTag 高精度定位、TensorFlow Lite（SSD MobileNet）目标检测三套视觉方案按场景切换
- **鲁棒运动控制**：多级 PID 巡线（弯道降速、十字路口策略、丢线自恢复）、距离维持 PID、五阶段对准状态机，参数均经实车调优
- **大模型驱动任务规划**：接入百度千帆 LLM 将自然语言/语音指令解析为机器人动作序列，内置熔断降级与本地规则兜底，保证竞赛现场稳定性
- **多机器人协同**：UDP 广播任务分发，UGOT 下发任务、Yanshee 接收执行并语音反馈，构成完整闭环
- **工程化细节**：SSL 兼容补丁、子进程环境隔离（conda）、摄像头复位时序保护、多线程任务队列调度

## 技术栈

| 领域 | 技术 |
|---|---|
| 语言 / 环境 | Python 3.8、Conda 多环境隔离（板载 / tf-od） |
| 视觉感知 | OpenCV（HSV 阈值、形态学）、AprilTag、TensorFlow Lite（SSD MobileNet v2 FPN） |
| 运动控制 | 多级 PID（巡线 / 距离 / 角度）、有限状态机 |
| AI 能力 | 百度千帆 LLM（DeepSeek-V4-Flash）、语音 ASR / TTS |
| 通信架构 | UDP 广播、Socket、多线程任务调度 |

## 核心功能

- **颜色识别抓取**：HSV 精确识别红/绿/蓝色块，面积分级渐进靠近，机械臂抓取入库（6→1 区搜索）
- **智能巡线**：PID 巡线 + 路口警戒 + 十字路口多轮转向 + 丢线摆动找回 + 自动返回中转区
- **AprilTag 精准定位**：五阶段对准状态机（搜索→粗→中→精→稳定确认），标定偏移补偿，精度达像素级
- **大模型任务规划**：LLM 解析任务文本为动作序列，连续失败自动熔断切换本地规则
- **多机器人协同**：UGOT 广播任务 JSON，Yanshee 人形执行（动作播放 + 视觉对准 + 语音播报）
- **目标检测识别**：TFLite 识别区域标签，按标签面积 5 档分级决定放置策略

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
├── yanshee_move/      # Yanshee 动作库（.hts 动作文件：行走/转弯/抓取/蹲起等 47 个）
├── test_scripts/      # 分平台测试与调试脚本
└── .gitignore
```

## 环境依赖

**第三方库**（已收录于 `requirements.txt`）：

| 库 | 用途 | 使用场景 |
|---|---|---|
| opencv-python | HSV 颜色识别、图像预处理 | ugot / ugotai / yanshee |
| numpy | 图像数组、数值计算 | 全部 |
| qianfan | 百度千帆 LLM（DeepSeek-V4-Flash） | ugotai |
| certifi | SSL 证书校验（解决板载 SSL 报错） | ugotai |
| requests | HTTP 请求 | ugotai |
| Pillow | 图像处理 | ugotai |
| nest_asyncio | 事件循环兼容 | ugotai |
| tensorflow / tflite-runtime | 目标检测模型推理（SSD MobileNet） | ugotai / models |

**分平台安装说明**：

| 平台 | Python | 依赖差异 | 安装方式 |
|---|---|---|---|
| UGOT 板载 | 3.8（aarch64） | TensorFlow **2.10.1**（ARM wheel 单独安装，不入库） | `pip install -r requirements.txt`，再 `pip install tensorflow_aarch64-2.10.1-....whl` |
| PC 开发 | 3.8+（x86_64） | 用 `tflite-runtime` 或 TensorFlow CPU 版替代 | `pip install -r requirements.txt` 后按需安装推理库 |

**其他说明**：
- `ugot` 机器人 SDK 由厂商提供，按官方文档安装（本仓库不包含）
- 语音识别/合成依赖 UGOT 板载能力或云端 API，无需额外安装
- 完整环境一键安装：

```bash
pip install -r requirements.txt
```

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


## 代码要点注释

> 有价值的算法/参数/协议注释已从代码移入本表；代码内仅保留编码声明、必要 docstring 与极少量内联提示。数值均为各脚本当前实际使用值。

### ugot（UGOT 竞赛主程序）

- **UGOT 板载 IP**：`10.233.104.242`（main_gs.py / pid_controller.py / apriltag_detect.py 共用）。
- **PID 巡线系数**：主巡线 `set_pid(0.12, 0.001, 0.08)`；入库/返回巡线 `set_pid(0.12, 0.0015, 0.15)`；速度一阶平滑滤波权重：上轮 0.4 / 本轮 0.6。
- **巡线车速**：max_rotate_speed=20、max_forward_speed=30、base_forward_speed=20；弯道降速 55%、第 2 次转弯后 ×0.85、起步前 30 帧限速 15、偏移 >80 限 18 / >50 限 22、路口警戒锁 15、line_type=3 降到 12。
- **路口警戒机制**：弯道（line_type=2）或单帧偏移 >90px 触发，持续 35 帧（ALERT_TOTAL_FRAMES）后解除；ALERT_FORCE_TRIGGER=25 已定义但当前未使用。
- **十字路口策略**：trun_time 最多 3 次——第 1 次直行 80 + 转 89°、第 2 次直行 50 + 转 91°、第 3 次直行后结束；路口冷却 2 帧防重复触发；丢线 30 帧原地旋转找线。
- **转向后矫正**：quick_align_after_turn 最多 4 轮——偏移 >40 横移 7、>20 横移 5、≤20 用 PID 旋转微调（限 ±8）；enable_forward_slide=False 时跳过结尾前滑行（近路口场景）。
- **返回中转区参数**：LOCAL_BASE_SPEED=22 / LOCAL_MAX_SPEED=27 / LOCAL_MAX_ROTATE=30；A 区经 3 个、B 区经 4 个十字路口；独立 PID 实例；丢线 8 帧后以速度 10 摆动找线、10 步换向。
- **线路类型枚举**：`get_single_track_total_info()` 返回 `(offset, line_type, x, y)`——0=丢线、1=直线、2=弯道、3=十字路口、4=T 字路口。
- **AprilTag 数据格式**（apriltag_detect.py）：`get_qrcode_apriltag_total_info()` 返回 13 元组——[1]=id、[2,3]=中心 xy、[6]=面积、[7/8/9]=5/7/10cm 距离（米）、[10/11/12]=xyz 姿态角（rad）；TAG_SIZE='10' 取索引 9。
- **AprilTag 对准**（apriltag_detect.py）：阈值 粗 60px→中 30px→精 17px；标定偏移 CALIBRATION_OFFSET=-16px；连续 2 帧（STABLE_FRAMES）居中即完成，最多 3 次稳定确认（MAX_STABLE_ATTEMPTS）。速度：搜索自转 8、粗对准横移 15、中速横移 8、精调最低 3；FINE_KP=0.25，限幅 3~8。状态机五阶段：搜索→粗对准→中速对准→精对准→稳定确认，丢标签逐级回退（精对准丢标签回中速、中速丢回粗对准）。
- **HSV 颜色阈值**（grasp_vision.py）：红 [0,167,132]-[10,255,255] 与 [161,167,132]-[180,255,255]；绿 [33,113,62]-[76,255,255]；蓝 [100,167,95]-[143,255,255]。
- **色块检测参数**（grasp_vision.py）：detect_color_objects 最小面积 500px、5×5 核形态学开闭；get_bbox 严格过滤：左右各 20% 边缘无效（INVALID_EDGE_RATIO=0.2）、最小面积 STRICT_MIN_AREA=2000（3 号区放宽 500）。
- **前进到位判定**（go_forward）：底部 Y≥0.9×高且面积≥9% 全屏，或底部 1/4 区域颜色占比≥0.39，任一满足即停；接近后（Y≥0.7 或面积≥阈值×0.6）降速 3。go_forward_zone3（3 号区）阈值更低：Y≥0.8 / 面积≥8.5% / 底部占比≥0.2。
- **横移对齐参数**：ALIGN_THRESHOLD=±18px、ALIGN_SPEED=3、ALIGN_DURATION=1s。
- **库区搜索顺序**：get_bbox 从 6 号区开始→右移搜索 5→4→3→2→1；3 号区搜索超时 3 次转 2 号区；1 号区右移 3 次（速度 5、时长 2）未中再左移 5 次。
- **巡线视觉复位**（reset_line_vision）：先停稳 0.3s→关/开摄像头→重载 line_recognition 模型→set_track_recognition_line(0)→预热 5 帧，需连续 3 帧读到有效线类型（1~4）。
- **UDP 任务广播**（main_gs.py 发送端）：broadcast_port=9527、broadcast_addr=255.255.255.255；JSON 格式 `{"first_task":{"color","storage_area"},"second_task":{...}}`，UTF-8 编码；任务列表 ≥2 条才广播。
- **语音识别参数**（main_gs.py）：ASR_TIMEOUT=15s、ASR_INTERVAL=0.3s 轮询 `start_audio_asr()`；识别失败 TTS"请重试"后循环重听。
- **任务解析规则**（main_gs.py）：颜色关键词集合 {红/蓝/绿/黄/白/黑}；库区正则 `([A-Ba-b0-9])\s*号\s*(存储区|库区|仓库|存放区)`；按文本出现顺序两两配对成 (color, store)。
- **主流程时序**（main_gs.py）：初始化→巡线 pid_find_line→wttag 对准→reset_line_vision→第 1 轮抓取入库→reset→return_to_transit_zone→wttag→reset→第 2 轮抓取入库→TTS 播报结束。

### ugotai（UGOT + AI）

- **机器人连接 IP**：`10.233.104.242`（main.py 及各独立脚本共用）。
- **子进程运行环境**：四大场景脚本统一用 tf-od conda 环境的 python.exe 经 subprocess 调用（板载环境无 tensorflow）。
- **SSL 兼容补丁**（main.py / smart_companion.py）：必须在 import qianfan（aiohttp）之前生效，否则读 Windows 证书存储会触发 ASN1 NOT_ENOUGH_DATA 错误。
- **千帆 LLM 调用**（llm_planner.py / action_scheduler.py）：模型 DeepSeek-V4-Flash；规划请求 top_p=0.8 / temp=0.3 / 超时 15s（action_scheduler 中 temp=0.7 / 超时 8s）；retry_count=0（配额类错误重试无意义）；千帆不支持 system 角色作首条消息（系统提示拼进 user）；对话历史只保留最近 2 轮（4 条）。
- **熔断兜底**（llm_planner.py）：连续失败 API_FAIL_LIMIT=2 次，本场会话转本地规则；调用成功一次计数清零。
- **本地兜底规则**（llm_planner.py）：关键词匹配五场景；仓储用正则提取位置号（\d+号）/ 后退 Ncm / 左转正右转负 / 前置转圈默认 360。
- **AprilTag 对准参数**（action_scheduler.py）：帧宽 640 / 中心 320；阈值 粗 50 / 中 25 / 精 10px；转速 搜索 6 / 粗 12 / 中 8，精调 KP=0.35×偏差、最低速 5；连续 3 帧 ≤10px 判定稳定；TAG_SIZE='10'；偏差 ≤10px 可捷径直进稳定确认。
- **基础移动速度**（action_scheduler.py）：前进/后退 20、转向 130、平移 15。
- **HSV 颜色阈值**（color.py）：同 grasp_vision.py（红 H 0-10∪161-180、S≥167、V≥132；绿 H 33-76、S≥113、V≥62；蓝 H 100-143、S≥167、V≥95，OpenCV HSV）。
- **色块抓取参数**（color.py）：最小面积 500px；横向对准阈值 35px；搜索 5 次未中切左区；前进到位=底部 1/4 区域目标色占比≥0.5。
- **解算动作格式**（smart_geometry.py）：`polygon:<边3-8>,<边长≤50cm>,<cw|ccw>[,<forward|backward>]`；`move:<距离>,<方向>`；`turn:<角度≤360>,<left/right>`；`rotate:<角度>`；`arm:<时长≥3s>`；动作分号分隔，兼容括号写法自动转换。运动速度：走边 20 / 转向 60 / 平移 15 / 旋转 80。多边形原理：外角和 360°，每走一边转 360/边数 度，末边不转。
- **侦察参数**（smart_recon.py）：CHECK_INTERVAL=0.2s 轮询；MAX_LOOP_SEC=120s；STABLE_FRAMES=2（色块需 3 帧才确认）；MIN_AREA=500。交通标志白名单：左转/右转/红灯/绿灯/黄灯；文字清洗后截断 10 字；HSV 阈值同 color.py。
- **智能陪伴运行参数**（smart_companion.py）：ROBOT_IP=10.233.104.242；命令行参数 chat_times / emotion / sound / light_color / light_duration / repeat；超时：LLM 8s、天气接口 5s、ASR 15s；表情保持 2.5s；默认城市杭州；屏幕文字截断 30 字、回答截断 50 字；天气接口 `wttr.in/{city}?format=%C+%t+%h+%w`（requests 未装则降级）。
- **仓储流程**（smart_warehouse.py）：7 步 = Tag 对准→机械臂复位→视觉复位→6 号区（6→1）搜索抓取→回中转→再 Tag 对准→搜索放置→放置后回中转；6 步流程去掉末步；抓取区==放置区自动改放 6 号。
- **仓储 AprilTag 对准**（smart_warehouse.py）：帧宽 640 / 中心 320；标定偏移 CALIBRATION_OFFSET=16px（正对时原始偏移 +16 需减去）；阈值 粗 60 / 中 30 / 精 25；转速 搜索 8 / 粗 4 / 中 8，精调 KP=0.25、最低速 5；搜索需连续 2 帧同 ID 且偏差差 ≤5px；稳定 2 帧确认。
- **放置对准与 TFLite**（smart_warehouse.py）：ALIGN_THRESHOLD=18px、ALIGN_SPEED=3 / 时长 1s；LABEL_AREA_THRESHOLD=13000；边缘无效区 0.2；STRICT_MIN_AREA=2000（3 区放宽 500）。模型 `models/ssd_mobilenet_v2_fpn_320.tflite` + label_map.pbtxt；输入 320×320 BGR→RGB；输出 boxes/scores/classes；min_conf=0.8；num_threads=4。
- **放置前进到位（标签面积 5 档）**（smart_warehouse.py）：<9000 前进 5cm；9000-11000 再进 4cm 放；11000-15000 进 3cm；15000-17000 进 2cm；17000-21000 进 1cm；>21000 后退 6cm；3 号标签专用区间 8000-11000。
- **释放物块（target_id=6）**（smart_warehouse.py）：后退 3cm→关节 3 抬到 25°→前进 2cm→松爪，防卡板。
- **色块前进到位**（smart_warehouse.py）：底部 Y≥0.9 且面积占比≥9%，或底部 1/4 区域色占比≥0.39 即停；3 号区阈值更低（0.8 / 8.5% / 0.2）。

### yanshee

- **YanAPI 初始化**：独立脚本开头须 `YanAPI.yan_api_init("127.0.0.1")` 连本机 Yanshee 服务（main.py 运行环境已初始化，不显式调用）。
- **YanAPI 动作约定**：`sync_play_motion("<动作名>")` 同步播放动作；`sync_do_tts(文本, True)` 语音播报。
- **UDP 任务广播接收**（main.py）：绑定 0.0.0.0:9527（BROADCAST_PORT=9527），收 UGOT 广播的 UTF-8 JSON，字段 first_task/second_task，每趟含 color（红/绿/蓝）、storage_area（A/B，代码转大写），收一帧即关 socket（发送方见 ugot「UDP 任务广播」）。
- **AprilTag 识别参数**（yanshee）：识别 id 0~5、物理尺寸 size=0.06m；位姿取 data.AprilTagStatus[0]，position-x 为左右偏移（米），orientation-x/y/z/w 四元数转 pitch 做角度对准；用完须 stop_aprilTag_recognition()。
- **对准阈值**（translate / tat）：translate() 左右阈值 align_threshold=0.02m；tat() 状态机 pos_threshold=0.013m；角度粗调 ≤10° 最多 2 次、精调 ≤6° 最多 3 次。
- **tat 六阶段状态机**（turn_and_translate.py）：阶段1 初始转向粗调（≤10°，≤2 次）→阶段2 初始平移（0.013m）→阶段3 大步幅搜目标 id→阶段4 一次平移对准→阶段5 角度精调（≤6°，≤3 次）→阶段6 二次平移对准。
- **小/大步幅与转向动作**：小步幅对准 te_left_1_S6 / te_right_S4；大步幅搜索 trte_L_J_big_02 / trte_R_J_big_04t；原地小角度转 turn_L_S2 / turn_r_S1。
- **色块 HSV 检测**（yanshee）：红 H 0-10 与 156-180、绿 H 33-76、蓝 H 90-143（S≥60，V≥40）；最小轮廓面积 min_area=200；先 5×5 高斯模糊再开闭运算。
- **人脸样本**：样本图 /home/pi/Desktop/sample/{Michael,Nancy,Simon}.png；upload_vision_photo_sample 上传后 set_vision_tag 绑名；识别满 3 人退出并 TTS 播报。
- **任务二动作序列**（single_test.py）：有效序列 A_start_all_908A → come_up_09712 / restoration_h_04 循环 → come_down_0908(A)；跨栏 hurdles_807 等为已注释废弃备选。
- **演示流程**（test.py）：起姿抓取(clamp) → turn_r_90_806A 右转 → tat(id=3) + A_walk_J_02_b×3 投放(let_go) → 蓝色区(B) 抓取 → turn_r_90_806A → tat(id=1) 投放。
- **偏航角对准**（turn_lrb.py）：对准字段 rotation-yaw（字段名待确认，当前调试态会打印完整 tag 数据）；yaw_threshold=5°；主循环 sleep 0.5s 便于观察。

### test_scripts

- **UDP 广播发送示例**（broadcast_task_old.py）：目标 10.233.104.217:9527；JSON 报文含 first_task/second_task 两个 {color, storage_area}；需至少 2 组（颜色+存储区配对）；连发 20 次、间隔 0.1s。
- **摄像头取帧模式**（color_test.py）：got.read_camera_data() 返回 JPEG 字节，需 np.frombuffer + cv2.imdecode(IMREAD_COLOR) 解码。
- **HSV 颜色阈值（UGOT 系）**：同 grasp_vision.py / color.py；最小识别面积 500px，5×5 形态学开/闭去噪。
- **距离维持 PID（对齐版 distance_align_old.py）**：按色块面积测距，目标 30000px、死区 25000~35000、Kp=0.0008、速度钳制 MIN=MAX=5、连续 1 帧在区间内即对齐成功；dir_code 0=前进 1=后退。
- **距离维持 PID（保持版 distance_keep_test.py）**：目标 45000px、死区 38000~50000、Kp=0.0008、速度钳制 MIN=3 / MAX=8。
- **麦轮运动 API 约定**：`mecanum_move_speed_times(dir, speed, dist, unit=1)`：0=前进/1=后退，dist 单位厘米；`mecanum_turn_speed_times(2/3=左/右转, speed, 角度°, unit=2)`；`mecanum_translate_speed_times(90/-90=右/左移, speed, dist_cm, unit)`。
- **AprilTag 五阶段对准**（apriltag_opt.py）：阈值 ±60→±30→精 ±16px；速度 搜索 8 / 粗 15 / 中 8；精调 FINE_KP=0.25、最低速 3；稳定确认连续 2 帧；TAG_SIZE='10'（距离取索引 9，米×100=cm）。
- **AprilTag 对准参数**（test_apriltag.py）：阈值 ±50→±25→精入口 ±10→死区 ±1，连续 3 帧稳定；速度 搜索 6 / 粗 12 / 中 8，精调 FINE_KP=0.35、最低速 2、小偏差强制速 1 防过冲；TAG_SIZE='10'，距离索引 '5'=7 / '7'=8 / '10'=9。
- **摄像头标定偏移**（main_old.py）：AprilTag 物理正对时识别偏移 +16px，所有偏差计算减 CALIBRATION_OFFSET=16；搜索阶段连续 2 帧 id 相同且偏差差 ≤5px 才认为数据稳定。
- **抓取/放置区冲突规则**（main_old.py）：抓取区与目标放置区编号相同改放 6 号区；返回中转区后需再次 AprilTag 对准修正车身朝向。
- **TFLite 输出顺序**（main_old.py）：ssd_mobilenet_v2_fpn_320.tflite 输出 out[0]=boxes（归一化 ymin,xmin,ymax,xmax）、out[1]=scores、out[2]=classes；输入 320×320 RGB，uint8 模型直接送 uint8 张量，否则 /255 转 float32。
- **视觉模式状态机**（test.py）：_current_vision_mode 标记 None/"raw"/"line"/"apriltag"；启动与退出须 close→open→load line_recognition→close 复位摄像头，否则语音识别失效；AprilTag 加载会覆盖巡线模型，复位前须判当前模式避免重复加载。
- **六区搜索框架**（test.py）：顺序 6→5→4→3→2→1；非 1/6 区过滤左右各 20% 边缘 + 面积 ≥2000（3 区 ≥500）；对齐阈值 ±18px、横移速 3 / 时长 1；连续丢帧 30 次退回搜索；3 区最多搜 3 次转 2 区；初始化 5 帧、摄像头重试 5 次。
- **语音任务解析**（test.py）：语音文本提取颜色（红/绿/蓝口语变体）与存储区编号（阿拉伯 1-6 或汉字 一-六 + "号"）。
- **模型参数**（test.py）：模型 C:\Users\16231\Desktop\ros\pb_tflite\ssd_mobilenet_v2_fpn_320.tflite + label_map.pbtxt；置信度阈值 0.5；标签到位面积阈值 14000px。
- **标签面积分级靠近策略**（test_v2.py）：<9000px 前进 5cm 再找；9000~11000 / 11000~15000 / 15000~17000 / 17000~21000 放下前各先前进 4/3/2/1cm；>21000 后退 6cm；前进速 5，每次移动后等 0.3s 稳定。
- **区域标签识别播报**（tf_test.py）：one~six 六区标签 TFLite 识别，取置信度最高者播中文（一号区~六号区）；同标签播报冷却 5s；识别间隔 0.1s；最低置信度 0.5。
- **tf-od 环境**：ugot 板载环境无 tensorflow；含 TF 脚本须用 `C:\Users\16231\anaconda3\envs\tf-od\python.exe` 运行（tflite.Interpreter num_threads=4）。
- **机器人端侧识别部署**（tf_test_robot.py）：板载 Linux 用 tflite-runtime + Pillow（无 opencv）；模型放 /root/models/；MobaXterm 传脚本到 /root/；ssh 后 `python3 /root/tf_test_robot.py`；本机 initialize('127.0.0.1')；Pillow 直接 JPEG→RGB numpy。

---

**作者**：赖智坤
**时间**：2026/10/17