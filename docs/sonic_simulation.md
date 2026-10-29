# UnifoLM-WLA → SONIC 仿真客户端

本客户端连接现有 UnifoLM WebSocket/MsgPack 服务，并使用 NVIDIA SONIC 的原生 ZMQ 协议控制 G1。推理服务器与客户端使用独立 Python 环境；客户端不导入训练栈。当前实现仅用于仿真。

桌子、自由物体、两指夹爪和按指令抓取的后续工作见 [桌面抓取说明](sonic_grasp.md)。下文验收表记录的是此前的站立与上身动作试验。

## 已完成的物理验收

2026-10-27 UTC，在 `pi0.5` 使用真实 UnifoLM Base 权重、SONIC V2 planner/encoder/decoder 和官方 MuJoCo/DDS 仿真完成一次无吊带闭环。测试指令为 `stand still and keep both hands steady`，不是抓取任务。

| 项目 | 实测结果 |
| --- | --- |
| 运行时间 | 墙钟 45.013 s，含 5 s 原生站立预热；无吊带物理仿真推进 30.750 s |
| 已接纳模型动作 / 控制包 | 51 段 / 2143 包；过期动作 0 段 |
| 落地稳定后基座高度 | 0.7848–0.7880 m |
| 落地稳定后最大倾角 / 平面位移 | 6.83° / 3.56 cm |
| 实测手臂单关节最大变化范围 | 1.593 rad，来自 51 次真实反馈采样 |
| 最大 IK 位置 / 旋转残差 | 19.85 mm / 0.00382 rad，限速前的解误差 |
| 最大推理与 IK 总延迟 | 1.526 s，低于配置的 2 s 预算 |
| 受限处理 | 2 段动作截断，共 24 帧；151 个小幅越界夹爪预测值被限幅 |
| 仿真重置 / 回归测试 | 0 次 / 84 项通过 |

稳定区间从运行开始后第 6 秒计，排除落地瞬态；统计截至客户端发送停止命令前。报告在服务器 `artifacts/validation.json`，本机副本在 `results/sonic/validation.json`，外部相机画面为 `results/sonic/snapshot-overview.jpg`。此前的 1 cm 容差运行因模型给出不可达目标而停止，失败日志仍保留。最终结果仅证明站立和上身动作链路连通；未验证抓取成功率、长期稳定性或实机操作。验收后已停止本任务启动的模型、控制器与仿真进程，保留环境和权重。

## 两种执行方式

默认 `sonic_mode: planner`：

```text
MuJoCo 三路相机 + SONIC g1_debug 状态
                 ↓
UnifoLM-WLA：双臂末端位姿、夹爪、腿腰、底盘速度和高度
                 ↓
双臂 IK + 腰部目标 + 物理速度限制
                 ↓
SONIC ZMQ manager
  planner：vx/vy/vw/height + 17 维上身关节 + 双手
                 ↓
SONIC 导航规划器生成腿部参考 → 编码器 → 全身控制策略 → MuJoCo
```

**控制权约定：planner 模式下，12 个腿关节由 SONIC 规划器负责。** 模型预测的腿关节仍保存在原始回复中，但不直接下发，避免同时让模型和规划器控制同一组腿关节。模型负责双臂末端目标、腰部、底盘速度和高度；IK 保持测得的腿部配置，求解双臂。

`sonic_mode: joint` 是诊断路径：IK 后的 29 维关节参考通过 SONIC protocol v1 发送。该路径执行模型的腿腰预测，但没有独立底盘速度/高度入口，因此只接受原地命令，超出配置死区的移动命令会报错。

两条路径都不是 SONIC 的 token-v4 客户端：不要求重新训练 UnifoLM 来输出 SONIC 的 64 维 token。

## 源码与兼容版本

- SONIC 源码快照：`NVlabs/GR00T-WholeBodyControl@1fa5f7fa3966cbd4845c4e641d13c3f0cd5b0d9b`。
- ZMQ 消息为单帧：topic + **1280 字节 JSON header** + little-endian 数组。
- protocol v1 输入为 IsaacLab 顺序的 29 维绝对关节角；`g1_debug.body_q` 已是 MuJoCo/SDK 顺序的绝对关节角，**不再添加默认角度**。
- planner 的 17 维上身位置/速度使用 IsaacLab 上身顺序，包含腰 3 维和双臂各 7 维。
- 兼容性测试直接导入该快照的原生消息构造器，逐字节比较 pose、planner、command 消息，并对照 C++ 的关节排列常量。
- 本任务在 `deps/trt10` 使用独立 TensorRT 10.13.3.9 运行库和匹配头文件，构建使用 FP32。服务器的 TensorRT 11 能运行 encoder/decoder，但不能构建当前 planner 图，因此完整链路使用 TensorRT 10。
- `prepare_build.py` 仅修补本任务外部快照的 TensorRT API 兼容分支，并增加 `SONIC_DDS_DOMAIN` 环境变量。不会修改系统 TensorRT、CUDA 或现有训练环境。

## 当前服务器目录

SSH 别名为 `pi0.5`，本任务使用 `/mlplatform/jy/unifolm-sonic`：

```text
src/                         此项目与客户端
GR00T-WholeBodyControl/       固定版本 SONIC 源码及 G1 网格
models/UnifoLM-WLA-1.0-Base/  官方 Base checkpoint、tokenizer、统计量
models/GEAR-SONIC/            SONIC encoder/decoder/planner 与 TRT 缓存
venv-model/                  推理环境，复用服务器已有 torch/torchvision
venv-sim/                    MuJoCo、Unitree SDK、客户端环境
deps/                        任务内开发头文件、sysroot 和运行库
build-controller/            C++ 构建产物
artifacts/                   配置、参考、动作回复、物理测量及验收报告
logs/                        各组件日志
```

仿真固定使用回环 `lo`、DDS domain **73**；相机/动作/状态端口分别为 **15555/15556/15557**，模型服务仅监听 **127.0.0.1:18600**。GPU 通过 `CUDA_VISIBLE_DEVICES` 指定，避免使用所有 GPU。

## 启动

以下命令在服务器执行。先确认端口未被其他程序占用，不要重复启动同一组件。

当前服务器的依赖和控制器已在上述目录准备好。需要重新编译时执行：

```bash
cd /mlplatform/jy/unifolm-sonic/src
../venv-sim/bin/python -m robot_client.sonic.prepare_build --workspace ..
```

该命令要求 `deps/headers/tensorrt10`、`deps/trt10/tensorrt_libs`、`deps/sysroot` 和 ONNX Runtime 已就绪。TensorRT wheel 来自 NVIDIA 官方索引，版本 `10.13.3.9`，SHA256 为 `bf2008eca911411aa93b852825ea992de33396451ea11713a6eb97c411d3b2e9`；匹配头文件来自 TensorRT `release/10.13` 和 onnx-tensorrt `10.13-GA`。首次启动会为当前 GPU 生成 TRT 缓存，等待控制器日志显示初始化完成后再启动客户端。

### 1. 模型服务

```bash
cd /mlplatform/jy/unifolm-sonic/src
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
../venv-model/bin/python -m model_server.action_server_wbc_msgpack_unitree \
  --ckpt_path ../models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
  --host 127.0.0.1 --port 18600 \
  --instruction "stand still and keep both hands steady" \
  --image_size 336 448 --strict_rot6d \
  --joint_limits model_server/joint_limits.g1_dex1.yaml
```

Base checkpoint 同时包含两组统计量，客户端明确选择 `UnifoLM_G1_Dex1`，不能依赖默认选择。首次模型初始化和 GPU 冷启动比稳定推理慢。

### 2. MuJoCo 物理仿真与相机

```bash
cd /mlplatform/jy/unifolm-sonic/src
PYTHONPATH=../GR00T-WholeBodyControl:$PWD MUJOCO_GL=egl \
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
../venv-sim/bin/python -m robot_client.sonic.launch_sim \
  --sonic-root ../GR00T-WholeBodyControl --headless --overview \
  --camera-port 15555 --domain-id 73 \
  --release-file ../artifacts/release-sim \
  --metrics-file ../artifacts/physics-planner.jsonl
```

启动前确保上次的 `release-sim` 文件已移除；初始虚拟吊带保持启用。`launch_sim` 在临时 MJCF 中增加头部、左腕、右腕相机，编译后校验资源路径，不改 SONIC 原始场景。它使用官方 MuJoCo 动力学和 SDK bridge；运动学用的去网格 URDF 不承担物理仿真。

需要窗口时去掉 `--headless`，并使用可用的显示环境。头/腕相机是仿真测试安装位置，尚未复现训练数据的实机标定。

### 3. SONIC 控制器

```bash
cd /mlplatform/jy/unifolm-sonic/src
../venv-sim/bin/python -m robot_client.sonic.make_stand_reference \
  --sonic-root ../GR00T-WholeBodyControl --output ../artifacts/startup-reference
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
../venv-sim/bin/python -m robot_client.sonic.launch_controller \
  --workspace .. --input-mode planner --seconds 180
```

该启动器为官方控制器提供持久 PTY，便于无桌面服务器运行，并在到时后停止自己启动的子进程。它强制回环接口和 DDS domain 73。规划器文件放在官方约定的 `planner/target_vel/V2/` 下，因为上游根据路径判断版本。

### 4. 客户端

新一轮控制器日志重建、显示 `Init Done` 后，先在另一终端运行以下有界等待命令，再启动客户端。它只创建仿真释放文件，不访问机器人硬件：

```bash
cd /mlplatform/jy/unifolm-sonic/src
../venv-sim/bin/python - <<'PY'
import time
from pathlib import Path
root = Path('/mlplatform/jy/unifolm-sonic')
deadline = time.monotonic() + 40
while time.monotonic() < deadline:
    if 'transitioning to CONTROL' in (root / 'logs/controller.log').read_text():
        (root / 'artifacts/release-sim').touch()
        break
    time.sleep(0.1)
else:
    raise RuntimeError('控制器未启动，保持仿真吊带')
PY
```

```bash
cd /mlplatform/jy/unifolm-sonic/src
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
../venv-sim/bin/python -m robot_client.sonic.run \
  --sonic-root ../GR00T-WholeBodyControl \
  --config ../artifacts/sim.yaml --simulation --max-seconds 30 \
  --report ../artifacts/client-report.json
```

`artifacts/sim.yaml` 是根据仓库 `robot_client/sonic/sim.yaml` 生成的服务器配置，修改上述四个端口并保留 `sonic_mode: planner`。客户端发送 manager 的 start/stop 命令；不用向 manager 输入旧关节流模式的 `]` 或 Enter。

客户端先启动原生 IDLE 规划器，因为 SONIC 进入 CONTROL 后才发布 `g1_debug`。看到控制器日志的 `transitioning to CONTROL` 后，应立即创建本任务的 `artifacts/release-sim` 文件释放虚拟吊带；不要让行走控制器在悬空状态持续运行。`startup_hold_seconds: 5` 给落地稳定预留时间，然后才请求模型动作。应分别记录吊带内检查与无吊带的实际运行区间，不能将吊带支撑下的姿态当作独立站立。

## 动作与失效处理

- 坐标系由 `reference_frame`（默认 pelvis）和左右 `tool_offsets`（xyz/rpy）明确配置。观察 FK 和动作 IK 使用同一工具定义；旋转 6D 使用前两列列优先展开。
- 双臂 IK 使用 MuJoCo 解析雅可比，保留小姿态正则抑制 7 自由度手臂的零空间跳变。
- 仿真配置允许 IK 位置残差最多 2 cm、旋转残差最多 0.05 rad，超过即拒绝。这是未完成工具标定时的连通验证容差，不代表精确操作精度。报告记录整个已接纳 chunk 的最大残差；该数值针对限速前 IK 解，不能当作物理末端跟踪误差。
- `truncate_infeasible_tail: true` 时，若后半段预测超出 IK 可达域，只保留至少两帧的可达前缀并重新规划；报告记录截断的 chunk/帧数。首两帧不可达、关节限位错误或非法数据仍直接停止。
- 模型时间步为 30 Hz；SONIC 参考为 50 Hz。默认 `action_timebase: receipt` 从推理和 IK 完成时开始执行新的绝对目标；原始观测时间仍记录，超过 `max_action_latency: 2.0` 秒的回复丢弃。`observation` 模式按请求时间锚定并跳过已经过去的动作，短前缀在共享 GPU 延迟较高时可能全部过期。两种模式都对首步和新 chunk 接缝按物理 rad/s 限速。
- 接收观测保留真实测量；仅 IK 种子允许小范围软限位误差并裁剪回合法范围。生成的关节参考受严格 URDF 限位约束。
- planner 路径将 body-frame vx/vy 转到初始航向参考系，积分受限 yaw rate；测试默认限制平移速度 0.4 m/s、yaw rate 0.4 rad/s、平移加速度 0.5 m/s²，高度限制在 0.5–0.8 m。训练数据的速度坐标定义仍须与实际数据采集配置核对。
- 超过动作时域不延长行走命令；传感器停更、推理断开、非法目标或参考长期未更新会停止发布。planner 模式退出时重复发送 manager stop；joint 模式需要操作员在官方控制器终端按 O。
- 不补零或复制相机来假装三路图像齐全。
- 官方 Base 统计量中的 Dex1 夹爪 offset 和 scale 约为 2.75，模型域范围约为 0–5.5。通过 `gripper_range` 显式映射到 SONIC 的模拟 Dex3 开合端点。这是**仿真替身**，不代表物理 Dex1 驱动、极性或力控已完成标定。
- `gripper_overshoot_tolerance: 0.05` 允许回归模型在该范围内略微越界，限幅到有效端点并记录数量；更大的越界仍拒绝。

## 验证层级

本机无 GPU 的回环测试：

```bash
python -m venv .venv-sonic
.venv-sonic/bin/pip install -r robot_client/requirements.txt pytest
SONIC_ROOT=/path/to/GR00T-WholeBodyControl \
  .venv-sonic/bin/python -m pytest model_server/tests -q
.venv-sonic/bin/python -m robot_client.sonic.smoke \
  --sonic-root /path/to/GR00T-WholeBodyControl \
  --report results/sonic/protocol_smoke.json
```

`smoke` 明确使用确定性模型回复和运动学接收替身，验证真实网络传输、G1 FK/IK 与时序，不声称运行了 SONIC 神经控制器或真实 UnifoLM 权重。

服务器物理验收必须同时检查：真实权重服务日志的 `get_action`、客户端接纳的 chunk 和包计数、SONIC 模型维度/规划器状态、无吊带物理测量。原始模型回复可通过 `trace_dir` 保存为 NPZ，便于复查速度限制前的动作。

仿真链路连通不等于抓取任务成功。默认场景用于站立连通测试；新增桌面场景与独立抓取评分器见 [桌面抓取说明](sonic_grasp.md)。

## 上游参考

- [SONIC 流协议](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/zmq.html)
- [SONIC 源码](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/1fa5f7fa3966cbd4845c4e641d13c3f0cd5b0d9b)
- [官方消息构造器](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1fa5f7fa3966cbd4845c4e641d13c3f0cd5b0d9b/gear_sonic/utils/teleop/zmq/zmq_planner_sender.py)

SONIC 源码许可证和模型权重许可证分别保留在外部仓库与模型目录中。
