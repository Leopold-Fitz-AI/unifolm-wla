# WLA 无抓取接管基线：官方客户端检索与实验边界

本轮目标是关闭已有 RGB 对准、触觉闭爪和抬升辅助，观察官方 WLA 权重产生的任务动作能否完成抓取。此前辅助配置的成功率不能用作 WLA 动作头独立抓取能力的证据。

## 官方配套 WBC 客户端检索结论

**在本轮核查的公开官方来源中，未找到可直接运行、与 WLA 服务器协议匹配的机器人 WBC 客户端。** 这表示尚未找到公开实现，不表示已经证明该客户端不存在。

官方 `unitreerobotics/unifolm-wla` 的 [部署文档](https://github.com/unitreerobotics/unifolm-wla/blob/0a1aa87be8f775c7883fcd27216e3666ac18a8c5/docs/train_action_expert_en.md#model-server) 提供模型服务器和录制数据评估脚本，说明服务器面向 Unitree G1 WBC 客户端；该说明没有给出真实机器人客户端的仓库、启动命令或 WBC 控制器配置。本文核对的官方提交为 `0a1aa87be8f775c7883fcd27216e3666ac18a8c5`，不是本地含适配代码的 HEAD。

公开问题也尚未补上这一环：[issue #20](https://github.com/unitreerobotics/unifolm-wla/issues/20) 询问 WLA 输出如何连接 IK/WBC，以及是否发布客户端和执行代码，目前页面没有维护者回复；[issue #17](https://github.com/unitreerobotics/unifolm-wla/issues/17) 询问闭环仿真环境，仅见其他用户附和。这些是用户提问，不能当成官方承诺或不发布声明。

### 匹配客户端必须满足的协议

[官方服务器源码](https://github.com/unitreerobotics/unifolm-wla/blob/0a1aa87be8f775c7883fcd27216e3666ac18a8c5/model_server/action_server_wbc_msgpack_unitree.py) 明确描述 `WebsocketClientPolicy` 契约：

| 项目 | WLA 服务器要求 |
| --- | --- |
| 通信 | WebSocket + msgpack；连接后接收 metadata；`get_action` 请求包含 `obs` |
| 相机 | `cam_left_high`、`cam_left_wrist`、`cam_right_wrist`，uint8 BGR |
| 机器人状态 | 双手 `*_ee_6d`，各 9 维 xyz + rot6d；双夹爪；15 维 lower_body |
| 末端动作 | `action.left_ee_rpy`、`action.right_ee_rpy`，各 `(1,T,6)`；由模型相对动作与当前末端状态组合得到绝对目标 |
| 其他动作 | 双夹爪、15 维 lower_body、4 维 base_command、7 维 pivot |

当前官方服务器只覆盖 Dex1 两指夹爪动作空间；不能把它直接当作 WBT 灵巧手/全身相对基座动作的完整服务。协议匹配之外，真实复现还需要确定末端坐标标定、IK、WBC 控制器和动作执行时序。

### 已排除的相似候选

| 候选 | 为什么不能认定为配套 WLA WBC |
| --- | --- |
| [UnifoLM-VLA-0 部署说明](https://github.com/unitreerobotics/unifolm-vla#-real-world-inference-evaluation) | 客户端安装链接指向旧 `unifolm-world-model-action`。后者的 [robot_client.py](https://github.com/unitreerobotics/unifolm-world-model-action/blob/main/unitree_deploy/scripts/robot_client.py) 使用 HTTP `LongConnectionClient`、G1 16 维双臂/夹爪动作和单头部相机；不是上述 WLA 全身接口。 |
| [UniBot submission](https://github.com/unitreerobotics/unibot_submission) | 虽含 `action.pivot`，但 [run_client.py](https://github.com/unitreerobotics/unibot_submission/blob/main/example/run_client.py) 调用的 [ExampleEnv](https://github.com/unitreerobotics/unibot_submission/blob/main/example/example_env.py) 仅生成占位观察、检查动作格式；没有机器人执行或 WBC。其 EE 字段也与 WLA 不同。 |
| [SONIC / GR00T WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | 来自 NVIDIA 的控制栈。本仓库对它的适配不能称为已获取或复现了 Unitree 配套 WBC。 |

## 本地无接管配置的含义

配置入口：[vla_model_only.yaml](../robot_client/sonic/vla_model_only.yaml)。执行检查位于 [run.py](../robot_client/sonic/run.py) 的 `execution_contract()`，记录 `official_wbc_client: false`、`execution_backend: sonic_planner_adapter`。

`execution_mode: model_only` 指 **WLA 独立决定任务动作**，并不表示关节命令逐值透传。现有链路仍为：官方模型服务器 → 本地坐标/IK/夹爪映射与限制 → SONIC planner/执行器 → 仿真。

该模式禁止 RGB 目标选择、抓取状态机、触觉锁定与力控闭爪、抬升反射、单臂固定策略、基座位置保持、首帧末端重对齐及 Cartesian 轨迹接管。`action_time_scale` 必须为 1，保留官方 encoder prompt 约定；启动前检查和实际运行入口都会校验配置，避免命令行覆盖后重新开启辅助。

保留的执行与安全机制包括：

- 预备姿态过渡与稳定检查、末端工具偏移、IK、关节/夹爪范围和速度限制、不可达动作处理、数据过期停止。
- SONIC planner 产生腿部参考；这不是把 WLA 的 15 维 lower_body 原样执行。基座速度、加速度、转速和高度仍受配置上限约束。
- `waist_tracking_compensation` 使用实测机器人姿态追踪两个模型末端目标；不选择目标物、不修改夹爪、不锁定抓取姿态。当前配置关闭 `compensate_base_motion`。
- 独立机器人反馈只允许仿真时间、关节状态及基座位姿等字段；不把物体位置、触觉接触力输入无接管执行逻辑。姿态异常或外部停止信号只能中止执行。

当前配置为模型时间步 30 Hz、控制循环 50 Hz，`execute_full_chunk: true`、`action_timebase: receipt`。这些执行参数属于本地适配选择，不能声称等同官方机器人客户端。预备动作耗时和模型开始后的任务耗时应分别报告。

模型服务器保留现有通用安全限制：归一化动作幅度 5、相邻步变化 10、物理关节范围、2 秒推理预算（超时可能返回上次保持姿态）；跨块混合为 0。这里的“原始回复”是服务器解码并经过这些安全限制的回复，不是未处理的神经网络张量。服务器虽公布候选生成和 grounding 能力，本配置只调用 `get_action`。`pivot` 是基座命令与腰部姿态的组合，适配器使用对应的 `base_command` 和 `lower_body` 字段。

## 结果解释规则

成功只能证明“官方 WLA + 当前无接管 SONIC 适配”在该场景下可完成任务。失败需要区分模型目标、IK/跟踪误差、输入相机/坐标差异、动作执行时序与碰撞问题；不能直接归结为 WLA 本身不会抓取。

穿桌、穿模需要单独记录实际几何接触/穿透与发生阶段，包括预备姿态阶段。此处关闭任务接管不构成已经解决碰撞的证据。独立审计可以触发停止，不能通过读取目标物真值悄悄修正抓取轨迹。

## 本轮实验结果

运行日期：2026-10-28 UTC。官方 `UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors` 权重未训练或改写，本轮实读 SHA-256 为：

```text
65ac8911ce5556631cb538fb4d3499d9131d95b1da4eb1ce970c4215997d3299
```

所有试验使用 `arms_pd`：双臂执行 IK 后的关节目标，SONIC 控制腿部及腰部；目标物真值只进入独立评估器。每次重启仿真、固定推理种子 0，CPU 亲和固定为 `4,12,14,34,40,42`，单次客户端预算 60 秒（包含约 12 秒准备与首次推理），没有因为碰撞失败而放宽安全阈值。

| 场景 | 接受的模型动作块 | 有效双指离桌支撑 | 结束原因 | 全程最大穿透 |
| --- | ---: | --- | --- | ---: |
| 左手红块 | 3 | 无 | 左夹指撞桌，安全停止 | 1.358 mm |
| 左手绿块 | 7（收到 8 次回复） | 无 | 左夹指撞桌，安全停止 | 1.262 mm |
| 右手黄块 | 26 | 无 | 执行预算结束，任务未成功 | 0 mm |
| 红块 ±1 cm 扰动，第 1 次 | 22 | 无 | 右夹指撞桌，安全停止 | 1.487 mm |
| 红块 ±1 cm 扰动，第 2 次 | 4 | 无 | 左夹指撞桌，安全停止 | 1.644 mm |
| 红块 ±1 cm 扰动，第 3 次 | 3 | 无 | 左夹指撞桌，安全停止 | 1.215 mm |

**本批物理抓取 0/6，6 秒保持质量 0/6。** 没有出现有效双指离桌支撑段，故尚无可用于评价 30 秒长时间保持的抓取。三次扰动同样在初始化及准备阶段零穿透；所有六次源文件哈希均证明运行期间代码未变，最终独立审计确认无任务接管且准备阶段未提前夹持或抬升目标。

红块首次记录模型执行为仿真 38.605 秒，首次超限为 42.960 秒；绿块对应 38.805 秒和 50.900 秒。黄块客户端正常结束只代表执行循环完成，不代表抓取成功：没有有效抬升，还推动了非目标红块约 10.94 mm。三种名义场景的初始化及准备阶段均为零穿透，45 个机器人碰撞几何始终启用，物理步覆盖完整。

### 红块撞桌的几何归因

用原始回复、同步基座姿态、现有 TCP 偏移和 Dex1 指尖网格重建目标几何：第三个动作块在首次碰撞附近要求左夹指最低点进入桌面约 **22–23 mm**（控制采样时间约有 20 ms 不确定性），该块末端目标约为桌下 **31.5 mm**。实际指尖受到桌面接触阻挡，实际末端反而比目标高约 23.8 mm；重建实际指尖约 1.03 mm 的穿透与物理审计吻合。

因此本例不能解释为“安全目标仅因执行跟踪误差被压到桌下”。在当前坐标、TCP 和场景约定下，模型目标已经与桌面几何冲突。仍需核实这些约定是否匹配官方训练及 WBC；缺少官方客户端时不能据此断言 WLA 网络本身无能力。IK 求解残差很小也不等于轨迹无碰撞。

还有独立的执行层偏差：约 4.36 秒模型阶段内，底座实际向前移动 **54.1 mm**，而模型速度积分仅要求前后方向约 **−1.9 mm**。当前配置取消了小速度死区，SONIC 对微小非零速度也进入 WALK；这暴露出其低速执行与目标命令不一致。碰撞附近抓取中心横向偏离红块约 **42.5 mm**，夹指内侧距离约 **76.1 mm**，目标块边长 40 mm，尚未闭合夹持。

这些证据把后续重点指向官方执行协议、TCP/安装标定和底盘小速度行为，而不是继续添加自动闭爪/抬升来掩盖偏差。复现几何分析见 [`analyze_red_collision.py`](../results/sonic/grasp/model-only-20261028/wla-model-only-red-001/analyze_red_collision.py) 与 [`red-collision-analysis.json`](../results/sonic/grasp/model-only-20261028/wla-model-only-red-001/red-collision-analysis.json)。

### 复现与证据

完整证据在 [`results/sonic/grasp/model-only-20261028`](../results/sonic/grasp/model-only-20261028/)。各目录保留实际配置、源文件哈希、原始回复、三路输入图像、控制轨迹、物理日志及从初始化开始的 `overview.mp4`。`experiment-runtime.tar.gz` 保存本批实际运行源码，任何失败都未删除。

试验结束后另外加固了两处入口/监督边界：`model_only` 必须选择 VLA 客户端；首次碰撞报告最多等待 60 秒，之后 5 秒无有效物理进展即停止。没有重新运行物理批次，已执行针对性回归。最终审计还拒绝准备阶段已夹持/抬升或首条模型记录前归因不明确的“成功”；全部六次已重新离线审计，结果见各目录 `raw-quality-final.json` 和总表 [`summary.json`](../results/sonic/grasp/model-only-20261028/summary.json)。原运行版本保存在上述源码快照中。

最终完整回归：**275 passed, 1 skipped，38.51 秒**。跳过项为未克隆 `third_party` 中官方 URDF 的独立测试；SONIC/Dex1 几何和本轮新增执行、监督、审计测试已运行。回归发现并修复了短时间重写报告时文件元数据不变的兼容问题：监视器以物理步数严格增长作为活跃证据，重复写入相同步数不能刷新期限。本轮模型与仿真进程已停止，15555–15558 和 18600 端口均已释放。

在 `/mlplatform/jy/unifolm-sonic/src`，先启动模型服务：

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
  ../venv-model/bin/python -u -m model_server.action_server_wbc_msgpack_unitree \
  --ckpt_path ../models/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
  --host 127.0.0.1 --port 18600 \
  --instruction "pick up the red cube with your left hand" \
  --image_size 336 448 --strict_rot6d \
  --joint_limits model_server/joint_limits.g1_dex1.yaml
```

使用未存在的新输出目录运行；绿块、黄块分别换为 `tabletop_rgb_green_safe.yaml`、`tabletop_rgb_yellow_safe.yaml`，任务指令随场景更新：

```bash
SONIC_ROOT=/mlplatform/jy/unifolm-sonic/GR00T-WholeBodyControl \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 4,12,14,34,40,42 ../venv-sim/bin/python -u \
  -m robot_client.sonic.grasp_repeat --workspace .. --simulation \
  --arm-tracking arms_pd --task robot_client/sonic/tabletop_rgb_red_safe.yaml \
  --config robot_client/sonic/vla_model_only.yaml --repeats 1 --seconds 60 \
  --output ../artifacts/grasp/wla-model-only-red-NEW
```

扰动批使用同一红块命令，改为 `--repeats 3 --jitter-m 0.01 --scene-seed 20261028`。每个物体的 XY 独立均匀扰动，实际坐标只供场景生成及评估。

```bash
../venv-sim/bin/python -m robot_client.sonic.raw_grasp_quality \
  ../artifacts/grasp/wla-model-only-red-NEW \
  --hold-seconds 6 --max-slip-mm 5 \
  --output ../artifacts/grasp/wla-model-only-red-NEW/raw-quality.json
```

抓取失败时审计命令返回非零是预期行为。该审计不要求辅助状态机事件，独立报告物理支撑、完整保持窗口、滑移、碰撞、动作来源和准备阶段归因；没有有效抓取时不能声称完成了长时间保持验证。
