# SONIC 桌面抓取仿真

本任务在 G1 全身 MuJoCo 模型中加入桌子和两块 5 cm、60 g 的自由方块。SONIC 原生 planner/encoder/decoder 负责双足平衡与全身跟踪，双臂通过 IK 下发目标，夹爪通过原生手部通道驱动。桌面任务使用明确标注的两指夹爪替身，保留原来的 29+7+7 个 SDK 关节和执行器顺序；它不是经过标定的 Dex1 CAD。

## 已完成的物理验证

2026-10-27 UTC 在 `pi0.5` 使用真实 UnifoLM Base 权重和 SONIC V2 完成下列两条指令。最终桌面高 0.85 m，两块方块中心的初始平面位置为前方 0.27 m、左右各 0.18 m。

| 指令 | 最低点最大离桌高度 | 双指连续有效保持（仿真时间） | 独立评分 |
| --- | --- | --- | --- |
| `pick up the red cube with your right hand` | 20.77 cm | 6.94 s | 通过，`red-vlm-007` |
| `pick up the blue cube with your left hand` | 20.99 cm | 6.76 s | 通过，`blue-vlm-002` |

两轮均无吊带、无仿真重置，另一块方块未被抬起。此前右手红块复测 `red-vlm-006` 也通过（17.95 cm、6.92 s）；更早的调参失败和蓝块远侧位置的可达性失败均保留。以上是开发验收案例，不是随机任务分布上的成功率。

服务器结果目录为 `/mlplatform/jy/unifolm-sonic/artifacts/grasp/`，本机副本为 `results/sonic/grasp/`。最终两轮的 `overview.mp4`、物理轨迹和模型输入/回复均可复查。回归测试共 99 项通过。试验结束后停止本任务的仿真、控制器及模型服务，保留源码、环境、权重与结果。

## 两条策略路径

`robot_client.sonic.run --config robot_client/sonic/grasp_client.yaml` 使用 UnifoLM 原始 VLA 动作头。当前零样本仿真试验没有通过抓取验收，失败记录保留在服务器 `artifacts/grasp/red-zero-shot-*`。模型服务连通、IK 可解或动作执行结束均不代表抓取成功。

后续保持官方权重的专门排查见 [原始动作头诊断](sonic_vla_diagnostics.md)：已出现最高 19.51 cm 的真实双指抬升，但连续保持 1.78 s，仍未达到 2 s 标准；该记录使用上身位置伺服和 SONIC 腿部平衡。不能与下述视觉加几何控制的通过结果混算。

进一步加入触觉夹持保持的执行路径见 [冻结 VLA 抓取保持](sonic_vla_stable.md)。非 reflex 模式由 VLA 产生接近和抬升；reflex 模式明确包含有限补高辅助。来源标签和统计分别保留，目前尚未通过稳定性验收。

`robot_client.sonic.visual_grasp` 使用混合控制：同一份官方 UnifoLM Base checkpoint 的 Qwen VLM 根据指令在头部相机中指向目标，RGB-D 像素和相机标定将其转为三维位置，几何状态机执行接近、下降、稳定、闭合、抬升和保持。模型的原生 `point_2d` 回复会被保存；颜色由模型指向位置的图像像素识别，左右手由指令中的显式手臂词解析。该路径的结果应称为 **UnifoLM 视觉定位 + RGB-D/几何控制 + SONIC**，不能归为原始 VLA 动作头的成功。

RGB-D 通道仅包含图像、深度和机器人相机标定，不包含物体 ID、位置、接触或评分器数据。物体尺寸为这个方块基准的已知规格。目标包围框内的颜色与深度点用于定位，没有从任务配置读取目标坐标执行抓取。

## 运行一次试验

在 `pi0.5` 的 `/mlplatform/jy/unifolm-sonic/src` 执行。模型服务需按 [基础部署说明](sonic_simulation.md) 启动于 `127.0.0.1:18600`，并使用包含 `ground_object` 能力的新服务代码。首次加载模型后再运行：

```bash
../venv-sim/bin/python -m robot_client.sonic.grasp_trial \
  --workspace .. --simulation \
  --task robot_client/sonic/tabletop.yaml \
  --output ../artifacts/grasp/my-red-trial
```

输出目录必须尚不存在。启动器检查本任务的端口空闲，启动自己的官方仿真和控制器，在 CONTROL 状态出现时释放初始吊带，完成试验后仅停止自己启动的进程。推理服务继续运行。仿真/控制使用 GPU 0，现有推理服务使用 GPU 3；不需要独占 GPU。相机、动作、状态、RGB-D 分别使用回环端口 15555–15558，DDS domain 73。

测试左手抓蓝块时将 `--task` 改为 `robot_client/sonic/tabletop_blue.yaml`，并使用新的输出目录。两个配置使用同一桌面布局，仅 `instruction`、`target`、`required_hand` 不同。保留两个物体可检验指令是否选择正确目标。`task.yaml` 同时供独立评分器设置验收目标；视觉控制器只接收其中的指令。

## 物理验收和产物

只有同时满足以下条件才通过：指定物体的最低点离桌至少 8 cm；指定手的两个不同夹指均有实际接触力；连续保持至少 2 秒**物理仿真时间**；物体没有桌面支撑；吊带已释放；机器人没有倒地或重置；另一个物体没有被抬起。启动器只有读到独立 `task-report.json` 的 `success: true` 才返回成功。

物体为自由刚体，抬升依靠接触与摩擦，没有附着、焊接或直接改写物体位置。初始 SDK 姿势的手低于桌面，因此 reset/ready 阶段临时关闭机器人与桌子/物体的碰撞；进入策略阶段恢复几何体和 MuJoCo 刚体 broadphase 的全部碰撞掩码。物体与桌面的碰撞始终存在。

每轮目录保存：

- `task-report.json`、`physics.jsonl`：独立物理评分和完整采样。
- `client.json`、`episode.json`：控制序列结果与策略来源；单独的客户端 `passed` 仅代表序列完成。
- `grounding-input.jpg`、`grounding-raw.json`、`grounding.json`：模型实际看到的画面、原始回复与解析结果。
- `grounding-sensor.npz`、`visual-target.json`：初始深度、标定和视觉估计位置。
- `overview.mp4`、各相机关键帧、组件日志、解析后的配置与任务 YAML。

原始失败试验保留，不从成功率分母中悄悄删除；开发调参轮次不构成统计意义上的泛化成功率。

## 已修复的接口与标定问题

- MuJoCo 输出 RGB，而上游 OpenCV JPEG 发布器按 BGR 编码。发布前转换后，模型端的既有 BGR→RGB 恢复正确颜色。
- 根据官方数据集的关节与末端**观测**拟合 gripper_base 工具偏移，左右位置 RMS 约 1.12/0.98 mm；未使用未来动作作为控制轨迹。
- Dex1 原始电机位置增大表示张开：约 5.5 为开、0 为闭。客户端显式转换到仿真夹爪闭合比例。
- MuJoCo 动态恢复碰撞时同时更新 `geom_conaffinity` 和 `body_conaffinity`，避免仅改几何体掩码仍被 broadphase 排除。
- RGB-D 顶面定位保留腐蚀后分裂的图像区域，静态双物体标定测试的位置误差约为 2.25/2.82 mm。这是特定标定画面的误差，不代表整个运动中的定位精度。
- 腰部参考保持在稳定站姿，避免把测得的跟踪偏差逐帧反馈为新的腰部目标；所选手的末端位置由测量反馈积分修正，另一只手保持准备姿势的关节参考。

回归测试命令同 [基础部署说明](sonic_simulation.md)。物理夹爪台架测试使用独立竖直伺服夹具验证接触抬升和松手后自然下落，它不作为 SONIC 全身任务通过的证据。
