# WLA 无接管校准配置：TCP、夹爪与底座执行

本轮按用户要求修正工具坐标、夹爪命令端点和底座低速执行，然后复跑与[上一批](sonic_wla_model_only.md)相同的六轮：左手红块、左手绿块、右手黄块各一次，以及红块场景三次 ±1 cm 位置扰动。官方权重保持冻结，抓取接管保持关闭。结果只评价“WLA + 当前 SONIC 适配器”；仍未获取配套 Unitree WBC 客户端，不能称为官方整套控制链复现。

**正式六轮已完成：任务成功 0/6，完整质量验收 0/6。其中四轮进入 WLA 执行，均未建立有效抓取；两轮在准备阶段停止。TCP 定义及夹爪电机端点已纠正，底座漂移有所减小，但整机执行稳定性没有修好。不能把该结果归纳为“WLA 本身不能抓取”。**

实际冻结配置为 [vla_model_only_calibrated_joint_pd.yaml](../robot_client/sonic/vla_model_only_calibrated_joint_pd.yaml)。带朝向 P 反馈的 [vla_model_only_calibrated.yaml](../robot_client/sonic/vla_model_only_calibrated.yaml) 已在开发对照中否决，仅保留失败复现。正式六轮使用与上一批完全相同的 `tabletop_rgb_{red,green,yellow}_safe.yaml`，逐轮核对生成场景一致；没有改变场景几何、官方权重或启用抓取接管。新配置是实验记录，不是通过稳定性验收的推荐配置。

## 1. 工具坐标：确定为腕前 105 mm，区分模型参考与实际机器人

官方同名字段 `left/right_ee_pose_gripper_base` 的含义是“夹爪末端在机器人 base 坐标系中的位姿”，其中 `base` 是表达参考系，不能据名称把末端原点当作夹爪 CAD 的 `base_link` 或指垫中心。见[官方接口定义](https://github.com/unitreerobotics/unibot_submission/blob/edd0a1c98630671db3375f9c2483d7870dbe031a/README.md#observation-catalog)。

用官方 G1 mode 15 运动学与公开数据的关节/末端观察对重算，左右工具变换均为：

```text
T_wrist_yaw_to_policy_tool:
  xyz = [0.105, 0, 0] m
  rpy = [0, 0, 0] rad
```

该结果来自不同姿态下 `inverse(T_wrist) × T_dataset_ee` 的一致性，不使用方块位置、抓取成功率或接触结果反推。[标定脚本](../robot_client/sonic/calibrate_tools.py) 用交替样本拟合及留出验证，并输出原始数据哈希、参考 URDF 哈希和残差。它确认的是数据标签中的工具坐标定义，**不等于完成了实机 TCP 测量**。

本轮核验 3,007 对官方观察，其中 1,504 对拟合、1,503 对留出验证。两侧留出位置残差最大 **2.12 × 10⁻⁸ m**，说明数据标签与该运动学定义一致；该数值不是实机标定精度。[完整标定证据](../results/sonic/grasp/model-only-calibrated-20261028/tool-calibration.json)同时保留旧 SONIC 链的对照，其固定工具拟合最大残差为 7.26 mm。

参考为[官方完整 G1 + Dex1 URDF](https://github.com/unitreerobotics/unitree_ros/blob/deeb607f3a8945a394f53068c28b3a88dde7ff06/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf)，SHA-256 为 `1a0a4f493d6686cd93952d8e67d6144d4c337929151c745e0440b8a28f08ed1a`，随本仓库保留其 BSD 3-Clause 许可证。

官方 mode 15 的腕 yaw 原点沿父坐标 X 为 51 mm，而现有 SONIC 旧链为 46 mm。此前左右约 106–109 mm、带横向偏置的拟合吸收了部分骨架差异，不能作为真实安装尺寸。本轮仅把工具定义改为两侧相同的 105 mm：**实际 SONIC 动力学模型、旧腕链及其 FK/IK 保持不改**，不把标定参考 URDF 偷换成受控机器人的骨架。

## 2. 夹爪：修正电机端点，保留现有 CAD 行程

官方 [Dex1 驱动](https://github.com/unitreerobotics/dex1_1_service/blob/ced3216290bb22dae356213993786beeaa7143ab/main.cpp#L64) 把 DDS `q` 乘减速比后发送到电机、把反馈除以减速比，故客户端命令是输出轴角度，不是毫米。[官方校准步骤](https://github.com/unitreerobotics/dex1_1_service/blob/ced3216290bb22dae356213993786beeaa7143ab/README.md#3--calibration) 要求完全闭合后标零；[官方遥操源码](https://github.com/unitreerobotics/xr_teleoperate/blob/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b/teleop/robot_control/robot_hand_unitree.py#L289) 使用 0–5.4 rad，并注明约 5.4 rad 对应 90 mm 开口。

| 参数 | 本轮约定 |
| --- | --- |
| 完全闭合电机命令 | 0 rad |
| 完全张开电机命令 | 5.4 rad，替换旧 5.5 rad |
| 电机到仿真夹爪 | 线性映射到当前 CAD 关节端点 |
| 每指仿真行程 | 44.5 mm，保持原端点不变 |

90 mm 是官方源码的近似开口说明；当前 CAD 两指合计行程为 89 mm，不能据近似规格擅自加长几何。本轮未找到可验证的非线性标定曲线，不加入经验非线性修正。训练数据的归一化范围、某批样本的最大开度与机械终点是三个不同概念。

**现有独立 Dex1 CAD 的 40 mm 腕部安装仍是仿真假设，本轮不改。** 新版官方完整装配的 41.5 mm 变换对应另一套网格坐标定义，不能只取该数字套用旧资产。新版官方指尖距腕约 184.76 mm，与现有约 187.3 mm 接近；这不支持通过整体前后移动约 50 mm 来“修好”抓取。工具参考点与接触几何中心不同，仍需分别审计。

## 3. 底座：过滤不受支持的微小平移，保留有效运动与朝向

[PlannerMapper](../robot_client/sonic/planner.py) 新增 `minimum_walking_speed_mps: 0.1`。依据是 SONIC 的 [`SLOW_WALK=1` 枚举说明](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1fa5f7fa3966cbd4845c4e641d13c3f0cd5b0d9b/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner.hpp#L77)将平移速度范围标为 0.1–0.8 m/s；这是控制接口范围说明，不是本场景已验证的实际速度精度。官方[手柄输入实现](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1fa5f7fa3966cbd4845c4e641d13c3f0cd5b0d9b/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/gamepad.hpp#L354)采用更高的 0.2 m/s 下限；本轮选择枚举的最低值 0.1 m/s，实际低速跟踪仍由实验检验。

本轮保留模型请求的平移方向、偏航和高度，按以下执行限制转换：

- 限速/加速度处理后的平移速度低于 0.1 m/s 时不进入 WALK，发送零平移；不把微小请求放大到 0.1 m/s。
- 达到最低速度后允许 mode 1 平移，仍受 `max_speed: 0.15`、`max_acceleration: 0.5` 约束。
- 静止时使用 mode 4，仍累计模型偏航为 `facing`，并保留受限高度；纯转向不再强行使用零平移 WALK。
- 控制记录区分模型请求、过滤后速度、实际发送的速度/模式及抑制原因，便于比较底座位移和模型速度积分。

在本地发送包里，静态模式的零速度表示没有平移请求；官方 [ZMQ 输入层](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1fa5f7fa3966cbd4845c4e641d13c3f0cd5b0d9b/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp#L604)会把静态模式内部速度设为 `-1`，不能把它误读为倒退命令。以上链接使用[原下载记录](sonic_simulation.md)中固定的 SONIC 快照版本。

这会显式舍弃接口范围以下的平移请求，属于执行限制。它不是基座位置保持：`base_position_hold: false`，`compensate_base_motion: false`；不根据物体或抓取误差纠正底座轨迹。

### 开发对照：朝向 P 反馈已否决

先做的三轮开发对照只包含 TCP/夹爪端点与最低步行速度修正，结果保留如下，不能混入随后正式六轮：

| 开发对照 | 结果 |
| --- | --- |
| 左手红块 | 仍撞桌，安全停止 |
| 左手绿块 | 仍撞桌，安全停止 |
| 右手黄块 | 执行预算结束，未抓取 |

开发分析发现准备期存在约 5° 的持续偏航偏差，随后试验了有界机器人朝向 P 反馈。**该方案未通过物理验证，已否决；以下参数仅保留用于复现失败，不作为正式六轮配置。**

被否决方案的模型目标 `target = self.facing` 在初始化后只积分模型请求的 `yaw_rate`，不被反馈重设。执行层对目标与实测朝向的环绕角误差做 P 校正：增益 2、校正幅度上限 0.15 rad、平滑时间常数 0.3 秒；发送朝向为模型目标加平滑校正，合成发送朝向变化率限制为 0.3 rad/s。

| 朝向反馈开发对照 | 结果 |
| --- | --- |
| 左手红块 | 准备期 XY 偏移超过 140 mm，准备检查超时 |
| 左手绿块 | 进入 2 个模型动作块后撞桌 |
| 右手黄块 | 准备检查超时 |
| 扰动批第 1 轮 | 已中断，未完成整批 |

这些失败及中断记录全部保留，不能被计为完成正式六轮。反馈虽然只使用机器人姿态，物理验证仍不支持采用它；本轮不继续通过调整 P 增益凑抓取结果，当前候选明确设置 `heading_tracking_gain: 0.0`。

## 4. 正式六轮执行方式：腰部与双臂直接 PD

`vla_model_only_calibrated_joint_pd.yaml` 使用 `arm_tracking: joint_pd`：腰部跟踪模型原腰部目标，双臂跟踪模型末端目标经 IK 得到的关节目标，SONIC 继续负责腿部平衡。它调整的是原目标的关节执行方式，不新增抓取目标、闭爪或抬升动作。

该候选关闭朝向 P 反馈，保留原生 PD 增益及 [SONIC 原始机器人模型](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1fa5f7fa3966cbd4845c4e641d13c3f0cd5b0d9b/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf)的腰部 yaw/roll/pitch 力矩上限 **88/50/50 Nm**，不提高力矩限制来强行跟踪。TCP、夹爪端点、最低步行速度和严格准备检查继续保留；`base_position_hold` 与所有抓取接管仍关闭。

四轮通过准备检查时，腰部误差约 0.0009–0.0018 rad，相比开发对照约 0.094 rad 明显下降。腰部补偿仍基于实测误差，每步从原模型目标重算；当腰部误差归零，双臂补偿也归零。直接 PD 并未解决腿腰平衡耦合，正式测试仍出现准备漂移和碰撞，故不能据腰部跟踪改善宣布整机稳定。

## 5. 准备检查：持续静止、腰部与末端达标后才交给模型

最终配置在原有姿态/末端条件之外要求：

- 腰部误差不超过 0.02 rad，较旧 0.2 rad 收紧；
- 底座相对准备参考的位置偏差不超过原有 0.12 m；
- 完整三维角速度范数小于 0.1 rad/s，不能只检查 yaw 速度；
- 平移速度小于 0.02 m/s；
- 上述条件连续满足至少 0.5 秒仿真时间。

开发时增加的“相对世界目标偏航不超过 2°”已移除：`joint_pd` 开发对照在速度 0.00125 m/s、角速度 0.0262 rad/s、腰误差 0.00727 rad 时仍因稳定的 4.59° 朝向偏差被阻挡。固定朝向偏差不等于正在漂移；任务没有要求对齐世界零朝向。最终保留实测角速度约束，关闭朝向 P 反馈，没有放宽原位置、碰撞和姿态限制。

准备完成后，在首次推理返回并开始执行模型动作之前继续复核；等待首个模型回复不能绕过检查。条件失效时恢复准备检查，重新积累稳定时间；45 秒超时停止并记录原因。准备期间的碰撞审计继续有效。这些条件是执行前约束，不利用目标物反馈，也不能用预备动作提前夹持或抬升目标。

## 6. 不变的任务所有权与评估规则

`execution_mode: model_only` 继续禁止 RGB 对准、目标选择、触觉闭爪、抓取状态机、抓取锁定、抬升反射、单臂固定及首帧轨迹重对齐。当前候选保留 IK、腰部实测姿态补偿、原目标的关节 PD 执行、关节/夹爪限制、数据过期与碰撞停止；关闭朝向 P 反馈。这些机制不能自行选择抓取目标或追加抬升动作。准备阶段与模型执行阶段分开统计，物体真值仅用于独立评估。

六轮均保留配置、源码/权重哈希、原始模型回复（有回复的轮次）、准备阶段碰撞记录、物理轨迹和录像。比较抓取与保持、穿透、底座请求/实际位移及耗时；未产生有效抓取时，不能把执行到时当作完成长时间保持。修正后失败仍需区分模型动作、执行跟踪和剩余几何假设。

## 7. 正式六轮结果

每轮客户端预算 60 秒，固定 CPU 核 `4,12,14,34,40,42`；扰动种子 `20261028`，各物体 XY 独立均匀扰动 ±0.01 m，与上一批生成的六个场景逐一相等。碰撞穿透阈值保持 1 mm。抓取评估要求至少 6 秒有效支撑保持，质量评估还要求滑移不超过 5 mm。

| 轮次 | 模型块 | 停止原因 | 前 3 秒底座误差：旧 → 新 | 全程最大穿透 |
| --- | ---: | --- | ---: | ---: |
| 左手红块 | 8 | 模型执行时撞桌 | 60.94 → 16.93 mm | 1.520 mm |
| 左手绿块 | 7 | 补偿后的 IK 目标未达到执行容差 | 52.77 → 21.34 mm | 0.757 mm |
| 右手黄块 | 0 | 准备位置偏移 191 mm，超过 120 mm 上限 | 不适用 | 2.557 mm，停止阶段 |
| 红块扰动 1 | 0 | 准备阶段撞桌 | 不适用 | 3.565 mm，停止阶段 |
| 红块扰动 2 | 8 | 模型执行时撞桌 | 58.02 → 12.11 mm | 1.271 mm |
| 红块扰动 3 | 4 | 模型执行时撞桌 | 50.60 → 9.25 mm | 1.038 mm |

四轮进入模型的有效支撑保持均为 0 秒；没有完成抓起或 6 秒保持。右手黄块和扰动 1 没有进入模型执行，不能当成两次模型抓取失败。完整结果见 [comparison.json](../results/sonic/grasp/model-only-calibrated-20261028/comparison.json) 与[逐轮原始审计](../results/sonic/grasp/model-only-calibrated-20261028/wla-calibrated-final-summary.json)。

正式四轮中，左 TCP 相对模型原目标的跟踪 RMS 仍为约 **35–42 mm**，峰值约 **59–78 mm**，故底座早期改善并不等于末端执行合格。没有任何一轮建立双指同时接触与有效支撑。原始动作也不能概括成“一律不闭爪”：绿块最后一个已接受动作块预测了从约 4.64 到 2.99 rad 的闭合，但执行在块内约第 20/30 帧因补偿 IK 容差失败停止，明显闭合的尾部尚未执行。红块也预测了部分闭合；两次进入模型的扰动测试仍基本张开。失败停止截断了后续动作，不能从这些前缀推断模型没有后续抬升意图。

底座误差定义为“模型执行起点之后的实际 XY 位移”与“按实测 yaw 将记录的模型平移速度积分到世界系”的差值，取前 3 秒最大范数。它不是相对物体误差，也不包含准备期已产生的偏移。积分使用约 10 Hz 控制记录，是离线估算。四个进入 WLA 的配对轮次中位数从 **55.40 mm 降到 14.52 mm（约下降 74%）**；两轮未执行模型的结果标为不适用，不能填零。四轮完整模型执行窗口的最大误差分别为 49.65、26.14、31.43、14.62 mm；早期改善不能外推到长期稳定。

碰撞没有通过验收。六轮初始化阶段未记录穿透，但扰动 1 的准备阶段已达到 1.616 mm；另外黄块与扰动 1 在失败/停止阶段继续出现接触，峰值分别为 2.557、3.565 mm。后两轮停机跌倒后自动复位，分别在 `failed` 标记之后 0.832、0.747 秒发生时钟回退，因此 `complete_step_coverage=false`。这不是首次停止的原因，也不是 5 ms 物理步长漏审；仍保留全流程失败判定，不能删除停机段改成通过。其余四轮步覆盖完整，六轮 `disabled_steps=0`。这些记录不支持“开场穿模已全部解决”。

客户端耗时依次为 26.62、23.97、50.59、8.44、25.20、19.68 秒；四轮模型执行的已记录仿真窗口为 14.00、11.14、13.00、7.48 秒。前者包括客户端准备与推理等待，未计入独立仿真初始化；后者是仿真时钟，两者不能相减或当成成功抓取耗时。由于本轮无成功抓取，不报告抓取提速比例。

## 8. 能确定的原因与仍缺失的验证

1. **旧 TCP 拟合混入了不同腕链的差异，旧夹爪全开端点也不符合官方约定。** 已按可核验定义纠正；仍没有实机 TCP 或非线性开口曲线，也没有更换当前物理网格及安装假设。
2. **原配置把低于 SONIC 接口说明范围的微小平移请求送入 WALK。** 本轮同时调整最低速度限制、工具定义和 `joint_pd` 后，早期误差减小；该配对实验没有隔离各项修改的贡献，不能把全部改善归因于速度过滤。静态模式也不是根部 XY 位置伺服，不能保证底座不漂。
3. **朝向反馈不是独立 yaw 伺服。** SONIC 将朝向变化用于全身轨迹重规划；开发反馈试验在模型开始前已出现大幅后退，已否决。直接跟踪腰部能减小腰误差，却仍可能改变腿腰平衡耦合。
4. **模型目标自身与场景接触也存在疑点。** 三轮开发对照中的红块记录，在接触时模型目标对应指几何已低于桌面约 11–14 mm，实际 TCP 反而在目标上方约 14 mm；后续块末端目标更低。该例不支持把撞桌全归因于机械向下超调。这是开发样本的几何诊断，不能外推成正式六轮全部失败原因。

因此，本轮确认的是“校准和执行限制改善了一部分问题，但当前 WLA + SONIC 链未达到可靠评估模型抓取能力的执行条件”。下一项必要条件是独立验证整机根部位置/速度及腿腰协调，并核对实际手部几何与观测域；仍需配套 WBC 或有等效跟踪证据的执行器。当前结果不足以证明 WLA 能独立抓起，也不足以证明它在官方执行链上不能抓起。

## 9. 复现与证据

证据目录：[model-only-calibrated-20261028](../results/sonic/grasp/model-only-calibrated-20261028)。其中 `calibrated-client.yaml` 是实际运行配置；`experiment-runtime.tar.gz` 是实验时源码，后续只改配置注释与报告。`run_six_suites.py` 保存本轮四个 suite、共六次尝试的原命令，拒绝覆盖已有汇总。所有失败开发记录及朝向反馈中断记录也保留，未混入正式六轮。

官方权重 SHA-256 保持 `65ac8911ce5556631cb538fb4d3499d9131d95b1da4eb1ce970c4215997d3299`，逐轮源码完整性检查均通过，六轮任务接管清单均为空。完整回归 **332 passed、1 skipped，38.00 秒**；跳过项为缺少 `third_party` 中独立官方 URDF，本轮新增参考 URDF 标定测试实际执行。实验模型服务已停止，15555–15558、18600 端口已释放。

先按[上一批模型服务命令](sonic_wla_model_only.md#复现与证据)启动相同权重服务。在 `pi0.5:/mlplatform/jy/unifolm-sonic/src`，以下命令使用新的输出目录复跑红块：

```bash
SONIC_ROOT=/mlplatform/jy/unifolm-sonic/GR00T-WholeBodyControl \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 4,12,14,34,40,42 ../venv-sim/bin/python -u \
  -m robot_client.sonic.grasp_repeat \
  --workspace /mlplatform/jy/unifolm-sonic --simulation \
  --arm-tracking joint_pd \
  --task robot_client/sonic/tabletop_rgb_red_safe.yaml \
  --config robot_client/sonic/vla_model_only_calibrated_joint_pd.yaml \
  --repeats 1 --seconds 60 \
  --output ../artifacts/grasp/wla-calibrated-final-red-repro
```

绿块、黄块分别将任务文件替换为 `tabletop_rgb_green_safe.yaml`、`tabletop_rgb_yellow_safe.yaml`，各运行一次并使用独立输出目录。扰动批继续使用红块任务，改为 `--repeats 3 --jitter-m 0.01 --scene-seed 20261028` 并另设输出目录。每轮独立重建仿真，不通过重复某一失败轮替换原结果。

对每个 `trial-NNN` 重算审计：

```bash
../venv-sim/bin/python -m robot_client.sonic.raw_grasp_quality \
  ../artifacts/grasp/wla-calibrated-final-red-repro/trial-001 \
  --hold-seconds 6 --max-slip-mm 5 \
  --output ../artifacts/grasp/wla-calibrated-final-red-repro/trial-001/raw-quality.json
../venv-sim/bin/python -m robot_client.sonic.execution_quality \
  ../artifacts/grasp/wla-calibrated-final-red-repro/trial-001 \
  --output ../artifacts/grasp/wla-calibrated-final-red-repro/trial-001/execution-quality.json
```

失败录像：[红块模型执行](../results/sonic/grasp/model-only-calibrated-20261028/wla-calibrated-final-red-001/trial-001/overview.mp4)、[黄块准备失败](../results/sonic/grasp/model-only-calibrated-20261028/wla-calibrated-final-yellow-001/trial-001/overview.mp4)、[扰动 1 准备碰撞](../results/sonic/grasp/model-only-calibrated-20261028/wla-calibrated-final-jitter-001/trial-001/overview.mp4)。
