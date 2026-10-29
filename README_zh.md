# UnifoLM-WLA-1.0
<div align="right"><a href="README.md"><kbd>English</kbd></a> | <a href="README_zh.md"><kbd>简体中文</kbd></a></div>
<div align="center">

[项目主页](https://unigen-x.github.io/unifolm-wla.github.io/) | [模型](https://huggingface.co/collections/unitreerobotics/unifolm-wla-10) | [数据集](https://huggingface.co/collections/unitreerobotics/unifolm-wla-10)

<p align="center">
  <a href="https://www.youtube.com/watch?v=GHySQMMrIa4">
    <img src="assets/unifolm-wla-zh-cover.png" alt="UnifoLM-WLA-1.0 中文视频" width="800">
  </a>
</p>

</div>

UnifoLM-WLA-1.0 是宇树科技全面升级的新一代通用人形机器人基础模型，拥有 6B 参数。模型依托大规模通用多模态感知与理解数据，融合以交互为中心的世界建模，全面提升空间感知与理解能力，在多项具身推理评测中达到业界领先。通过约 2,500 小时高质量真机数据训练，单模型统筹64个任务，覆盖桌面操作与全身移动操作，适配二指夹爪及多种五指灵巧手，具备跨任务、跨末端执行器的泛化能力。

## 📋 目录

- [新闻](#-新闻)
- [开源计划](#-开源计划)
- [机器人动作、状态与统计量处理说明](docs/robot_action_state_processing.md)
- [训练与评估动作专家](docs/train_action_expert.md)
  - [安装](docs/train_action_expert.md#安装)
  - [评估 Checkpoint](docs/train_action_expert.md#评估-checkpoint)
  - [模型部署服务端](docs/train_action_expert.md#模型服务)
  - [SONIC 机器人客户端与 MuJoCo 仿真](docs/sonic_simulation.md)
  - [SONIC 桌面抓取仿真与物理验收](docs/sonic_grasp.md)
  - [原始 VLA 动作头的零样本仿真排查](docs/sonic_vla_diagnostics.md)
  - [冻结 VLA 的目标锁定、执行反馈与重复验收](docs/sonic_vla_stable.md)
  - [WLA 无抓取接管基线、官方 WBC 检索与失败归因](docs/sonic_wla_model_only.md)
  - [WLA TCP/夹爪标定与无接管六轮复测](docs/sonic_wla_calibrated.md)
  - [微调 UnifoLM-WLA-1.0-Base 动作专家](docs/train_action_expert.md#微调-unifolm-wla-10-base-动作专家)
  - [从零训练](docs/train_action_expert.md#从零训练)
- [引用](#引用)
- [致谢](#致谢)
- [开源许可](#开源许可)

## 🔥 新闻
- 2026年9月28日：🚀 我们发布了 [UnifoLM-WLA-1.0-Base](https://huggingface.co/unitreerobotics/UnifoLM-WLA-1.0-Base) 和微调代码。
- 2026年9月20日：🚀 我们发布了模型结构和训练动作专家代码
- 2026年9月11日：🚀 我们发布了模型权重 [UnifoLM-ER-1](https://huggingface.co/unitreerobotics/UnifoLM-ER-1)
- 2026年9月11日：🚀 我们发布了模型权重 [UnifoLM-ER-Flow](https://huggingface.co/unitreerobotics/UnifoLM-ER-Flow)

## 📑 开源计划

- **代码**
  - [x] [基于 UnifoLM-ER 模型训练动作专家的代码](docs/train_action_expert.md#从零训练)
  - [x] [基于 UnifoLM-WLA-1.0-Base 的微调代码](docs/train_action_expert.md#微调-unifolm-wla-10-base-动作专家)
  - [x] [LoRA 微调代码](docs/train_action_expert.md#LoRA-微调)
- **模型**
  - [x] [**UnifoLM-ER-1**](https://huggingface.co/unitreerobotics/UnifoLM-ER-1)
  - [x] [**UnifoLM-ER-Flow**](https://huggingface.co/unitreerobotics/UnifoLM-ER-Flow)
  - [x] [**UnifoLM-WLA-1.0-Base**](https://huggingface.co/unitreerobotics/UnifoLM-WLA-1.0-Base)
- **数据集**
  - [x] [**UniBot-V1 Challenge Dataset**](https://huggingface.co/collections/unitreerobotics/unibot-v1-challenge-dataset)
  - [x] [**UnifoLM-WBT-Dataset**](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset)
  - [x] [**UnifoLM-Dex1-Dataset**](https://huggingface.co/collections/unitreerobotics/unifolm-g1-dex1-dataset)


## 📘 技术文档

- [机器人动作、状态与统计量处理说明](docs/robot_action_state_processing.md)
- [训练与评估动作专家](docs/train_action_expert.md)

## 引用

```bibtex
@misc{unifolm-wla-1.0,
  author = {Unitree},
  title  = {UnifoLM-WLA-1.0: One Model Driven, Whole-Body Coordination},
  year   = {2026},
}
```

## 致谢

本项目基于 [starVLA](https://github.com/starVLA/starVLA) 和
[Qwen-Image](https://github.com/QwenLM/Qwen-Image) 继续开发。诚挚感谢两个项目的作者与贡献者将其工作开放给社区。

## 开源许可

除另有说明外，本项目以 [Apache License 2.0](LICENSE) 开源。项目中的第三方组件仍适用其原始许可条款，归属与详细说明请参阅 [NOTICE](NOTICE) 和
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。
