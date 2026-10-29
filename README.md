# UnifoLM-WLA-1.0
<div align="right"><a href="README.md"><kbd>English</kbd></a> | <a href="README_zh.md"><kbd>简体中文</kbd></a></div>
<div align="center">

[Project Page](https://unigen-x.github.io/unifolm-wla.github.io/) | [Models](https://huggingface.co/collections/unitreerobotics/unifolm-wla-10) | [Datasets](https://huggingface.co/collections/unitreerobotics/unifolm-wla-10)

<p align="center">
  <a href="https://www.youtube.com/watch?v=GHySQMMrIa4">
    <img src="assets/unifolm-wla-en-cover.png" alt="UnifoLM-WLA-1.0 video" width="800">
  </a>
</p>

</div>

UnifoLM-WLA-1.0 is Unitree Robotics' comprehensively upgraded, next-generation general-purpose humanoid robot foundation model with 6B parameters. Built on large-scale general multimodal perception and understanding data and interaction-centric world modeling, it substantially advances spatial perception and understanding, achieving leading results across multiple embodied reasoning benchmarks. Trained on approximately 2,500 hours of high-quality real-robot data, a single model coordinates 64 tasks spanning desktop manipulation and whole-body manipulation. It supports two-finger grippers and multiple five-finger dexterous hands, with strong generalization across tasks and end effectors.

## 📋 Table of Contents

- [News](#-news)
- [Open-Source Plan](#-open-source-plan)
- [Robot Action, State, and Statistics Processing Specification](docs/robot_action_state_processing_en.md)
- [Train and Evaluate an Action Expert](docs/train_action_expert_en.md)
  - [Installation](docs/train_action_expert_en.md#installation)
  - [Evaluating a Checkpoint](docs/train_action_expert_en.md#evaluating-a-checkpoint)
  - [Model Server](docs/train_action_expert_en.md#model-server)
  - [SONIC Robot Client and MuJoCo Simulation (Chinese)](docs/sonic_simulation.md)
  - [SONIC Tabletop Grasp Simulation (Chinese)](docs/sonic_grasp.md)
  - [Frozen VLA Action-Head Simulation Diagnostics (Chinese)](docs/sonic_vla_diagnostics.md)
  - [Frozen VLA Target Selection, Execution Feedback and Repeated Validation (Chinese)](docs/sonic_vla_stable.md)
  - [WLA Model-Only Grasp Baseline and Official WBC Search (Chinese)](docs/sonic_wla_model_only.md)
  - [WLA TCP/Gripper Calibration and Six Model-Only Retests (Chinese)](docs/sonic_wla_calibrated.md)
  - [Fine-tuning the UnifoLM-WLA-1.0-Base Action Expert](docs/train_action_expert_en.md#fine-tuning-the-unifolm-wla-10-base-action-expert)
  - [Training from Scratch](docs/train_action_expert_en.md#training-from-scratch)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## 🔥 News
- Sep 28, 2026: 🚀 we released the [UnifoLM-WLA-1.0-Base](https://huggingface.co/unitreerobotics/UnifoLM-WLA-1.0-Base) and fine-tuning code.
- Sep 20, 2026: 🚀 we released the model modules and training action expert code
- Sep 11, 2026: 🚀 we released the model weights of [UnifoLM-ER-1](https://huggingface.co/unitreerobotics/UnifoLM-ER-1)
- Sep 11, 2026: 🚀 we released the model weights of [UnifoLM-ER-Flow](https://huggingface.co/unitreerobotics/UnifoLM-ER-Flow)

## 📑 Open-Source Plan

- **Code**
  - [x] [Code for training action experts based on UnifoLM-ER models](docs/train_action_expert_en.md#training-from-scratch)
  - [x] [Code for fine-tuning based on UnifoLM-WLA-1.0-Base](docs/train_action_expert_en.md#fine-tuning-the-unifolm-wla-10-base-action-expert)
  - [x] [Code for LoRA fine-tuning](docs/train_action_expert_en.md#lora-fine-tuning)
- **Models**
  - [x] [**UnifoLM-ER-1**](https://huggingface.co/unitreerobotics/UnifoLM-ER-1)
  - [x] [**UnifoLM-ER-Flow**](https://huggingface.co/unitreerobotics/UnifoLM-ER-Flow)
  - [x] [**UnifoLM-WLA-1.0-Base**](https://huggingface.co/unitreerobotics/UnifoLM-WLA-1.0-Base)
- **Datasets**
  - [x] [**UniBot-V1 Challenge Dataset**](https://huggingface.co/collections/unitreerobotics/unibot-v1-challenge-dataset)
  - [x] [**UnifoLM-WBT-Dataset**](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset)
  - [x] [**UnifoLM-Dex1-Dataset**](https://huggingface.co/collections/unitreerobotics/unifolm-g1-dex1-dataset)

## 📘 Technical Documentation

- [Robot Action, State, and Statistics Processing Specification](docs/robot_action_state_processing_en.md)
- [Train and Evaluate an Action Expert](docs/train_action_expert_en.md)

## Citation

```bibtex
@misc{unifolm-wla-1.0,
  author = {Unitree},
  title  = {UnifoLM-WLA-1.0: One Model Driven, Whole-Body Coordination},
  year   = {2026},
}
```

## Acknowledgements

This project is built upon and continues the work of
[starVLA](https://github.com/starVLA/starVLA) and
[Qwen-Image](https://github.com/QwenLM/Qwen-Image). We sincerely thank their
authors and contributors for making their work publicly available.

## License

Except where otherwise noted, this project is released under the
[Apache License 2.0](LICENSE). Third-party components remain subject to their
original licenses. See [NOTICE](NOTICE) and
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for attribution and details.
