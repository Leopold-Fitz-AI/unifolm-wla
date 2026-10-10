export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenMMDiT
freeze_module_list=''
base_vlm=/path/to/UnifoLM-ER-1
config_yaml=./unifolm_wla/config/training/unifolm_wla_pretrain_multisource.yaml
data_config_path=./unifolm_wla/dataloader/multi_source_dataset/configs/unitree.yaml
run_root_dir=./playground/Checkpoints
run_id=debug_mmdit
# === End of environment variable configuration ===
###########################################################################################
export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

.venv/bin/accelerate launch \
  --config_file unifolm_wla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  unifolm_wla/training/train_unifolm_wla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.robot_state_projector.enabled True \
  --datasets.vla_data.data_config_path ${data_config_path} \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.num_workers 4 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 200000 \
  --trainer.num_warmup_steps 2000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 500 \
  --trainer.gradient_accumulation_steps 4 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}

# NOTE (removed flags): this script previously also passed
#     --framework.omnivggt_fusion.enabled / .fusion_dim
#     --framework.action_tokenizer.enabled / .ce_loss_weight
#     --framework.action_model.rtc / .max_delay
# All six are read nowhere in this repository (`OmegaConf.merge` accepts unknown
# keys silently), so they only created the illusion that those features were on.
# Re-add them together with the corresponding implementation, not before.
#
# NOTE (state injection): the `--framework.robot_state_projector.enabled True`
# flag in the launch block above is NOT cosmetic. `QwenMMDiT.__init__` forces
# `action_model.state_dim = 0` whenever the projector is enabled, so this script
# trains the VLM-side projector path (robot state is projected into the
# `<|robot_state|>` token) and NOT the DiT-side state-token path that
# `unifolm_wla_pretrain_multisource.yaml` implies with its `state_dim: 60`.
# Launch the YAML without that flag to get the state-token variant instead --
# the two produce incompatible checkpoints.
