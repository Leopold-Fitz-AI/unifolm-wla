import json
from pathlib import Path

def build_dataloader(cfg, dataset_py="single_source_datasets"):
    """Single entry point for building the VLA training dataloader."""
    # Inference imports the action/state geometry modules through this package.
    # Load dataset readers only when actually building a training dataloader.
    from .multi_source_dataset import create_training_dataloader
    import torch.distributed as dist
    from accelerate.logging import get_logger

    logger = get_logger(__name__)
    if dataset_py not in ("multi_source_datasets", "single_source_datasets"):
        raise ValueError(
            f"Unsupported dataset_py={dataset_py!r}; expected "
            "'single_source_datasets' or 'multi_source_datasets'."
        )

    msd_cfg = cfg.datasets.vla_data
    vla_train_dataloader, dataset = create_training_dataloader(
        config_path=msd_cfg.data_config_path,
        batch_size=msd_cfg.per_device_batch_size,
        num_workers=getattr(msd_cfg, "num_workers", 4),
        distributed=dist.is_initialized(),
        num_replicas=dist.get_world_size() if dist.is_initialized() else None,
        rank=dist.get_rank() if dist.is_initialized() else None,
    )

    if not dist.is_initialized() or dist.get_rank() == 0:
        output_dir = Path(cfg.output_dir)
        source_datasets = getattr(dataset, "datasets", (dataset,))
        stats = {
            source_dataset.config.name: {
                "action": {
                    "offset": source_dataset._action_norm_offset.tolist(),
                    "scale": source_dataset._action_norm_scale.tolist(),
                },
                "state": {
                    "offset": source_dataset._state_norm_offset.tolist(),
                    "scale": source_dataset._state_norm_scale.tolist(),
                },
            }
            for source_dataset in source_datasets
        }
        with open(output_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved dataset statistics at {output_dir / 'dataset_statistics.json'}")

    return vla_train_dataloader
