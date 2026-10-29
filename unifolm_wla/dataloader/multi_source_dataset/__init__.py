def create_training_dataloader(*args, **kwargs):
    """Load training-only dependencies on demand, preserving the public entry point."""
    from .dataloader import create_training_dataloader as create

    return create(*args, **kwargs)

__all__ = ["create_training_dataloader"]
