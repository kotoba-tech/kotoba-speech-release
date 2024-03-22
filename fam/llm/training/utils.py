import os
from dataclasses import dataclass

def get_first_stage_path(model_dir: str):
    """Absolute path to checkpoint for the first stage model."""
    return os.path.join(os.path.expanduser(model_dir), "first_stage.pt")


def get_second_stage_path(model_dir: str):
    """Absolute path to checkpoint for the second stage model."""
    return os.path.join(os.path.expanduser(model_dir), "second_stage.pt")


@dataclass
class TrainingConfig:
    ckpt_path: str  # path to checkpoint
    output_dir: str
    num_samples: int = 10  # number of samples to draw
    seed: int = 1337  # random seed
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False
    init_from: str = "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    train_from_scratch: bool = False  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
    spkemb_dropout: float = 0.0 

    def __str__(self):
        field_strs = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            field_strs.append(f"  {field.name}: {value}")

        return "TrainingConfig:\n" + "\n".join(field_strs)
