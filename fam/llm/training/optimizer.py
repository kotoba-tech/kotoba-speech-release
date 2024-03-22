import torch
import inspect
from IPython import embed
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


def set_schedule(pl_module):
    optimizer_type = pl_module.configs["optimizer_type"]
    lr = pl_module.configs["learning_rate"]
    beta_one = pl_module.configs["beta_one"]
    beta_two = pl_module.configs["beta_two"]
    eps = pl_module.configs["epsilon"]
    wd = pl_module.configs["weight_decay"]
    #fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    #use_fused = fused_available and pl_module.device.type == "cuda"
    use_fused=False
    extra_args = dict(fused=True) if use_fused else dict()

    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(pl_module.first_stage_model_cls.model.parameters(), lr=lr, betas=(beta_one, beta_two), eps=eps, **extra_args)
    elif optimizer_type == "AdamW":
        model = pl_module.first_stage_model_cls.model
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": wd},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(beta_one, beta_two), eps=eps, **extra_args)

    if pl_module.trainer.max_steps == -1:
        # TODO: this is note tested in multi-node set-up.
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            // len(pl_module.trainer.device_ids)
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    end_lr = pl_module.configs["end_lr"]
    decay_power = pl_module.configs["decay_power"]
    warmup_steps = pl_module.configs["warmup_steps"]
    if warmup_steps <= 1:
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )