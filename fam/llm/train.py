import os
import json
import torch
import wandb
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import FSDPStrategy, DDPStrategy
from pytorch_lightning.plugins.environments import ClusterEnvironment
from huggingface_hub import snapshot_download

# Specific to your project's structure
from fam.llm.training import parse_args, get_first_stage_path, get_second_stage_path, TrainingConfig, VoiceDataModule, WandbLogger, dist_utils, optimizer_utils, Evaluator
from fam.llm.decoders import Decoder, EncodecDecoder
from fam.quantiser.text.tokenise import TrainedBPETokeniser
from fam.llm.adapters import FlattenedInterleavedEncodec2Codebook
from fam.llm.sample import Model

class MyClusterEnvironment(ClusterEnvironment):
    @property
    def creates_processes_externally(self) -> bool:
        """Return True if the cluster is managed (you don't launch processes yourself)"""
        return True

    def world_size(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])

    def global_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])

    def local_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

    def node_rank(self) -> int:
        return int(os.environ["OMPI_COMM_WORLD_NODE_RANK"])

    main_address = os.getenv("MASTER_ADDR","")

    main_port = int(os.getenv("MASTER_PORT", "0"))
    
    def set_global_rank(self, rank):
        self.global_rank_ = rank

    def set_world_size(self, size):
        self.world_size_ = size

    def detect(self):
        return True

class KotobaSpeechModelFirstStage(pl.LightningModule):
    """
    A PyTorch Lightning module for the first stage of KotobaVoice model training.
    """
    
    def __init__(self, config_first_stage, config_second_stage, device, use_kv_cache, logger, is_debug=False, configs=None):
        super().__init__()
        self.configs = vars(configs)
        self.config_first_stage = config_first_stage
        self.use_kv_cache = use_kv_cache
        self.prev_step = -1
        self.evaluator = Evaluator()
        if dist_utils.is_main_process():
            self.wandb_logger = logger

    def configure_model(self):
        """
        Configures the model and its components.
        """
        self.data_adapter = FlattenedInterleavedEncodec2Codebook(end_of_audio_token=1024)
        self.first_stage_model_cls = Model(
            self.config_first_stage,
            TrainedBPETokeniser,
            EncodecDecoder,
            data_adapter_fn=self.data_adapter.decode,
            use_kv_cache=self.use_kv_cache,
        )
        self.first_stage_model_transformer = self.first_stage_model_cls.model

    def forward(self, text_tokens, embedding, inputs, targets):
        truncated_inputs = inputs[:,:2048] 
        truncated_targets = targets[:,:2048]
        truncated_inputs = truncated_inputs.unsqueeze(1)
        truncated_targets = truncated_targets.unsqueeze(1)
        list_logits, first_stage_loss = self.first_stage_model_transformer(idx=truncated_inputs, embedding=embedding, targets=truncated_targets, loss_reduce="mean")
        return first_stage_loss

    def training_step(self, batch, batch_idx):
        tokens, embedding, audio_tokens, first_stage_input, first_stage_output = batch 
        embedding = embedding.unsqueeze(1)
        loss = self(text_tokens=tokens, embedding=embedding, inputs=first_stage_input, targets=first_stage_output)

        if dist_utils.is_main_process():
            if self.prev_step == self.global_step:
                pass
            else:
                stats = {
                    "loss": loss.item(),
                }
                if self.wandb_logger is not None:
                    self.wandb_logger.log(results=stats, split="training", step=self.global_step, commit=False)
                    lr_stats = {f"lr_group{i}": list(self.optimizers().param_groups)[i]["lr"] for i in range(len(self.optimizers().param_groups))}
                    self.wandb_logger.log(results=lr_stats, split="lr", step=self.global_step, commit=True)
                    self.prev_step = self.global_step

        return loss
        
    def on_train_start(self): 
        if dist_utils.is_main_process():
            if self.wandb_logger is not None:
                wandb.watch(self.first_stage_model_transformer, log='parameters', log_freq=1000)
            
    def validation_step(self, batch, batch_idx):
        tokens, embedding, audio_tokens, first_stage_input, first_stage_output = batch
        tokens = tokens.unsqueeze(1)
        embedding = embedding.unsqueeze(1)
        loss = self(text_tokens=tokens, embedding=embedding, inputs=first_stage_input, targets=first_stage_output)
        stats = {
            "loss": [loss.item()],
        }
        self.evaluator.update(stats)
        return loss

    def on_validation_epoch_end(self):
        self.evaluator.synchronize_between_processes()
        if dist_utils.is_main_process():
            summary = self.evaluator.summarize()
        self.evaluator.reset()
        if dist_utils.is_main_process():
            if self.wandb_logger is not None:
                self.wandb_logger.log(results=summary, split="validation", step=self.global_step, commit=False)

    def configure_optimizers(self):
        return optimizer_utils.set_schedule(self)

# Train the model
if __name__ == "__main__":
    training_args = parse_args()
    data_dir =  training_args.debug_data_dir if training_args.debug else training_args.data_dir
    data_module = VoiceDataModule(training_args.per_gpu_batchsize, data_dir)

    model_dir = snapshot_download(repo_id=training_args.huggingface_repo_id)
    first_stage_ckpt_path = get_first_stage_path(model_dir)
    second_stage_ckpt_path = get_second_stage_path(model_dir)
    config_first_stage = TrainingConfig(
        ckpt_path=first_stage_ckpt_path,
        num_samples=training_args.num_samples,
        seed=training_args.seed,
        device=training_args.device,
        dtype=training_args.dtype,
        compile=training_args.compile,
        init_from=training_args.init_from,
        train_from_scratch=training_args.train_from_scratch,
        output_dir=training_args.output_dir,
        spkemb_dropout=training_args.spkemb_dropout
    )

    config_second_stage = TrainingConfig(
        ckpt_path=second_stage_ckpt_path,
        num_samples=training_args.num_samples,
        seed=training_args.seed,
        device=training_args.device,
        dtype=training_args.dtype,
        compile=training_args.compile,
        init_from=training_args.init_from,
        train_from_scratch=training_args.train_from_scratch,
        output_dir=training_args.output_dir,
        spkemb_dropout=training_args.spkemb_dropout
    )

    is_debug = training_args.debug
    logger = WandbLogger(training_args) if training_args.use_wandb else None
    model = KotobaSpeechModelFirstStage(config_first_stage, config_second_stage, device=training_args.device, use_kv_cache=training_args.use_kv_cache, logger=logger, is_debug=is_debug, configs=training_args)

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        verbose=True,
        monitor=None,
    )
    callbacks = [checkpoint_callback]

    num_gpus = (
        training_args.num_gpus
        if isinstance(training_args.num_gpus, int)
        else len(training_args.num_gpus)
    )

    grad_steps = training_args.batch_size // (
        training_args.per_gpu_batchsize * num_gpus * training_args.num_nodes
    )

    max_steps = training_args.max_steps if training_args.max_steps is not None else None
    
    if training_args.dtype == "bfloat16":
        precision="bf16-mixed"
    else:
        raise "Precision needs to be studied well."
    
    if training_args.fsdp_strategy is not None: 
        strategy = FSDPStrategy(
            sharding_strategy=training_args.fsdp_strategy
        )
    elif training_args.use_ddp_strategy: 
        strategy = DDPStrategy(
            static_graph=True,
        )
    else:
        strategy = 'ddp_find_unused_parameters_true'
    
    if training_args.num_nodes > 1:
        plugins = [MyClusterEnvironment()]
    else:
        plugins = None
    
    if training_args.val_check_interval is not None:
        if training_args.val_check_interval >= 1:
            training_args.val_check_interval=int(training_args.val_check_interval)


    trainer = Trainer(
        callbacks=callbacks,
        devices=training_args.num_gpus,
        strategy=strategy,
        num_nodes=training_args.num_nodes,
        precision=precision,
        accelerator="cuda",
        benchmark=True,
        deterministic=True,
        max_epochs=training_args.max_epoch if max_steps is None else 1000,
        accumulate_grad_batches=grad_steps,
        val_check_interval=training_args.val_check_interval,
        check_val_every_n_epoch=training_args.check_val_every_n_epoch,
        log_every_n_steps=10,
        fast_dev_run=training_args.fast_dev_run,
        plugins=plugins,
        gradient_clip_val=training_args.gradient_clip_val, 
    )
    
    trainer.fit(
        model,        
        ckpt_path=training_args.ckpt_path,
        datamodule=data_module
    )
    trainer.print(torch.cuda.memory_summary())

    
