import argparse
from typing import Optional, Literal

def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained model.")
    
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size.")

    parser.add_argument("--compile", action='store_true',
                        help="Whether to compile the model using PyTorch 2.0.")

    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path to a checkpoint file to resume training from.")

    parser.add_argument("--data_dir", type=str, default="",
                        help="A path to the dataset dir.")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for sampling.")

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32", "tfloat32"],
                        help="Data type to use for sampling.")

    parser.add_argument("--exp_name", type=str, default="kotoba_voice_1.3B",
                        help="A path to the dataset dir.")

    parser.add_argument("--fast_dev_run", action='store_true', default=False,
                        help="Run a quick development check, usually used for debugging and testing purposes.")

    parser.add_argument("--huggingface_repo_id", type=str, required=False, default="kotoba-tech/kotoba-speech-v0.1",
                        help="Absolute path to the model directory.")

    parser.add_argument("--train_from_scratch", action='store_true', default=False, 
                        help="Run a quick development check, usually used for debugging and testing purposes.")

    parser.add_argument("--init_from", type=str, default="resume",
                        help="Either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl').")

    parser.add_argument("--max_epoch", type=int, default=5,
                        help="Number of nodes")

    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps to train")

    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs per node")

    parser.add_argument("--num_nodes", type=int, default=1,
                        help="Number of nodes")
    
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate from each model.")

    parser.add_argument("--output_dir", type=str, default="samples/",
                        help="Relative path to output directory")

    parser.add_argument("--per_gpu_batchsize", type=int, default=16,
                        help="Batch size per GPU.")

    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for sampling.")

    parser.add_argument("--use_kv_cache", type=str, default=None, choices=[None, "flash_decoding", "vanilla"],
                        help="Type of kv caching to use for inference.")

    parser.add_argument("--val_check_interval", type=float, default=None,
                    help="This overwrites check_val_every_n_epoch. If this value is less than 1, for example, 0.25, it means the validation set will be checked 4 times during a training epoch. If this value is greater than 1, for example, 1000, the validation set will be checked every 1000 training batches, either across complete epochs or during iteration-based training.")

    parser.add_argument("--check_val_every_n_epoch", type=int, default=1,
                        help="Validate every n epoch.")
    
    parser.add_argument('--fsdp_strategy', type=str, default=None, choices=[None, 'FULL_SHARD', 'SHARD_GRAD_OP', 'HYBRID_SHARD', 'NO_SHARD'],
                        help='Use fully sharded data parallel type,'
                        'FULL_SHARD: Shard weights, gradients, optimizer state (1 + 2 + 3)'
                        'SHARD_GRAD_OP: Shard gradients, optimizer state (2 + 3)'
                        'HYBRID_SHARD: Full-shard within a machine, replicate across machines'
                        "NO_SHARD: Don't shard anything (similar to DDP, slower than DDP)"
                        'None: Use DDP'
                        )
    
    parser.add_argument('--use_ddp_strategy', action='store_true', default=False,
                        help='use DDPStrategy()')
    
    # WandB settings   
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Enable integration with Weights & Biases for experiment tracking.')

    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity (team or user) under which the project is located (optional).")

    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Weights & Biases project name to which the run will be logged (optional).")

    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Unique identifier for the Weights & Biases run, allowing for run resumption or other operations (optional).")

    # Debug Setting
    parser.add_argument("--debug", action='store_true', default=False,
                    help="Debug mode")

    parser.add_argument("--debug_data_dir", type=str, default="",
                        help="A path to the dataset dir.")

    # Optimizer Setting
    parser.add_argument("--optimizer_type", type=str, default="AdamW",
                        choices=["Adam", "AdamW"], help="Type of optimizer to use.")

    parser.add_argument("--beta_one", type=float, default=0.9,
                        help="Coefficient for computing running averages of gradient.")

    parser.add_argument("--beta_two", type=float, default=0.95,
                        help="Coefficient for computing running averages of the square of the gradient.")

    parser.add_argument("--epsilon", type=float, default=1e-5,
                        help="Term added to the denominator to improve numerical stability.")

    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Coefficient for computing running averages of the square of the gradient.")

    parser.add_argument("--decay_power", type=str, default="cosine",
                        help="Type of learning rate decay to use.")

    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Initial learning rate.")

    parser.add_argument("--warmup_steps", type=float, default=0.01,
                        help="Fraction of total training steps to use for learning rate warmup. If warmup_steps is greater than 1, then the value specified in warmup_steps will represent the exact number of steps to be used for the warmup phase.")

    parser.add_argument("--end_lr", type=float, default=0,
                        help="Final learning rate after decay.")

    parser.add_argument("--gradient_clip_val", type=float, default=0,
                        help="Clip gradients' maximum magnitude.")

    # Model config setting
    parser.add_argument("--spkemb_dropout", type=float, default=0.1,
                        help="Fraction of total training steps to use for learning rate warmup. If warmup_steps is greater than 1, then the value specified in warmup_steps will represent the exact number of steps to be used for the warmup phase.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Now you can use args to access your configuration, for example:
    # print(args.huggingface_repo_id)
