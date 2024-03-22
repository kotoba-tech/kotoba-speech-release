# Kotoba-Speech Version. 0.1
Welcome to the code repository for Kotoba-Speech v0.1, a 1.2B Transformer-based speech generative model designed for generating fluent Japanese speech. This model represents one of the most advanced open-source options available in the field.

Questions, feature requests, or bug reports? Join [our community](https://discord.com/invite/qPVFqhGN7Z)!

<img src="assets/logo.png" width="300" height="300" alt="Kotoba-Speech Logo">

## About
Kotoba-Speech Version 0.1 distinguishes itself as an open-source solution for generating high-quality Japanese speech from text prompts, while also offering the capability for voice cloning through speech prompts.

| Sentence                | Amazon Poly      | Google Text-to-Speech    | Kotoba-Speech v0.1 | Kotoba-Speech v0.1 (Voice-Cloning) |
|------------------------|-----------|-----------|-----------|-----------|
| "コトバテクノロジーズのミッションは音声基盤モデルを作る事です。"       | [Download](assets/aws.wav)       | [Download](assets/google.wav)          | [Download](assets/kotoba.wav)            | [Download](assets/kotoba_cloning.wav)                |

- **Demo:** Experience Kotoba-Speech in action [here](https://huggingface.co/kotoba-tech/kotoba-speech-v0.1).
- **Model Checkpoint:** Access our pre-trained model [here](https://huggingface.co/kotoba-tech/kotoba-speech-v0.1).　The model checkpoint is commercially usable.
- **Open-sourced Code:** This repository opensources the training and inference code, along with the Gradio demo code. We borrow code from [MetaVoice](https://github.com/metavoiceio/metavoice-src) as a starting point.

## Table of Contents

1. **Installation**
2. **Preparing Datasets**
3. **Training** 
4. **Inference**
5. **Other Notes**

## 1. Installation  
```bash
# Installing ffmpeg
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz.md5
md5sum -c ffmpeg-git-amd64-static.tar.xz.md5
tar xvf ffmpeg-git-amd64-static.tar.xz
sudo mv ffmpeg-git-*-static/ffprobe ffmpeg-git-*-static/ffmpeg /usr/local/bin/
rm -rf ffmpeg-git-*

# Setting-up Python virtual environment
python -m venv myenv
source myenv/bin/activate
pip install -U --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -r requirements.txt
pip install flash-attn==2.5.3
pip install -e .
```

## 2. Preparing Datasets
We provide an example of preparing datasets to train our model. We use Reazon Speech, the largest open-sourced Japanese speech dataset, as an example. (Note that our model is not necessary trained on Reazon Speech solely.)
```bash
# Download & Format Data
python preprocess/download_reazon.py

# Pre-calculate Speaker Embeddings
python preprocess/spk_embed.py

# Tokenize Audio
python preprocess/audio_tokenize.py

# Tokenize Text Captions
python preprocess/text_tokenize.py

# Split data into (training/validation/test)
python preprocess/split.py
```

## 3. Training
```bash
# Fine-tuning from our pre-trained checkpoint
# Replace YOUR_WANDB_ENTITY and YOUR_WANDB_PROJECT
CUDA_VISIBLE_DEVICES=1 python fam/llm/train.py --num_gpus 1 --batch_size 32 --per_gpu_batchsize 2 --max_epoch 5 --learning_rate 0.00005 --data_dir data --exp_name reazon_small_exp_finetuning --spkemb_dropout 0.1 --check_val_every_n_epoch 1 --wandb_entity YOUR_WANDB_ENTITY --wandb_project YOUR_WANDB_PROJECT --use_wandb

# Multi-GPU Fine-tuning (e.g., using 2 GPUs)
# Replace YOUR_WANDB_ENTITY and YOUR_WANDB_PROJECT
python fam/llm/train.py --num_gpus 2 --batch_size 32 --per_gpu_batchsize 2 --max_epoch 5 --learning_rate 0.00005 --data_dir data --exp_name reazon_small_exp_finetuning --spkemb_dropout 0.1 --check_val_every_n_epoch 1 --wandb_entity YOUR_WANDB_ENTITY --wandb_project YOUR_WANDB_PROJECT --use_wandb

# Fine-tuning (without WandB logging)
CUDA_VISIBLE_DEVICES=1 python fam/llm/train.py --num_gpus 1 --batch_size 32 --per_gpu_batchsize 2 --max_epoch 5 --learning_rate 0.00005 --data_dir data --exp_name reazon_small_exp_finetuning --spkemb_dropout 0.1 --check_val_every_n_epoch 1 

# Training from scratch
# Replace YOUR_WANDB_ENTITY and YOUR_WANDB_PROJECT
python fam/llm/train.py --num_gpus 1 --batch_size 64 --per_gpu_batchsize 2 --max_epoch 20 --learning_rate 0.0001 --data_dir data --exp_name reazon_small_exp --spkemb_dropout 0.1 --check_val_every_n_epoch 1 --wandb_entity YOUR_WANDB_ENTITY --wandb_project YOUR_WANDB_PROJECT --use_wandb --train_from_scratch
```

## 4. Inference
```bash
# Our Pre-trained Checkpoint
python -i fam/llm/fast_inference.py  --model_name kotoba-tech/kotoba-speech-v0.1
tts.synthesise(text="コトバテクノロジーズのミッションは音声基盤モデルを作る事です。", spk_ref_path="assets/bria.mp3")

# Inference from Our Pre-trained Checkpoint (関西弁)
python -i fam/llm/fast_inference.py  --model_name kotoba-tech/kotoba-speech-v0.1-kansai  
tts.synthesise(text="コトバテクノロジーズのミッションは音声基盤モデルを作る事です。", spk_ref_path="assets/bria.mp3")

# Inference from Your Own Pre-trained Checkpoint
# YOUR_CHECKPOINT_PATH is something like /home/checkpoints/epoch=0-step=1810.ckpt
python -i fam/llm/fast_inference.py  --first_model_path YOUR_CHECKPOINT_PATH
tts.synthesise(text="コトバテクノロジーズのミッションは音声基盤モデルを作る事です。", spk_ref_path="assets/bria.mp3")
```

## 5. Other notes
## 5.1 Contribute
- See all [active issues](https://github.com/kotoba-tech/kotoba-voice/issues)!

## 5.2 Enhancements
- [ ] Write an explanation about multi-node training
- [ ] Integrade a gradio demo

## 5.3　Acknowledgements
We thank [MetaVoice](https://github.com/metavoiceio/metavoice-src) for releasing their code and their English pre-trained model.
