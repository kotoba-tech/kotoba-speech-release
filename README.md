# Kotoba-Speech Version. 0.1
Welcome to the code repository for Kotoba-Speech v0.1, a 1.2B Transformer-based speech generative model designed for generating fluent Japanese speech. This model represents one of the most advanced open-source options available in the field.

Questions, feature requests, or bug reports? Join [our Discord community](https://discord.com/invite/qPVFqhGN7Z)!

<img src="assets/logo.png" width="300" height="300" alt="Kotoba-Speech Logo">

## About
Kotoba-Speech Version 0.1 distinguishes itself as an open-source solution for generating high-quality Japanese speech from text prompts, while also offering the capability for voice cloning through speech prompts.

- **_Demo:_** Experience Kotoba-Speech in action [here](https://huggingface.co/spaces/kotoba-tech/Kotoba-Speech).
- **_Model Checkpoint:_** Access our commercially usable pre-trained model [here](https://huggingface.co/kotoba-tech/kotoba-speech-v0.1).
- **_Open-sourced Code:_** This repository opensources the training and inference code, along with the Gradio demo code. We borrow code from [MetaVoice](https://github.com/metavoiceio/metavoice-src) as a starting point.

### Our Model vs. Leading TTS Providers for Japanese
https://github.com/kotoba-tech/kotoba-speech-release/assets/18011504/516f56f4-db92-45cb-b2e5-92863b36f8cd

### Fine-tuning Our Pre-trained Model for 関西弁
https://github.com/kotoba-tech/kotoba-speech-release/assets/18011504/0204938e-7bb2-4c9f-9c6b-cb1e5c2dcff4

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
pip install -r requirements.txt
pip install -U torch==2.2.0 torchaudio==2.2.0 xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu121
# you may see a dependency error about audiocraft requiring torch==2.1.0; this is safe to ignore, the model will run fine
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
python fam/llm/train.py --num_gpus 1 --batch_size 32 --per_gpu_batchsize 2 --max_epoch 5 --learning_rate 0.00005 --data_dir data --exp_name reazon_small_exp_finetuning --spkemb_dropout 0.1 --check_val_every_n_epoch 1 --wandb_entity YOUR_WANDB_ENTITY --wandb_project YOUR_WANDB_PROJECT --use_wandb

# Multi-GPU Fine-tuning (e.g., using 2 GPUs)
# Replace YOUR_WANDB_ENTITY and YOUR_WANDB_PROJECT
python fam/llm/train.py --num_gpus 2 --batch_size 32 --per_gpu_batchsize 2 --max_epoch 5 --learning_rate 0.00005 --data_dir data --exp_name reazon_small_exp_finetuning --spkemb_dropout 0.1 --check_val_every_n_epoch 1 --wandb_entity YOUR_WANDB_ENTITY --wandb_project YOUR_WANDB_PROJECT --use_wandb

# Fine-tuning (without WandB logging)
python fam/llm/train.py --num_gpus 1 --batch_size 32 --per_gpu_batchsize 2 --max_epoch 5 --learning_rate 0.00005 --data_dir data --exp_name reazon_small_exp_finetuning --spkemb_dropout 0.1 --check_val_every_n_epoch 1 

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
- See all [active issues](https://github.com/kotoba-tech/kotoba-speech-release/issues)!

## 5.2 Enhancements
- [ ] Write an explanation about multi-node training
- [ ] Integrade a gradio demo

## 5.3　Acknowledgements
We thank [MetaVoice](https://github.com/metavoiceio/metavoice-src) for releasing their code and their English pre-trained model.
