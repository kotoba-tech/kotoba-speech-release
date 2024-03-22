import os, torch, json
from tqdm import tqdm
import argparse
import julius
from audiocraft.data.audio import audio_read
from audiocraft.models import MultiBandDiffusion  # type: ignore
from utils import get_sorted_keys, get_ids, create_directory_recursive

mbd_sample_rate =  24_000
mbd_bandwidth = 6
num_codebooks = 8
mbd = MultiBandDiffusion.get_mbd_24khz(bw=mbd_bandwidth)

def tokenize(audio_path, output_file):
    wav, sr = audio_read(audio_path)
    if sr != mbd_sample_rate:
        wav = julius.resample_frac(wav, sr, mbd_sample_rate)
    if wav.ndim == 2:
        wav = wav.unsqueeze(1)
    wav = wav.to("cuda")
    with torch.no_grad():
        tokens = mbd.codec_model.encode(wav)
        tokens = tokens[0][0].cpu()
        torch.save(tokens, output_file)

def main(file_name, base_dir, num_shards, shard_id):
    files = get_sorted_keys(file_name)
    start, end = get_ids(len(files), num_shards, shard_id)
    print(start, end)
    for file_name in tqdm(files[start:end]):
        in_file = os.path.join(base_dir, "wav", file_name + ".wav")
        out_file = os.path.join(base_dir, "tok", file_name + ".tok")
        if os.path.exists(out_file):
            continue
        create_directory_recursive(out_file)
        tokenize(in_file, out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audito Tokenize")
    parser.add_argument("--in_file", default="data/small.jsonl", help="Name of the file")
    parser.add_argument("--base_dir", type=str, default="data/", help="base_dir")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID")
    args = parser.parse_args()
    main(args.in_file, args.base_dir, args.num_shards, args.shard_id)
