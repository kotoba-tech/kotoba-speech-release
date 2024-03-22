import os, subprocess, json
import pandas as pd
import argparse
from utils import get_dirs, get_sorted_keys, get_ids, create_directory_recursive

def download_tsv(split, base_dir):
    command = f"wget https://reazonspeech.s3.abci.ai/{split}.tsv -P {base_dir}"
    subprocess.run(command, shell=True)
    file_path = os.path.join(base_dir, f"{split}.tsv")
    return file_path

def download_file(file_name, base_dir):
    base_url = "https://reazonspeech.s3.abci.ai/data/"
    path = os.path.join(base_url, file_name + ".tar")
    command = f"wget {path} -P {base_dir}"
    subprocess.run(command, shell=True)
    path = os.path.join(base_dir, file_name + ".tar")
    command = f"tar xvf {path} -C {base_dir}"
    subprocess.run(command, shell=True)
    command = f"rm {path}"
    subprocess.run(command, shell=True)

def flac2wav(file_name, base_dir):
    flac_name = os.path.join(base_dir, file_name + ".flac")
    wav_name = os.path.join(base_dir, 'wav', file_name + '.wav')
    create_directory_recursive(wav_name)
    subprocess.run(f"ffmpeg -y -i {flac_name} {wav_name}", shell=True)

def download_files(in_file, base_dir, num_shards, shard_id):
    files = get_dirs(in_file)
    start, end = get_ids(len(files), num_shards, shard_id)
    for file_name in files[start:end]:
        path = os.path.join(base_dir, file_name)
        if not os.path.exists(path):
            download_file(file_name, base_dir)

def flac2wav_files(in_file, base_dir, num_shards, shard_id):
    files = get_sorted_keys(in_file)
    start, end = get_ids(len(files), num_shards, shard_id)
    for file_name in files[start:end]:
        path = os.path.join(base_dir, file_name)
        path = os.path.join(base_dir, 'wav', file_name + '.wav')
        if not os.path.exists(path):
            flac2wav(file_name, base_dir)

def tsv2jsonl(in_file, out_file):
    data = pd.read_csv(in_file, header=None, sep='\t')
    with open(out_file, "w") as f:
        for index, row in data.iterrows():
            json.dump({"key": row[0].replace(".flac", ""), "text": row[1]}, f, ensure_ascii=False)
            f.write('\n')

def main(split, base_dir, num_shards, shard_id):
    create_directory_recursive(base_dir)
    in_file = download_tsv(split, base_dir)
    out_file = in_file.replace(".tsv", ".jsonl")
    tsv2jsonl(in_file, out_file)
    download_files(out_file, base_dir, num_shards, shard_id)
    flac2wav_files(out_file, base_dir, num_shards, shard_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Reazon Dataset")
    parser.add_argument("--split", default="small", help="Reazon split")
    parser.add_argument("--base_dir", default="data/", help="path to data directory")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID")
    args = parser.parse_args()
    main(args.split, args.base_dir, args.num_shards, args.shard_id)
