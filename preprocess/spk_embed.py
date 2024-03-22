import os, torch, json
from tqdm import tqdm
import argparse
from fam.quantiser.audio.speaker_encoder.model import SpeakerEncoder
from utils import get_sorted_keys, get_ids, create_directory_recursive

smodel = SpeakerEncoder(device="cuda", eval=True, verbose=False)

def speaker_embed(audio_path, output_file):
    with torch.no_grad():
        embedding = smodel.embed_utterance_from_file(audio_path, numpy=False)
        embedding = embedding.cpu().detach()
        torch.save(embedding, output_file)
        return embedding


def main(file_name, base_dir, num_shards, shard_id):
    files = get_sorted_keys(file_name)
    start, end = get_ids(len(files), num_shards, shard_id)
    print(start, end)
    for file_name in tqdm(files[start:end]):
        in_file = os.path.join(base_dir, "wav", file_name + ".wav")
        out_file = os.path.join(base_dir, "emb", file_name + ".emb")
        if os.path.exists(out_file):
            continue
        create_directory_recursive(out_file)
        speaker_embed(in_file, out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Speaker Embeddings")
    parser.add_argument("--in_file", default="data/small.jsonl", help="Name of the file")
    parser.add_argument("--base_dir", type=str, default="data/", help="base_dir")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID")
    args = parser.parse_args()
    main(args.in_file, args.base_dir, args.num_shards, args.shard_id)
