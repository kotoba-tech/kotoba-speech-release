import os, torch, argparse, torch
from tqdm import tqdm
from fam.quantiser.text.tokenise import TrainedBPETokeniser
from utils import get_keys_texts, get_ids, create_directory_recursive
from huggingface_hub import snapshot_download
from fam.llm.training import get_first_stage_path

model_dir = snapshot_download(repo_id="metavoiceio/metavoice-1B-v0.1")
checkpoint = torch.load(get_first_stage_path(model_dir))
tokenizer = TrainedBPETokeniser(**checkpoint["meta"]["tokenizer"])

def text_tokenize(text, output_file):
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.int64)
    torch.save(tokens, output_file)

def main(file_name, num_shards, shard_id, base_dir):
    keys_text = get_keys_texts(file_name)
    start, end = get_ids(len(keys_text), num_shards, shard_id)
    print(start, end)
    for keys_texts in tqdm(keys_text[start:end]):
        file_name = keys_texts["key"]
        text = keys_texts["text"]
        out_file = os.path.join(base_dir, "txt", file_name + ".pt")
        create_directory_recursive(out_file)
        if os.path.exists(out_file):
            continue
        text_tokenize(text, out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize Text")
    args = parser.parse_args()
    parser.add_argument("--in_file", default="data/small.jsonl", help="Name of the file")
    parser.add_argument("--base_dir", type=str, default="data/", help="base_dir")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID")
    args = parser.parse_args()
    main(args.in_file, args.num_shards, args.shard_id, args.base_dir)
