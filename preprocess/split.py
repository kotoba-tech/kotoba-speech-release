import os, subprocess
import json
import argparse


def split_json_file(json_file_path, output_dir, val_size=500):
    with open(json_file_path, 'r') as f:
        # Read and modify lines
        lines = f.readlines()

    N = len(lines)
    train_data = lines[:N-val_size]
    val_data = lines[N-val_size*2:N-val_size]
    test_data = lines[N-val_size:]

    # Write to train.jsonl
    with open(os.path.join(output_dir, 'train.jsonl'), 'w', encoding='utf-8') as f:
        f.writelines(train_data)

    # Write to val.jsonl
    with open(os.path.join(output_dir, 'val.jsonl'), 'w', encoding='utf-8') as f:
        f.writelines(val_data)

    # Write to test.jsonl
    with open(os.path.join(output_dir, 'test.jsonl'), 'w', encoding='utf-8') as f:
        f.writelines(test_data)



# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split list into chunks")
    parser.add_argument("--in_file", default="data/small.jsonl", help="Name of the file")
    parser.add_argument("--base_dir", type=str, default="data/", help="base_dir")
    args = parser.parse_args()
    split_json_file(args.in_file, args.base_dir)
