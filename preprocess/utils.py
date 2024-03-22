import os, json


def create_directory(dir_path):
    """
    Recursively create directories if they do not exist.
    
    Args:
    directory_path (str): The directory path to create.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def list_parent_directories(path):
    """
    List all parent directories of a given path.

    Args:
    path (str): The directory path to find parent directories for.

    Returns:
    list: A list of parent directories.
    """
    parent_directories = []
    while True:
        path = os.path.dirname(path)
        if path == '/' or not path:
            break
        parent_directories.append(path)
    return parent_directories

def create_directory_recursive(path):
    dirs = list_parent_directories(path)
    dirs.reverse()
    for dir in dirs:
        create_directory(dir)

def get_sorted_keys(in_file, key="key"):
    files = []
    with open(in_file) as fin:
        for line in fin:
            line = json.loads(line)
            files.append(line[key])
    return files

def get_keys_texts(in_file, key="key"):
    data = []
    with open(in_file) as fin:
        for line in fin:
            line = json.loads(line)
            data.append(line)
    return data

def get_ids(num_keys, num_shards, shard_id):
    shard_size = num_keys//num_shards
    remainder = num_keys - shard_size*num_shards
    start_id = shard_size*shard_id + min([shard_id, remainder])
    end_id = shard_size*(shard_id+1) + min([shard_id+1, remainder])
    return start_id, end_id

def get_dirs(in_file):
    files = set()
    with open(in_file) as fin:
        for line in fin:
            line = json.loads(line)
            files.add(line["key"].split("/")[0])
    files = sorted(files)
    return files

def split_json_file(json_file_path, output_dir):
    with open(json_file_path, 'r') as f:
        # Read and modify lines
        lines = f.readlines()

    N = len(lines)
    train_data = lines[:N-1000]
    val_data = lines[N-1000:N-500]
    test_data = lines[N-500:]

    # Write to train.jsonl
    with open(os.path.join(output_dir, 'train.jsonl'), 'w', encoding='utf-8') as f:
        f.writelines(train_data)

    # Write to val.jsonl
    with open(os.path.join(output_dir, 'val.jsonl'), 'w') as f:
        f.writelines(val_data)

    # Write to test.jsonl
    with open(os.path.join(output_dir, 'test.jsonl'), 'w') as f:
        f.writelines(test_data)
