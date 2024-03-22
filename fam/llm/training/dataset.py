# python fam/llm/training/dataset.py
import os, json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class VoiceDataset(Dataset):
    def __init__(self, split="train", data_dir=""):
        self.data_dir = data_dir
        self.data = self._load_data(os.path.join(data_dir, f'{split}.jsonl'))

    def _load_data(self, file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        train_item = self.data[idx]
        key = train_item["key"]
        tokens = torch.load(os.path.join(self.data_dir, 'txt', f"{key}.pt"))
        embedding = torch.load(os.path.join(self.data_dir, 'emb', f"{key}.emb"))
        audio_tokens = torch.load(os.path.join(self.data_dir, 'tok', f"{key}.tok"))
        return tokens, embedding, audio_tokens

    def collate(self, batch):
        audio_pad = 2048
        num_bands = 2
        eoa = 2048
        loss_pad = -1
        audio_vocab_size = 1024
        # end of audio
        # batch is a list of samples, where each sample is a dictionary
        tokens = [sample[0] for sample in batch]
        padded_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
        embeddings = [sample[1] for sample in batch]
        padded_embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value=0)
        audios = [sample[2] for sample in batch]
        # Determine the maximum value of h
        max_l = max(tensor.size(1) for tensor in audios)
        max_t2l = max(tensor_text.size(0) + tensor_audio.size(1)*2 for tensor_text, tensor_audio in zip(tokens, audios))
        # Pad tensors to have the same width
        padded_audios = [torch.nn.functional.pad(tensor, (0, max_l - tensor.shape[1]), mode='constant', value=audio_vocab_size) for tensor in audios]
        # Concatenate tensors along dimension 0
        padded_audios = torch.stack(padded_audios, dim=0)
        # first_stage_output [B, 2, T+T+L (padded)]
        first_stage_input = torch.full([len(batch), max_t2l], audio_pad, dtype=torch.int64)
        first_stage_output = torch.full([len(batch), max_t2l], loss_pad, dtype=torch.int64)
        ## first_stage_output: [text, text, audio, pad]
        ## first_stage_input: [-1, audio, eoa, pad]
        for idx, tensor_text, tensor_audio in zip(range(len(batch)), tokens, audios):
            text_size = tensor_text.size(0)
            audio_size = tensor_audio.size(1)
            bands = tensor_audio[:num_bands]
            bands += torch.arange(num_bands).view([num_bands, 1])*audio_vocab_size
            bands = bands.transpose(0, 1).contiguous().view(-1)
            first_stage_input[idx, :text_size] = tensor_text
            first_stage_input[idx, text_size:text_size+audio_size*num_bands] = bands
            first_stage_output[idx, :text_size-1] = torch.full([text_size-1], loss_pad, dtype=torch.int64)
            eoa_tensor = torch.full([1], eoa, dtype=torch.int64)
            first_stage_output[idx, text_size-1:text_size+audio_size*2] = torch.cat([bands, eoa_tensor], dim=0)
            first_stage_output[idx, text_size+audio_size*2:] = loss_pad
        return padded_tokens, padded_embeddings, padded_audios, first_stage_input, first_stage_output


if __name__ == "__main__":
    for sp in ["train" , "val"]:
        voice_train = VoiceDataset(split=sp, data_dir="/debug_data")
        for i in tqdm(range(len(voice_train))):
            tokens, embedding, audio_tokens = voice_train.__getitem__(i)
        

        
