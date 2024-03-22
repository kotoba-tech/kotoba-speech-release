# python fam/llm/training/datamodule.py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from fam.llm.training.dataset import VoiceDataset

class VoiceDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, data_dir="/root/data/reazon_small/"):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

    def prepare_data(self):
        pass

    def setup(self, stage):
        # Load and split data
        if stage == "train" or stage == "fit":
            self.voice_train = VoiceDataset(split="train", data_dir=self.data_dir)
            self.voice_val = VoiceDataset(split="val", data_dir=self.data_dir)
            self.voice_test = None
        elif stage == "test":
            self.voice_train = None
            self.voice_val = None
            self.voice_test = VoiceDataset(split="test", data_dir=self.data_dir)
        else:
            assert False

    def train_dataloader(self):
        return DataLoader(self.voice_train, batch_size=self.batch_size, collate_fn=self.voice_train.collate, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.voice_val, batch_size=self.batch_size, collate_fn=self.voice_val.collate)
  
    def test_dataloader(self):
        return DataLoader(self.voice_test, batch_size=self.batch_size, collate_fn=self.voice_test.collate)

if __name__ == "__main__":
    voice_datamodule = VoiceDataModule(batch_size=32, data_dir="/root/data/reazon_small/")
    voice_datamodule.setup("train")
    train_dataloader = voice_datamodule.train_dataloader()
    for batch in train_dataloader:
        print(batch)  
    val_dataloader = voice_datamodule.val_dataloader()
    test_dataloader = voice_datamodule.test_dataloader()
