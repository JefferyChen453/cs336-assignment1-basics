import argparse
import os

import torch
from tqdm import tqdm
import yaml

from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.transformer.model import TransformerLM
from cs336_basics.transformer.optimizer import (
    AdamW,
    cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
)
from cs336_basics.transformer.train.data_loader import get_batch_data, read_nparray


class Trainer():
    def __init__(self, conf_path):
        self.load_config(conf_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_dataset = read_nparray(self.data_conf["train_file"])
        self.val_dataset = read_nparray(self.data_conf["val_file"])

        self.model = TransformerLM(**self.model_conf, device=self.device)
        self.tokenizer = self.get_tokenizer()
        self.optimizer = AdamW(self.model.parameters(), **self.optim_conf)

        self.total_steps = self.trainer_conf["total_steps"]
        self.batch_size = self.data_conf["batch_size"]
        self.context_length = self.model_conf["context_length"]
        self.warmup_iters = self.optim_conf["warmup_ratio"] * self.total_steps
        self.save_checkpoint_path = self.trainer_conf["save_checkpoint_path"]
        self.save_freq = self.trainer_conf["save_freq"]


    def get_tokenizer(self):
        tokenizer_path = self.tokenizer_conf["path"]
        vocab_filepath = os.path.join(tokenizer_path, "vocab.json")
        merges_filepath = os.path.join(tokenizer_path, "merges.txt")
        special_tokens = ["<|endoftext|>"]

        return Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    def train(self):
        max_lr, min_lr, clip = self.optim_conf["lr"], self.optim_conf["min_lr"], self.optim_conf["clip"]
        for it in tqdm(range(self.total_steps), desc="Training process", total=self.total_steps):
            self.optimizer.zero_grad()
            lr = get_lr_cosine_schedule(it, max_lr, min_lr, self.warmup_iters, self.total_steps)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            x, y = get_batch_data(self.train_dataset, self.batch_size, self.context_length, self.device)
            logits = self.model(x)
            loss = cross_entropy(logits, y, self.device)
            loss.backward()
            print(f"{loss=}")
            if self.optim_conf["gradient_clipping"]:
                gradient_clipping(self.model.parameters(), clip)
            self.optimizer.step()


    def load_config(self, conf_path):
        with open(conf_path, "r") as f:
            config = yaml.safe_load(f)
        self.data_conf = config["data"]
        self.tokenizer_conf = config["tokenizer"]
        self.model_conf = config["model"]
        self.optim_conf = config["optim"]
        self.trainer_conf = config["trainer"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file path")

    args = parser.parse_args()
    conf_path = args.config
    trainer = Trainer(conf_path)
    trainer.train()


if __name__ == "__main__":
    main()