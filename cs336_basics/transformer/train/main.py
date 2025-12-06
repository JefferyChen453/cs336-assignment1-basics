import argparse
from datetime import datetime
import logging
import os

import torch
from tqdm import tqdm
import wandb
import yaml

from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.transformer.model import TransformerLM, generate
from cs336_basics.transformer.optimizer import (
    AdamW,
    cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
)
from cs336_basics.transformer.train.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.transformer.train.data_loader import get_batch_data, read_nparray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Trainer():
    def __init__(self, conf_path):
        TIME = datetime.now().strftime("%Y%m%d_%H%M")

        self.load_config(conf_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_dataset = read_nparray(self.data_conf["train_file"])
        self.val_dataset = read_nparray(self.data_conf["val_file"])

        self.model = TransformerLM(**self.model_conf, device=self.device)
        self.tokenizer = self.get_tokenizer()
        self.optimizer = AdamW(self.model.parameters(), **self.optim_conf)

        self.total_steps = self.trainer_conf["total_steps"]
        self.train_batch_size = self.data_conf["train_batch_size"]
        self.val_batch_size = self.data_conf["val_batch_size"]
        self.context_length = self.model_conf["context_length"]
        self.warmup_iters = self.optim_conf["warmup_ratio"] * self.total_steps

        self.project_name = self.trainer_conf["project_name"]
        self.experiment_name = self.trainer_conf["experiment_name"] + "_" + TIME
        self.save_checkpoint_path = os.path.join(
            self.trainer_conf["save_checkpoint_path"],
            self.project_name,
            self.experiment_name,
        )
        self.save_freq = self.trainer_conf["save_freq"]
        self.val_freq = self.trainer_conf["val_freq"]

        self.run = wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config={
                "model": self.model_conf,
                "optim": self.optim_conf,
                "data": self.data_conf,
                "trainer": self.trainer_conf,
            }
        )

    def load_config(self, conf_path):
        with open(conf_path, "r") as f:
            config = yaml.safe_load(f)
        self.data_conf = config["data"]
        self.tokenizer_conf = config["tokenizer"]
        self.model_conf = config["model"]
        self.optim_conf = config["optim"]
        self.trainer_conf = config["trainer"]

    def get_tokenizer(self):
        tokenizer_path = self.tokenizer_conf["path"]
        vocab_filepath = os.path.join(tokenizer_path, "vocab.json")
        merges_filepath = os.path.join(tokenizer_path, "merges.txt")
        special_tokens = ["<|endoftext|>"]

        return Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    def train(self):
        if self.trainer_conf["val_before_train"]:
            self.evaluate(iter=0)
        
        max_lr, min_lr, clip = self.optim_conf["lr"], self.optim_conf["min_lr"], self.optim_conf["clip"]
        for it in tqdm(range(self.total_steps), desc="Training process", total=self.total_steps):
            self.optimizer.zero_grad()

            lr = get_lr_cosine_schedule(it, max_lr, min_lr, self.warmup_iters, self.total_steps)
            x, y = get_batch_data(self.train_dataset, self.train_batch_size, self.context_length, self.device)
            logits = self.model(x)
            loss = cross_entropy(logits, y, self.device)
            loss.backward()

            # Record grad_norm
            grad_norm = None
            if self.optim_conf["gradient_clipping"]:
                grad_norm = gradient_clipping(self.model.parameters(), clip)

            if self.optim_conf["gradient_clipping"]:
                gradient_clipping(self.model.parameters(), clip)

            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.optimizer.step()

            # Logging
            log_dict = {
                "train/loss": loss.item(),
                "train/lr": lr,
            }
            if grad_norm is not None:
                log_dict["train/grad_norm"] = grad_norm
            wandb.log(log_dict, step=it)

            if it > 0 and it % self.save_freq == 0:
                save_checkpoint(self.model, self.optimizer, it, self.save_checkpoint_path)
            
            if it > 0 and it % self.val_freq == 0:
                self.evaluate(iter=it)
        
        # save the last checkpoint
        save_checkpoint(self.model, self.optimizer, it, self.save_checkpoint_path)
        
    
    def evaluate(self, iter):
        self.model.eval()
        logger.info("Start evaluating...")
        with torch.no_grad():
            x, y = get_batch_data(self.train_dataset, self.val_batch_size, self.context_length, self.device)
            logits = self.model(x)
            loss = cross_entropy(logits, y, self.device)
        ppl = torch.exp(loss)

        wandb.log({
            "val/loss": loss.item(),
            "val/ppl": ppl.item()
        }, step=iter)

        self.model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file path")

    args = parser.parse_args()
    conf_path = args.config
    trainer = Trainer(conf_path)
    trainer.train()


if __name__ == "__main__":
    main()