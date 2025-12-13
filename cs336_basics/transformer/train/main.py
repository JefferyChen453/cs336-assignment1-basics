import argparse
from datetime import datetime
import logging
import os
import json

import torch
from tqdm import tqdm
import wandb
import yaml

from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.transformer.model import TransformerLM
from cs336_basics.transformer.optimizer import (
    AdamW,
    cross_entropy,
    get_lr_cosine_schedule,
    gradient_clipping,
)
from cs336_basics.transformer.train.checkpoint import save_checkpoint
from cs336_basics.transformer.train.data_loader import get_batch_data, read_nparray

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Trainer():
    def __init__(self, config):
        TIME = datetime.now().strftime("%Y%m%d_%H%M")

        self.load_config(config)
        logger.info(json.dumps(config, indent=4, ensure_ascii=False))
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
        os.makedirs(self.save_checkpoint_path, exist_ok=True)

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

    def load_config(self, config):
        self.data_conf = config["data"]
        self.tokenizer_conf = config["tokenizer"]
        self.model_conf = config["model"]
        self.optim_conf = config["optim"]
        self.trainer_conf = config["trainer"]
        self.debug_conf = config["debug"]

    def get_tokenizer(self):
        tokenizer_path = self.tokenizer_conf["path"]
        vocab_filepath = os.path.join(tokenizer_path, "vocab.json")
        merges_filepath = os.path.join(tokenizer_path, "merges.txt")
        special_tokens = ["<|endoftext|>"]

        return Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    def train(self):
        if self.trainer_conf["val_before_train"]:
            self.evaluate(iter=0)
        
        max_lr, min_lr, clip = self.optim_conf["lr"], self.optim_conf["lr"] * 0.01, self.optim_conf["clip"]
        log_data_sample_path = os.path.join(self.save_checkpoint_path, "log.txt")
        for it in tqdm(range(self.total_steps), desc="Training process", total=self.total_steps, leave=False, dynamic_ncols=True):
            self.optimizer.zero_grad()

            lr = get_lr_cosine_schedule(it, max_lr, min_lr, self.warmup_iters, self.total_steps)
            x, y = get_batch_data(self.train_dataset, self.train_batch_size, self.context_length, self.device)
            if self.debug_conf["debug_mode"] and self.debug_conf["log_data_sample"]:
                self.log_data_sample(log_data_sample_path, it, x, y)
            logits = self.model(x)
            loss = cross_entropy(logits, y, self.device)
            loss.backward()

            if not torch.isfinite(loss):
                logger.error(f"loss became {loss} at step {it} -> divergent")
                break
            if loss.item() > 1e6:
                logger.error("loss exploded -> divergent")
                break

            # Record grad_norm
            grad_norm = None
            if self.optim_conf["gradient_clipping"]:
                grad_norm = gradient_clipping(self.model.parameters(), clip)

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

            if (it + 1) % self.save_freq == 0:
                save_checkpoint(self.model, self.optimizer, it + 1, self.save_checkpoint_path, self.trainer_conf["save_model_only"])
            
            if (it + 1) % self.val_freq == 0:
                self.evaluate(iter=it)
        
        # save the last checkpoint
        save_checkpoint(self.model, self.optimizer, it + 1, self.save_checkpoint_path)
    
    def evaluate(self, iter):
        self.model.eval()
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

    def log_data_sample(self, log_path, it, x, y, sample_idx=100):
        x_data = x[sample_idx].tolist() if hasattr(x, "tolist") else x[sample_idx]
        y_data = y[sample_idx].tolist() if hasattr(y, "tolist") else y[sample_idx]
        
        x_text = self.tokenizer.decode(x_data)
        y_text = self.tokenizer.decode(y_data)
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*100}\nIteration: {it}\nX:\n{x_text}\nY:\n{y_text}\n{'='*100}\n")

def apply_overrides(config, overrides):
    """
    overrides: ["--data.train_batch_size", "16", "--optim.lr", "0.0005", ...]
    """
    def set_nested(cfg, key_path, value):
        keys = key_path.split(".")
        cur = cfg
        for k in keys[:-1]:
            if k not in cur:
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = auto_cast(value)

    i = 0
    while i < len(overrides):
        key = overrides[i]
        if not key.startswith("--"):
            i += 1
            continue
        key = key[2:]
        value = overrides[i + 1]
        set_nested(config, key, value)
        i += 2
    return config

def auto_cast(value):
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() in ["true", "false"]:
        return value.lower() == "true"
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file path")

    args, overrides = parser.parse_known_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = apply_overrides(config, overrides)

    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()