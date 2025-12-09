import argparse
import os

import torch
import yaml

from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.transformer.model import TransformerLM, generate
from cs336_basics.transformer.train.checkpoint import load_checkpoint


class TextGenerator():
    def __init__(self, config):
        self.load_config(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = TransformerLM(**self.model_conf, device=self.device)
        self.tokenizer = self.get_tokenizer()
        self.load_checkpoint_path = self.model_conf["load_checkpoint_path"]
        load_checkpoint(self.load_checkpoint_path, self.model)

    def load_config(self, config):
        self.tokenizer_conf = config["tokenizer"]
        self.model_conf = config["model"]

    def get_tokenizer(self):
        tokenizer_path = self.tokenizer_conf["path"]
        vocab_filepath = os.path.join(tokenizer_path, "vocab.json")
        merges_filepath = os.path.join(tokenizer_path, "merges.txt")
        special_tokens = ["<|endoftext|>"]

        return Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    def generate(self, sentence):
        prompt_ids = self.tokenizer.encode(sentence)
        prompt_ids = [prompt_ids]
        generate_ids = generate(
            self.model,
            prompt_ids,
        )
        generate_text = self.tokenizer.decode(generate_ids)
        print(generate_text)
        return generate_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--prompt", type=str, help="Prompt")


    args, overrides = parser.parse_known_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    textgenerator = TextGenerator(config)
    textgenerator.generate(args.prompt)
