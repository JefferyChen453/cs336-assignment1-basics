import argparse
import yaml

class Trainer():
    def __init__():
        
        self.optimizer = 

    def train():
        pass
    
    def load_config(self, conf_path):
        with open(conf_path, "r") as f:
            config = yaml.safe_load(f)
        self.data_conf = config["data"]
        self.tokenizer_conf = config["tokenizer"]
        self.model_conf = config["model"]
        self.optim_conf = config["optim"]
        self.trainer_conf = config["trainer"]



def main():
    trainer = Trainer()
