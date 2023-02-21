#!/usr/bin/env python3
import hydra
import omegaconf


@hydra.main(config_path="configs", config_name="main")
def train(cfg: omegaconf.DictConfig):
    """import here so hydra's --multirun works with slurm"""
    from experiments.training import train

    train(cfg)


if __name__ == "__main__":
    train()  # pyright: ignore
