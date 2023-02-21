import hydra
import omegaconf


@hydra.main(config_path="configs", config_name="main")
def train(cfg: omegaconf.DictConfig):
    from experiments.train import train

    train(cfg)


if __name__ == "__main__":
    train()  # pyright: ignore
