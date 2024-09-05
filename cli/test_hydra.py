# main.py
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="conf/eval", config_name="datasets")
def main(cfg: DictConfig):
    datasets = cfg.datasets
    print(OmegaConf.to_yaml(cfg))
    print("Datasets:", datasets)
    # Now you can use the `datasets` list as needed
    # For example, processing each dataset
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")


if __name__ == "__main__":
    main()
