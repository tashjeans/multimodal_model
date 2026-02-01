from boltz.main import train
import yaml

if __name__ == "__main__":
    config_path = "config/sequence_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train(config)
