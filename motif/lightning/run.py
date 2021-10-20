import hydra
import os
from omegaconf import DictConfig


@hydra.main(config_path="conf/", config_name="config.yaml")
def main(config: DictConfig):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train module
    return train(config)


if __name__ == "__main__":
    main()
