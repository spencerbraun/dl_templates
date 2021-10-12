import logging
from datetime import datetime

import hydra
import numpy as np
import torch
import wandb
from hydra import compose, initialize
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from transformers import AdamW
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)


def train(cfg: DictConfig, model, train_loader, val_loader):

    adamw_kwargs = {"betas": (0.9, 0.999), "eps": 1e-08}
    optimizer = AdamW(model.parameters(), lr=cfg.train.hyperparams.lr, **adamw_kwargs)

    if cfg.train.hyperparams.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.train.hyperparams.warmup_steps,
            num_training_steps=cfg.train.hyperparams.total_steps,
        )
    elif cfg.train.hyperparams.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.train.hyperparams.warmup_steps,
            num_training_steps=cfg.train.hyperparams.total_steps,
        )


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Using the model: {cfg.model.name}")
    logger.info(f"Using the tokenizer: {cfg.model.tokenizer}")

    root_dir = hydra.utils.get_original_cwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not cfg.train.state.debug:
        run_name = f"{cfg.train.run_name}_{cfg.model.model}_{cfg.data.name}_{timestamp}"
        wandb.init(
            project=args.project,
            config=OmegaConf.to_yaml(cfg, resolve=True),
            name=run_name,
        )

    wandb.finish()


if __name__ == "__main__":
    main()
