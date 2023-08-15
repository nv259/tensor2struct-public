import numpy as np
import torch
import wandb
import attr 

from tensor2struct.utils import registry, random_state, vocab
from tensor2struct.utils import saver as saver_mod
from tensor2struct.commands import train, eqrm_train
from tensor2struct.training import eqrm


@attr.s
class EQRMTrainConfig(eqrm_train.EQRMTrainConfig):
    use_bert_training = attr.ib(kw_only=True)


class EQRMTrainer(eqrm_train.EQRMTrainer):
    def load_train_config(self):
        self.train_config = registry.instantiate(
            EQRMTrainConfig, self.config["eqrm_train"]
        )

        if self.train_config.use_bert_training:
            if self.train_config.clip_grad is None:
                self.logger.info("Gradient clipping is recommend for BERT training")


def main(args):
    # setup logger etc
    config, logger = train.setup(args=args)
    
    # construct trainer and do training
    trainer = EQRMTrainer(logger, config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == '__main__':
    args = train.add_parser()
    main(args)
