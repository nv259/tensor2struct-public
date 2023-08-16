import os
import numpy as np
import torch
import wandb
import attr

from tensor2struct.commands import train
from tensor2struct.training import eqrm
from tensor2struct.utils import registry, random_state, vocab
from tensor2struct.utils import saver


@attr.s 
class EQRMTrainConfig(train.TrainConfig):
    burnin_iters = attr.ib(default=2500)
    quantile = attr.ib(default=0.75)
    # lr = attr.ib(default=1e-6)
    

class EQRMTrainer(train.Trainer):
    def load_train_config(self):
        self.train_config = registry.instantiate(
            EQRMTrainConfig, self.config["eqrm_train"]
        )

    def load_optimizer(self, config):
        with self.init_random:
            if self.train_config.use_bert_training:
                bert_params = self.model.get_bert_parameters()
                non_bert_params = self.model.get_non_bert_parameters()
                
                assert len(bert_params) + len(non_bert_params) == len(list(self.model.parameters()))
                assert len(bert_params) > 0
                
                self.logger.info(
                    f"{len(bert_params)} BERT parameters and {len(non_bert_params)} non-BERT parameters"
                )
                
                optimizer = registry.construct(
                    "optimizer",
                    config["optimizer"], 
                    non_bert_params=non_bert_params,
                    bert_params=bert_params
                )
                
                lr_scheduler = registry.construct(
                    "lr_scheduler",
                    config.get("lr_scheduler", {"name": "noop"}),
                    param_groups=[
                        optimizer.non_bert_param_group,
                        optimizer.bert_param_group,
                    ],
                )
            else:
                optimizer = registry.construct(
                    "optimizer", 
                    config["optimizer"],
                    params=self.model.get_non_bert_parameters(),
                )
                
                lr_scheduler = registry.construct(
                    "lr_scheduler",
                    config.get("lr_scheduler", {"name": "noop"}),  # if not exist lr, return NoOp
                    param_groups = optimizer.param_groups,
                )
                
            eqrm_trainer = eqrm.EQRM(
                device=self.device,
                quantile = self.train_config.quantile,
                burnin_iters= self.train_config.burnin_iters,
                # lr = self.train_config.eqrm_lr  # maybe it is the same as learning rate 
            )

            return optimizer, lr_scheduler, eqrm_trainer

    def step(self, config, train_data_loader, optimizer, lr_scheduler, last_step, eqrm_trainer):
        with self.model_random:
            losses = []
            for _i in range(self.train_config.num_batch_accumulated):  
                batch = next(train_data_loader)
                losses =  losses + eqrm_trainer.train(self.model, batch, last_step)
           
            # print('\nnum_batch_accumulated:', self.train_config.num_batch_accumulated)
            # print('\nbatch size:', self.train_config.batch_size)
            # print('\ntotal:', len(losses))
             
            loss, reset_opt = eqrm_trainer.transform(losses, last_step)
            
            # clip grad for both bert and non-bert params
            if self.train_config.clip_grad and self.train_config.use_bert_training:
                for param_group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(
                        param_group["params"], self.train_config.clip_grad,
                    )
            
            # reset
            if reset_opt:
                print('Reset optimizer and lr scheduler')
                if self.train_config.use_bert_training:
                    optimizer = registry.construct(
                        "optimizer",
                        config["optimizer"],
                        non_bert_params=self.model.get_non_bert_parameters(),
                        bert_params=self.model.get_bert_parameters()
                    )

                    lr_scheduler = registry.construct(
                      "lr_scheduler",
                      config.get("lr_scheduler", {"name": "noop"}),
                      param_groups=[
                          optimizer.non_bert_param_group,
                          optimizer.bert_param_group,
                      ],
                    )
                else:
                    optimizer = registry.construct(
                        "optimizer",
                        config["optimizer"],
                        params=self.model.get_non_bert_parameters()
                    )

                    lr_scheduler = registry.construct(
                      "lr_scheduler",
                      config.get("lr_scheduler", {"name": "noop"}),
                      param_groups = optimizer.param_groups,
                    )

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            new_lr = lr_scheduler.update_lr(last_step)
            
            if new_lr is None:
                new_lr = [param["lr"] for param in optimizer.param_groups]
            
            if last_step % self.train_config.report_every_n == 0:
                self.logger.info("Step {}: loss={:.4f}".format(last_step, loss))
                self.logger.info(f"Step {last_step}, lr={new_lr}")
                wandb.log({"train_loss": loss}, step=last_step)
                for i in range(len(new_lr)):
                    wandb.log({f"lr_{i}": new_lr[i]}, step=last_step)
                # print(f'{[param["lr"] for param in optimizer.param_groups]}')

    def train(self, config, modeldir):
        optimizer, lr_scheduler, eqrm_trainer = self.load_optimizer(config)
        saver, last_step = self.load_saver(config, modeldir, optimizer=optimizer, eqrm_trainer=eqrm_trainer)
        
        train_data_loader = self.load_train_data()
        train_eval_data_loader, val_data_loader = self.load_eval_data()
        
        with self.data_random:
            while last_step < self.train_config.max_steps:
                oom = False
                try: 
                    self.eval_model(last_step, train_eval_data_loader, val_data_loader)
                    self.step(config, train_data_loader, optimizer, lr_scheduler, last_step, eqrm_trainer)
                    last_step = last_step + 1 
                    self.save_state(saver, modeldir, last_step)
                except RuntimeError as e:
                    err_msg = str(e)
                    self.logger.warn(f"Forward Failed: {err_msg}")        
                    oom = True
            
                if oom:
                    # save the checkpoints and load to cpu
                    tmp_step = int(1e8)
                    saver.save(modeldir, step=tmp_step)
                    self.model.to('cpu')
                    del self.model
                    _optimizer_to(optimizer, 'cpu')
                    del optimizer, lr_scheduler
                    torch.cuda.empty_cache()
                    import gc; gc.collect()
                    
                    # load again
                    self.load_model()
                    optimizer, lr_scheduler, eqrm_trainer = self.load_optimizer(config)
                    saver, _ = self.load_saver(config, modeldir, optimizer)
                    
                    # remove the tmp checkpoint
                    os.unlink(os.path.join(modeldir, f"model_checkpoint-{tmp_step}"))
        
            saver.save(modeldir, last_step)


def _optimizer_to(optimizer, device):
    "Move optimizer state to cpu"
    for param in optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def main(args):
    # setup logger etc
    config, logger = train.setup(args)
    
    # construct trainer and do training
    trainer = EQRMTrainer(logger, config)
    trainer.train(config, modeldir=args.logdir)


if __name__ == '__main__':
    args = train.add_parser()
    main(args)
    
    