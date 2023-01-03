import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import json

from Bias_GAN.code.src.model import CycleGAN
from Bias_GAN.code.src.data import DataModule
from Bias_GAN.code.src.utils import get_version, set_environment, get_checkpoint_path, save_config
from Bias_GAN.code.src.callbacks import get_cycle_gan_callbacks

def train_cycle_gan(config):
    """ Main routing to train the Cycle GAN """

    version = get_version()
    print(f'Running model: {version}')
    print(json.dumps(config.__dict__, indent=4))
    checkpoint_path = get_checkpoint_path(config, version)
    set_environment()
    save_config(config, version)

    tb_logger = TensorBoardLogger(config.tensorboard_path,
                           name=config.model_name,
                           default_hp_metric=False,
                           version = version)

    trainer = pl.Trainer(gpus = 1,
                         max_epochs = config.epochs,
                         #progress_bar_refresh_rate = config.progress_bar_refresh_rate,
                         precision = 16, 
                         callbacks = get_cycle_gan_callbacks(checkpoint_path),
                         num_sanity_val_steps = 1,
                         logger = tb_logger,
                         log_every_n_steps = config.log_every_n_steps,
                         deterministic = False)

    datamodule = DataModule(config, training_batch_size = config.train_batch_size,
                                    test_batch_size = config.test_batch_size)

    datamodule.setup("fit")

    model = CycleGAN(epoch_decay = config.epochs // 2,
                     running_bias=config.running_bias)

    trainer.fit(model, datamodule)

    print('Training finished')
    return model

