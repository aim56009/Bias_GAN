from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

def get_cycle_gan_callbacks(checkpoint_path) -> list:

    lr_logger = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(monitor="g_tot_val_loss",
                                          dirpath=checkpoint_path,
                                          save_top_k = 50,
                                          #period=2,
                                          save_last=True)

    callbacks = [lr_logger, checkpoint_callback]

    return callbacks
