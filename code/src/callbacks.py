from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from Bias_GAN.code.src.utils import get_version
from Bias_GAN.code.src.inference import Inference, EvaluateCheckpoints, create_folder
import matplotlib.pyplot as plt

import glob
from datetime import datetime
from uuid import uuid1
from io import BytesIO
from PIL import Image
import torchvision
import os

def get_cycle_gan_callbacks(checkpoint_path) -> list:

    lr_logger = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(monitor="g_tot_val_loss",
                                          dirpath=checkpoint_path,
                                          save_top_k = 50,
                                          #period=2,
                                          save_last=True)

    callbacks = [lr_logger, checkpoint_callback]

    return callbacks


class MAE_Callback(Callback):
    def __init__(self,logger,checkpoint_path,config, validation=True, lat_mean=False, plt_hist=False):
        self.MAE_list = []
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.version = get_version(config.date,config.time)
        self.validation = validation
        self.lat_mean = lat_mean
        self.plt_hist = plt_hist
        

    def on_train_epoch_end(self, trainer, pl_module):
        checkpoint_files = glob.glob(str(self.checkpoint_path) + '/*.ckpt')
        if not checkpoint_files:
            test_data_ = None
        else:
            last_checkpoint = max(checkpoint_files, key=os.path.getctime)
            data = EvaluateCheckpoints(checkpoint_path=last_checkpoint, config_path=self.config.config_path + self.version + "/config_model.json", save_model=True,validation=self.validation, version=self.version)
            _, reconstruction_data = data.run()
            test_data_ = data.get_test_data()


        if test_data_ is None or not test_data_:
            print("No test data available.")
            return

        gan_data = getattr(test_data_, 'gan')
        era5_data = getattr(test_data_, "era5")
        bias = gan_data.mean('time') - era5_data.mean('time') 
        print("GAN-OBS",f" \t \t MAE: {abs(bias).values.mean():2.3f} [mm/d]")
        self.MAE_list.append(abs(bias).values.mean())
        print("MAE_list:",self.MAE_list)

        self.log('MAE', abs(bias).values.mean())

        if test_data_ is not None and self.lat_mean==True:
            data_era5 = era5_data.mean(dim=("longitude", "time"))
            data_gan= gan_data.mean(dim=("longitude", "time"))
            plt.figure()
            plt.plot(data_gan.latitude, data_gan.data,
                      label="gan",
                      alpha=0.9,
                      linestyle='-',
                      linewidth=2,
                      color="red")
            
            plt.plot(data_era5.latitude, data_era5,
                      label="era5",
                      alpha=1,
                      linestyle='--',
                      linewidth=2,
                      color="black")
            
            plt.ylim(0,3)
            plt.xlim(25,58)
            plt.xlabel('Latitude')
            plt.ylabel('Mean precipitation [mm/d]')
            plt.grid()
            plt.legend(loc='upper right')  
            #plt.show()
          
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            im = Image.open(buf)
            img = torchvision.transforms.ToTensor()(im)
            
            self.logger.experiment.add_image(f"latitudinal_mean", img, trainer.current_epoch)

        if test_data_ is not None and self.plt_hist==True:
            data_gan = getattr(test_data_, "gan").values.flatten()
            data_era5 = getattr(test_data_, "era5").values.flatten()
            plt.figure()
            _ = plt.hist(data_gan,
                        bins=100,
                        histtype='step',
                        log=True,
                        label="gan",
                        alpha=0.9,
                        density=True,
                        linewidth=2,
                        color="red")
            
            _ = plt.hist(data_era5,
                        bins=100,
                        histtype='step',
                        log=True,
                        label="era5",
                        alpha=1,
                        density=True,
                        linewidth=2,
                        color="black")

            plt.xlabel('Precipitation [mm/d]')
            plt.ylabel('Histogram')
            plt.xlim(0,150)
            plt.grid()
            plt.legend(loc='upper right')

            #plt.show()
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            im_ = Image.open(buf)
            img_ = torchvision.transforms.ToTensor()(im_)
            
            self.logger.experiment.add_image(f"histogram", img_, trainer.current_epoch)


