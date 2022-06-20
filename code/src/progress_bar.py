from pytorch_lightning.callbacks.progress import ProgressBar
from tqdm.notebook import tqdm as tqdm

class LitProgressBar(ProgressBar):

    
    def init_validation_tqdm(self):
        bar = tqdm(            
            disable=True,            
        )
        return bar
    
    def init_train_tqdm(self):
        bar = super().init_validation_tqdm()
        return bar
    
