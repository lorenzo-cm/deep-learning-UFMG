from tqdm import tqdm
from gensim.models.callbacks import CallbackAny2Vec

class TQDMProgress(CallbackAny2Vec):
    """Callback to log training progress with tqdm."""
    def __init__(self, epochs):
        self.epochs = epochs
        self.pbar = tqdm(total=epochs, desc="Training Progress")
        self.epoch = 0

    def on_epoch_begin(self, model):
        pass  # Optionally log information at the start of an epoch

    def on_epoch_end(self, model):
        self.epoch += 1
        self.pbar.update(1)

    def on_train_end(self, model):
        self.pbar.close()