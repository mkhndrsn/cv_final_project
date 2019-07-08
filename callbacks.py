from keras.callbacks import Callback
import keras.backend as K


class AccuracyRecorderCallback(Callback):
    def __init__(self):
        self.all_logs = []

    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        self.all_logs.append(logs)
