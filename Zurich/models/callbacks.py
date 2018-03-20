import keras
from IPython.display import clear_output
from keras.callbacks import *
import matplotlib.pyplot as plt


class PlotLosses(keras.callbacks.Callback):
    def __init__(self):
        self.epoch = 0
        self.losses = []
        self.acc = []
        self.val_acc = []
        self.acc_fn = []
        self.val_acc_fn = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_train_begin(self, logs=None):
        self.epoch = 0
        self.losses = []
        self.acc = []
        self.val_acc = []
        self.acc_fn = []
        self.val_acc_fn = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        self.logs.append(logs)
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.acc_fn.append(logs.get('fn'))
        self.val_acc_fn.append(logs.get('val_fn'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.epoch += 1
        fig, axes = plt.subplots(1, 3, sharex=True)
        fig.set_size_inches((20, 6))

        clear_output()

        axes[0].plot(np.arange(self.epoch), self.losses, label="Training")
        axes[0].plot(np.arange(self.epoch), self.val_losses, label="Validation")
        axes[0].legend()

        axes[1].plot(np.arange(self.epoch), self.acc, label="Training")
        axes[1].plot(np.arange(self.epoch), self.val_acc, label="Validation")
        axes[1].legend()

        axes[2].plot(np.arange(self.epoch), self.acc_fn, label="Training")
        axes[2].plot(np.arange(self.epoch), self.val_acc_fn, label="Validation")
        axes[2].legend()

        axes[0].set_title("Loss")
        axes[1].set_title("Accuracy")
        axes[2].set_title("Accuracy (wo. background)")
        plt.show()