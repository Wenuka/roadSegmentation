from albumentations import (Compose, HorizontalFlip, RandomCrop, RandomGamma,
                            RandomRotate90, ShiftScaleRotate,
                            Transpose, VerticalFlip)

from albumentations import Resize
import os
import matplotlib as plt


# Helper
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in sorted(os.listdir(d))]


# Data augmentation
def aug_with_crop(image_size=384, crop_prob=0.5):
    return Compose([
        RandomCrop(width=image_size, height=image_size, p=crop_prob),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        RandomGamma(p=0.25),
        Resize(height=image_size, width=image_size)
    ], p=1)


def plot_training_history(history):
    """
    Plots model training history
    """
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 5))
    ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
    ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_acc.plot(history.epoch, history.history["f1-score"], label="Train f1")
    ax_acc.plot(history.epoch, history.history["val_f1-score"], label="Validation f1")
    ax_acc.legend()
