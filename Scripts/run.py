# ## Install lib
# from IPython import get_ipython

# get_ipython().system('pip install git+https://github.com/qubvel/efficientnet')
# get_ipython().system('pip install git+https://github.com/qubvel/classification_models.git')
# get_ipython().system('pip install git+https://github.com/qubvel/segmentation_models')
# get_ipython().system('pip install -U git+https://github.com/albu/albumentations')

import callbacks
import generator
import make_submission
import constants
from models import *
import argparse
import helpers
import matplotlib.image as mpimg
import os, sys
import cv2
from skimage.io import imread
from skimage.transform import resize
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from keras.utils import Sequence
from tensorflow.keras.utils import Sequence
import tensorflow.python.platform
from keras.models import *
from keras.optimizers import Adam
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss, bce_dice_loss
import segmentation_models as sm
import data_division

def main():
    #########################################
    # input parser
    #########################################
    parser = argparse.ArgumentParser(description='Road Segmentation Challenge- EPFL.')

    group_data = parser.add_argument_group('model arguments')
    group_data.add_argument('--model',
                       type=str, default="unet",
                       choices = ["unet","manunet4","manunet5","manunet6"],
                       help='select the Neural Network model you desired to use.')
    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    modelname = args.model

    #########################################
    # generate data
    #########################################
    # 1: Devide the data
    data_division.make_folders()

    # 2 : Load entire images

    # Generators
    training_generator = generator.DataGenerator(constants.train_image_path, constants.train_mask_path,
                                                 augmentation=helpers.aug_with_crop, batch_size=1, )
    validation_generator = generator.DataGenerator(constants.val_image_path, constants.val_mask_path)

    #########################################
    # Model and training
    #########################################
    if (modelname == "manunet4"):
        model = unetManual()
        model.summary()
    elif (modelname == "manunet5"):
        model = unetManualFiveDeep()
        model.summary()
    elif (modelname == "manunet6"):
        model = unetManualSixDeep()
        model.summary()
    else:
        model = Unet(backbone_name='efficientnetb7', encoder_weights='imagenet', encoder_freeze=False)
        model.compile(optimizer='Adam', loss=bce_jaccard_loss, metrics=[sm.metrics.FScore(threshold=0.5)])


    history = model.fit_generator(training_generator, shuffle=True,
                                  epochs=30, workers=4, use_multiprocessing=True,
                                  validation_data=validation_generator,
                                  verbose=1, callbacks=[callbacks.lr_reducer])
    # plotting history
    #helpers.plot_training_history(history)

    # Save model
    model.save(constants.PATH + "saved_"+modelname+".h5")
    print("Trained model was successfully saved on disk.")
    #model = load_model(constants.PATH + "saved_"+modelname+".h5")

    #########################################
    # Testing and make predictions
    #########################################
    test = helpers.listdir_fullpath(constants.IMAGE_PATH)
    os.makedirs(constants.MASK_TEST_PATH)

    for pth in test:
        name = os.listdir(pth)[0]
        path = pth + "/" + name
        print(path)
        image = mpimg.imread(path) / 255
        if (modelname == "manunet4" or modelname == "manunet5" or modelname == "manunet6"): 
            image = cv2.resize(image, dsize=(384, 384), interpolation=cv2.INTER_CUBIC) # resize test images to (384,384) to feed to manual Unet
            prediction = cv2.resize(model.predict(np.expand_dims(image, axis=0)).reshape(384,384), dsize=(608, 608), interpolation=cv2.INTER_CUBIC) # resize the predictions to (608,608)
        else:
            prediction = model.predict(np.expand_dims(image, axis=0)).reshape(608, 608)
        mpimg.imsave(constants.MASK_TEST_PATH + name, prediction)
        print("Image " + name + " saved")

    submission_filename = constants.PATH + "test_final_"+modelname+".csv"
    image_filenames = helpers.listdir_fullpath(constants.MASK_TEST_PATH)
    make_submission.masks_to_submission(submission_filename, *image_filenames)


if __name__ == '__main__':
    main()
