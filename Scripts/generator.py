from skimage.io import imread
import tensorflow as tf
import numpy as np
import helpers
from tensorflow.keras.utils import Sequence
from albumentations import Resize


# Import from the tutorial, maybe use another lib
#tf.logging.set_verbosity(tf.logging.ERROR)



class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, image_path, mask_path,
                 batch_size=1, dim=(400, 400),
                 n_channels=3, augmentation=None, shuffle=True):

        self.image_path = image_path
        self.mask_path = mask_path
        self.image_filenames = helpers.listdir_fullpath(image_path)
        self.mask_filenames = helpers.listdir_fullpath(mask_path)
        self.batch_size = batch_size
        self.dim = dim
        self.out_dim = (
            (dim[0] // 32) * 32,
            (dim[1] // 32) * 32)  # Needs to be a multiple of 32 (Because of the structure of the NN)
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        # Generate indexes of the batch
        data_index_min = int(index * self.batch_size)
        data_index_max = int(min((index + 1) * self.batch_size, len(self.image_filenames)))

        indexes = self.image_filenames[data_index_min:data_index_max]

        this_batch_size = len(indexes)  # The last batch can be smaller than the others

        # Defining dataset
        X = np.empty((this_batch_size, self.out_dim[0], self.out_dim[0], self.n_channels), dtype=np.float32)
        y = np.empty((this_batch_size, self.out_dim[0], self.out_dim[0], 1), dtype=np.uint8)

        for i, sample_index in enumerate(indexes):  # For all batches
            X_sample, y_sample = self.read_image(self.image_filenames[index * self.batch_size + i],
                                                 self.mask_filenames[index * self.batch_size + i])

            # Augmentation code
            if self.augmentation is not None:
                augmented = self.augmentation(self.out_dim[0])(image=X_sample, mask=y_sample)
                image_augm = augmented['image']
                mask_augm = augmented['mask'].reshape(self.out_dim[0], self.out_dim[0], 1)
                X[i, ...] = np.clip(image_augm, a_min=0, a_max=1)
                y[i, ...] = mask_augm

            # No augmentation for validation.
            elif self.augmentation is None and self.batch_size == 1:
                X_sample, y_sample = self.read_image(self.image_filenames[index * 1 + i],
                                                     self.mask_filenames[index * 1 + i])
                augmented = Resize(height=(self.out_dim[0]), width=(self.out_dim[1]))(image=X_sample,
                                                                                      mask=y_sample)  # From albumentation, maybe use another lib
                X_sample, y_sample = augmented['image'], augmented['mask']
                return X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], self.n_channels).astype(np.float32), \
                       y_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], 1).astype(np.uint8)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
            """
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def read_image(self, image_name, mask_name):
        return imread(image_name) / 255, (imread(mask_name, as_gray=True) > 0).astype(np.int8)
