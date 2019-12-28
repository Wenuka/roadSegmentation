from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

# Define the callbacks

# reduces learning rate on plateau
from constants import PATH

lr_reducer = ReduceLROnPlateau(factor=0.1,
                               cooldown=10,
                               patience=10, verbose=1,
                               min_lr=0.1e-5)
# model autosave callbacks
mode_autosave = ModelCheckpoint(PATH + "checkpoint/road_crop.efficientnetb5imgsize.h5",
                                monitor='val_f1-score',
                                mode='max', save_best_only=True, verbose=1, period=10)

# stop learining as metric on validatopn stop increasing
early_stopping = EarlyStopping(patience=10, verbose=1, mode='auto')

# tensorboard for monitoring logs
tensorboard = TensorBoard(log_dir='./logs/tenboard', histogram_freq=0,
                          write_graph=True, write_images=False)

callbacks = [lr_reducer, tensorboard, early_stopping]
