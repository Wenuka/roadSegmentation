from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import segmentation_models as sm

## Unet with four layers in each side
def unetManual(pretrained_weights = None,input_size = (384,384,3)):
    inputs = Input(input_size)
    firstKernel = 64
    conv1 = Conv2D(firstKernel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(firstKernel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(firstKernel*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(firstKernel*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(firstKernel*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(firstKernel*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(firstKernel*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(firstKernel*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    dropmid1 = Dropout(0.5)(conv4)
    poolmid = MaxPooling2D(pool_size=(2, 2))(dropmid1)

    convmid = Conv2D(firstKernel*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(poolmid)
    convmid = Conv2D(firstKernel*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convmid)
    dropmid2 = Dropout(0.5)(convmid)

    up26 = Conv2D(firstKernel*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(dropmid2))
    merge26 = concatenate([dropmid1,up26], axis = 3)
    conv26 = Conv2D(firstKernel*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge26)
    conv26 = Conv2D(firstKernel*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv26)

    up27 = Conv2D(firstKernel*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv26))
    merge27 = concatenate([conv3,up27], axis = 3)
    conv27 = Conv2D(firstKernel*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge27)
    conv27 = Conv2D(firstKernel*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv27)

    up28 = Conv2D(firstKernel*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv27))
    merge28 = concatenate([conv2,up28], axis = 3)
    conv28 = Conv2D(firstKernel*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge28)
    conv28 = Conv2D(firstKernel*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv28)

    up29 = Conv2D(firstKernel, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv28))
    merge29 = concatenate([conv1,up29], axis = 3)
    conv29 = Conv2D(firstKernel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge29)
    conv29 = Conv2D(firstKernel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv29)
    conv29 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv29)
    convOut = Conv2D(1, 1, activation = 'sigmoid')(conv29)

    model = Model(input = inputs, output = convOut)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy',sm.metrics.FScore()])
    
    # if(pretrained_weights):
    # 	model.load_weights(pretrained_weights)

    return model

## Unet with six layers in each side
def unetManualFiveDeep(pretrained_weights = None,input_size = (384,384,3)):
    inputs = Input(input_size)
    firstKernel = 64
    conv1 = Conv2D(firstKernel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(firstKernel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(firstKernel*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(firstKernel*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(firstKernel*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(firstKernel*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(firstKernel*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(firstKernel*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(firstKernel*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(firstKernel*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    dropmid1 = Dropout(0.5)(conv5)
    poolmid = MaxPooling2D(pool_size=(2, 2))(dropmid1)

    convmid = Conv2D(firstKernel*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(poolmid)
    convmid = Conv2D(firstKernel*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convmid)
    dropmid2 = Dropout(0.5)(convmid)

    up25 = Conv2D(firstKernel*16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(dropmid2))
    merge25 = concatenate([dropmid1,up25], axis = 3)
    conv25 = Conv2D(firstKernel*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge25)
    conv25 = Conv2D(firstKernel*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv25)

    up26 = Conv2D(firstKernel*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv25))
    merge26 = concatenate([conv4,up26], axis = 3)
    conv26 = Conv2D(firstKernel*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge26)
    conv26 = Conv2D(firstKernel*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv26)

    up27 = Conv2D(firstKernel*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv26))
    merge27 = concatenate([conv3,up27], axis = 3)
    conv27 = Conv2D(firstKernel*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge27)
    conv27 = Conv2D(firstKernel*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv27)

    up28 = Conv2D(firstKernel*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv27))
    merge28 = concatenate([conv2,up28], axis = 3)
    conv28 = Conv2D(firstKernel*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge28)
    conv28 = Conv2D(firstKernel*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv28)

    up29 = Conv2D(firstKernel, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv28))
    merge29 = concatenate([conv1,up29], axis = 3)
    conv29 = Conv2D(firstKernel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge29)
    conv29 = Conv2D(firstKernel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv29)
    conv29 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv29)
    convOut = Conv2D(1, 1, activation = 'sigmoid')(conv29)

    model = Model(input = inputs, output = convOut)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy',sm.metrics.FScore()])

    return model

## Unet with six layers in each side
def unetManualSixDeep(pretrained_weights = None,input_size = (384,384,3)):
    inputs = Input(input_size)
    firstKernel = 64
    conv1 = Conv2D(firstKernel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(firstKernel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(firstKernel*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(firstKernel*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(firstKernel*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(firstKernel*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(firstKernel*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(firstKernel*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(firstKernel*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(firstKernel*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Conv2D(firstKernel*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
    conv6 = Conv2D(firstKernel*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    dropmid1 = Dropout(0.5)(conv6)
    poolmid = MaxPooling2D(pool_size=(2, 2))(dropmid1)

    convmid = Conv2D(firstKernel*64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(poolmid)
    convmid = Conv2D(firstKernel*64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(convmid)
    dropmid2 = Dropout(0.5)(convmid)

    up24 = Conv2D(firstKernel*32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(dropmid2))
    merge24 = concatenate([dropmid1,up24], axis = 3)
    conv24 = Conv2D(firstKernel*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge24)
    conv24 = Conv2D(firstKernel*32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv24)

    up25 = Conv2D(firstKernel*16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv24))
    merge25 = concatenate([conv5,up25], axis = 3)
    conv25 = Conv2D(firstKernel*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge25)
    conv25 = Conv2D(firstKernel*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv25)

    up26 = Conv2D(firstKernel*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv25))
    merge26 = concatenate([conv4,up26], axis = 3)
    conv26 = Conv2D(firstKernel*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge26)
    conv26 = Conv2D(firstKernel*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv26)

    up27 = Conv2D(firstKernel*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv26))
    merge27 = concatenate([conv3,up27], axis = 3)
    conv27 = Conv2D(firstKernel*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge27)
    conv27 = Conv2D(firstKernel*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv27)

    up28 = Conv2D(firstKernel*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv27))
    merge28 = concatenate([conv2,up28], axis = 3)
    conv28 = Conv2D(firstKernel*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge28)
    conv28 = Conv2D(firstKernel*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv28)

    up29 = Conv2D(firstKernel, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv28))
    merge29 = concatenate([conv1,up29], axis = 3)
    conv29 = Conv2D(firstKernel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge29)
    conv29 = Conv2D(firstKernel, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv29)
    conv29 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv29)
    convOut = Conv2D(1, 1, activation = 'sigmoid')(conv29)

    model = Model(input = inputs, output = convOut)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy',sm.metrics.FScore()])

    return model