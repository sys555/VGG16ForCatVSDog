import cv2
import numpy as np
from keras import backend as K
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from keras.optimizers import Adam
from keras.utils import get_file, np_utils
from PIL import Image

from model.VGG16 import VGG16

WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# 生成器 用于读取datasets文件夹里面的图片与标签

def generate_arrays_from_file(lines, batch_size):
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)

            # 读取图片 并 进行归一化 和resize

            name = lines[i].split(';')[0]
            img = Image.open("./data/image/train/" + name)
            img = np.array(img.resize((224, 224), Image.BICUBIC))
            img = img/255

            X_train.append(img)
            Y_train.append(lines[i].split(";")[1])
            i = (i+1) % n

        X_train = np.array(X_train)
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        yield(X_train, Y_train)

if __name__ == "__main__":
    # 分类个数
    NCLASSES = 2

    log_dir = "./logs/"
    model = VGG16(NCLASSES)

    # 这一步是获得主干特征提取网络的权重，即迁移学习

    # getfile 下载失败
    # weights_path = get_file("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", WEIGHTS_PATH_NO_TOP, cache_subdir="models", file_hash="6d6bbae143d832006294945121d1f1fc")
    weights_path = "./model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    model.load_weights(weights_path, by_name=True)

    with open("./data/train.txt","r") as f:
        lines = f.readlines()

    # 90% train 10 verify
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    #-----callback设置--------------------------------------------------------------------------#
    # checkpoint: 设置权制保存的细节，period用于修改多少epoch保存一次
    # reduce_lr: 学习率下降方式
    # early_stopping: 早停，val_loss多次不下降自动结束训练
    #-------------------------------------------------------------------------------#
    checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                 monitor="val_loss", save_weights_only=True, save_best_only=False, period = 2)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1)

    # backbone提取出来的特征是通用的
    # 首先不训练主干部分，因此训练部分分为两步，分别是冻结训练和解冻训练
    # 冻结训练不训练主干，解冻训练是训练主干的
    # 由于训练的特征层变多，解冻后所需显存变大

    trainable_layer = 19
    for i in range(trainable_layer):
        model.layers[i].trainable = False

    print("freeze the first {} layers of total {} layers.".format(trainable_layer, len(model.layers)))

    if True:
        lr = 1e-3
        batch_size = 4
        model.compile(loss = "categorical_crossentropy",
                      optimizer = Adam(lr=lr),
                      metrics=["accuracy"])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + "middle_one.h5")

    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    print("ice-out the first {} layers of total {} layers.".format(trainable_layer, len(model.layers)))

    if True:
        lr = 1e-4
        batch_size = 4
        model.compile(loss = "categorical_crossentropy",
                      optimizer = Adam(lr=lr),
                      metrics = ["accuracy"])

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=100,
                            initial_epoch=50,
                            callbacks=[checkpoint, reduce_lr,early_stopping])

        model.save_weights(log_dir+'last_one.h5')
