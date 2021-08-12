"""
MorphNet Model Zoo

Lei Mao
NVIDIA
https://github.com/leimao

Main script to start MorphNet training for selected models.
"""

import argparse
import tensorflow as tf
import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import keras
from keras.models import Sequential, Model, load_model
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from google.colab.patches import cv2_imshow
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,MaxPool2D
from keras.preprocessing import image
from keras.initializers import glorot_uniform

from model import MorphNetModel
from utils import set_reproducible_environment, select_keras_base_model, train_epoch, validate_epoch


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])  # SKIP Connection
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def resnet50(input_shape=(224, 224, 3)):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


def main():

    parser = argparse.ArgumentParser(
        description="Run MorphNet Algorithm on Image Classification Model Zoo.")

    num_epochs_default = 100
    num_classes_default = 18
    batch_size_default = 1024
    base_model_name_default = "ResNet50"
    learning_rate_default = 0.0001
    morphnet_regularizer_algorithm_default = "GroupLasso"
    morphnet_target_cost_default = "FLOPs"
    morphnet_hardware_default = "Others"
    morphnet_regularizer_threshold_default = 1e-2
    morphnet_regularization_multiplier_default = 1000.0
    log_dir_default = "./morphnet_log"
    main_train_device_default = "/cpu:0"
    main_eval_device_default = "/gpu:0"
    num_cuda_device_default = 1
    random_seed_default = 0
    base_model_choices = [
        "ResNet50", "ResNet101", "ResNet152", "ResNet50V2", "ResNet101V2",
        "ResNet101V2", "ResNet152V2", "VGG16", "VGG19", "Xception",
        "InceptionV3", "InceptionResNetV2", "MobileNet", "MobileNetV2",
        "DenseNet121", "DenseNet169", "DenseNet201", "NASNetLarge",
        "NASNetMobile"
    ]
    morphnet_regularizer_algorithm_choices = ["GroupLasso", "Gamma"]
    morphnet_target_cost_choices = ["FLOPs", "Latency", "ModelSize"]
    morphnet_hardware_choices = ["V100", "P100", "Others"]

    parser.add_argument("--num-epochs",
                        type=int,
                        help="The number of epochs for training.",
                        default=num_epochs_default)
    parser.add_argument("--num-classes",
                        type=int,
                        help="The number of classes for image classification.",
                        default=num_classes_default)
    parser.add_argument("--batch-size",
                        type=int,
                        help="Batch size.",
                        default=batch_size_default)
    parser.add_argument("--learning-rate",
                        type=float,
                        help="Learning rate.",
                        default=learning_rate_default)
    parser.add_argument("--base-model-name",
                        type=str,
                        choices=base_model_choices,
                        help="Select base model for image classification.",
                        default=base_model_name_default)
    parser.add_argument("--morphnet-regularizer-algorithm",
                        type=str,
                        choices=morphnet_regularizer_algorithm_choices,
                        help="Select MorphNet regularization algorithm.",
                        default=morphnet_regularizer_algorithm_default)
    parser.add_argument("--morphnet-target-cost",
                        type=str,
                        choices=morphnet_target_cost_choices,
                        help="Select MorphNet target cost.",
                        default=morphnet_target_cost_default)
    parser.add_argument("--morphnet-hardware",
                        type=str,
                        choices=morphnet_hardware_choices,
                        help="Select MorphNet hardware.",
                        default=morphnet_hardware_default)
    parser.add_argument(
        "--morphnet-regularizer-threshold",
        type=float,
        help="Set the threshold [0, 1] for killing neuron layers.",
        default=morphnet_regularizer_threshold_default)
    parser.add_argument(
        "--morphnet-regularization-multiplier",
        type=float,
        help=
        "Set MorphNet regularization multiplier for regularization strength. The regularization strength for ..."
        "training equals the regularization multiplier divided by the initial cost of the model. Set this value to ..."
        "zero turns of MorphNet regularization.",
        default=morphnet_regularization_multiplier_default)
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Log directory for TensorBoard and optimized model architectures.",
        default=log_dir_default)
    parser.add_argument("--num-cuda-device",
                        type=int,
                        help="Number of CUDA device to use.",
                        default=num_cuda_device_default)
    parser.add_argument("--random-seed",
                        type=int,
                        help="Random seed.",
                        default=random_seed_default)
    parser.add_argument(
        "--main-train-device",
        type=str,
        help="The device where the model parameters were located.",
        default=main_train_device_default)
    parser.add_argument("--main-eval-device",
                        type=str,
                        help="The device used for model evaluation",
                        default=main_eval_device_default)

    argv = parser.parse_args()

    num_epochs = argv.num_epochs
    num_classes = argv.num_classes
    batch_size = argv.batch_size
    base_model_name = argv.base_model_name
    learning_rate = argv.learning_rate
    morphnet_regularizer_algorithm = argv.morphnet_regularizer_algorithm
    morphnet_target_cost = argv.morphnet_target_cost
    morphnet_hardware = argv.morphnet_hardware
    morphnet_regularizer_threshold = argv.morphnet_regularizer_threshold
    morphnet_regularization_multiplier = argv.morphnet_regularization_multiplier
    log_dir = argv.log_dir
    num_cuda_device = argv.num_cuda_device
    random_seed = argv.random_seed
    main_train_device = argv.main_train_device
    main_eval_device = argv.main_eval_device

    set_reproducible_environment(random_seed=random_seed)

    # (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.cifar10.load_data()
    x_train = np.load('/content/drive/MyDrive/np/train.npy')
    y_train = np.load('/content/drive/MyDrive/np/train_label.npy')
    x_valid = np.load('/content/drive/MyDrive/np/validation.npy')
    y_valid = np.load('/content/drive/MyDrive/np/validation_label.npy')

    # Convert class vectors to binary class matrices.
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
    y_valid_onehot = tf.keras.utils.to_categorical(y_valid, num_classes)
    image_shape = x_train[1:]
    # Normalize image inputs
    # x_train = x_train.astype("float32") / 255.0
    # x_valid = x_valid.astype("float32") / 255.0

    # base_model = select_keras_base_model(base_model_name=base_model_name)
    base_model = resnet50()
    morphnet_regularization_strength_dummy = 1e-9
    model = MorphNetModel(
        base_model=base_model,
        num_classes=num_classes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_gpus=num_cuda_device,
        main_train_device=main_train_device,
        main_eval_device=main_eval_device,
        morphnet_regularizer_algorithm=morphnet_regularizer_algorithm,
        morphnet_target_cost=morphnet_target_cost,
        morphnet_hardware=morphnet_hardware,
        morphnet_regularizer_threshold=morphnet_regularizer_threshold,
        morphnet_regularization_strength=morphnet_regularization_strength_dummy,
        log_dir=log_dir)

    # Export the unmodified model configures.
    initial_cost = model.get_model_cost(inputs=x_train[:batch_size])
    print("*" * 100)
    print("Initial Model Cost: {:.1f}".format(initial_cost))
    morphnet_regularization_strength = 1.0 / initial_cost * morphnet_regularization_multiplier
    print("Use Regularization Strength: {}".format(
        morphnet_regularization_strength))
    model.set_morphnet_regularization_strength(
        morphnet_regularization_strength=morphnet_regularization_strength)
    print("*" * 100)
    # Export the unmodified model configures.
    model.export_model_config_with_inputs(inputs=x_train[:batch_size])

    for epoch in range(num_epochs):
        validate_epoch(epoch=epoch,
                       model=model,
                       x_valid=x_valid,
                       y_valid_onehot=y_valid_onehot,
                       batch_size=batch_size)
        train_epoch(epoch=epoch,
                    model=model,
                    x_train=x_train,
                    y_train_onehot=y_train_onehot,
                    batch_size=batch_size,
                    shuffle=True,
                    print_batch_info=False)
        # Export the model configure routinely.
        model.export_model_config_with_inputs(inputs=x_train[:batch_size])

    validate_epoch(epoch=num_epochs,
                   model=model,
                   x_valid=x_valid,
                   y_valid_onehot=y_valid_onehot,
                   batch_size=batch_size)

    model.close()

    return 0


if __name__ == "__main__":

    main()
