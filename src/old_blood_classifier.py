from __future__ import absolute_import, division, print_function, unicode_literals

import datetime

import tensorflow as tf
import os
import pathlib
import random
import matplotlib.pyplot as plt
import shutil
import cv2
import numpy as np
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16
PATH_TO_DATA = r"C:\_Programming\DataSets\First_png"
PATH_TEMP = r"C:\_Programming\DataSets\temp"
PATH_TO_TRAIN = "\TrainDataSet_20"
PATH_TO_TEST = "\TestDataSet_20"
LABEL_NAMES = ['Базофил', 'Бласты', 'Лимфоцит', 'Метамиелоцит', 'Миелоцит', 'Моноцит', 'Нормобласты',
               'Палочкоядерный нейтрофил', 'Промиелоцит', 'Сегментноядерный нейтрофил', 'Эозинофил']
EPOCHS = 10
act = 'relu'
TEST_STEPS = 100
# checkpoint_path = "Weights/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"/cp.ckpt"
checkpoint_path = "Weights/cp.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)
# Создаем коллбек сохраняющий веса модели
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def InitDataSet(path):
    # Прикрепление labloв
    path = pathlib.Path(path)
    all_image_paths = list(path.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    label_names = sorted(item.name for item in path.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    image_label_ds = image_label_ds.cache()
    return image_label_ds


def CountImg(path):
    path = pathlib.Path(path)
    all_image_paths = list(path.glob('*/*'))
    return len(all_image_paths)


def GelLabelsNames(path):
    path = pathlib.Path(path)
    return sorted(item.name for item in path.glob('*/') if item.is_dir())


# MLP
def InitModel2():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(192, 192, 3)),
        tf.keras.layers.Dense(1728, activation=act),
        tf.keras.layers.Dense(108, activation=act),
        tf.keras.layers.Dense(54, activation=act),
        tf.keras.layers.Dense(216, activation=act),
        tf.keras.layers.Dense(216, activation=act),
        tf.keras.layers.Dense(len(GelLabelsNames(PATH_TO_DATA)), activation='softmax')])
    return model

# CNN
def InitModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation=act, input_shape=(192, 192, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation=act),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation=act),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=act),
        tf.keras.layers.Dense(len(GelLabelsNames(PATH_TO_DATA)), activation='softmax')])
    return model


def StartTrain(path):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("train_80")
    
    # Логи обучения -- сохранение результатов каждой эпохи (accuracy, loss)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    image_label_ds = InitDataSet(path)
    ds = image_label_ds.shuffle(buffer_size=CountImg(path))
    print(CountImg(path))

    # Зацикливание датасета
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    # ds_test=InitDataSet(path_validation)
    # ds_test = ds_test.shuffle(buffer_size=CountImg(path))
    # print(CountImg(path))
    # # ds_test = ds_test.repeat()
    # ds_test = ds_test.batch(BATCH_SIZE)
    # # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
    # ds_test = ds_test.prefetch(buffer_size=AUTOTUNE)

    model = InitModel()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    print(CountImg(path) / BATCH_SIZE)
    steps_per_epoch = tf.math.ceil(CountImg(path) / BATCH_SIZE).numpy()

    model.fit(ds, epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              callbacks=[tensorboard_callback, cp_callback],
              verbose=1)
    print("обучение завершенно")


# TODO: Допилить визуализацию
# def Visualisation(history):
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']

#     loss = history.history['loss']
#     val_loss = history.history['val_loss']

#     epochs_range = range(EPOCHS)

#     plt.figure(figsize=(8, 8))
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs_range, acc, label='Training Accuracy')
#     plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#     plt.legend(loc='lower right')
#     plt.title('Training and Validation Accuracy')

#     plt.subplot(1, 2, 2)
#     plt.plot(epochs_range, loss, label='Training Loss')
#     plt.plot(epochs_range, val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.title('Training and Validation Loss')
#     plt.show()


def DevideData(ratio, path):
    path_train_data = pathlib.Path(PATH_TEMP + PATH_TO_TRAIN)
    path_test_data = pathlib.Path(PATH_TEMP + PATH_TO_TEST)
    path_train_data.mkdir()
    path_test_data.mkdir()

    path = pathlib.Path(path)
    all_image_classes = list(path.glob("*/"))

    for classs in all_image_classes:
        path_train_data = pathlib.Path(PATH_TEMP + PATH_TO_TRAIN)
        path_test_data = pathlib.Path(PATH_TEMP + PATH_TO_TEST)

        path = pathlib.Path(classs)
        all_images = list(path.glob("*/"))
        random.shuffle(all_images)

        count_images = len(all_images)
        count_test_images = int(count_images * (ratio / 100))
        train_images = []
        test_images = []

        for image in all_images[:count_test_images]:
            test_images.append(image)
        for image in all_images[count_test_images:]:
            train_images.append(image)

        path_train_data = path_train_data.joinpath(path.name)
        path_train_data.mkdir()
        path_test_data = path_test_data.joinpath(path.name)
        path_test_data.mkdir()

        for image in test_images:
            shutil.copy(image, path_test_data)
        for image in train_images:
            shutil.copy(image, path_train_data)

    # new_path.joinpath(path.name)
    # print(len(test_images),len(train_images),len(all_images))
    print("Готово")
    return


def ClearTempData():
    path = pathlib.Path(PATH_TEMP)
    data = list(path.glob("*/*"))
    for item in data:
        if os.path.isfile(item):
            os.remove(item)
    return


def TestNeiron(path):
    ds_test = InitDataSet(path)
    ds = ds_test.shuffle(buffer_size=CountImg(path))
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    model = InitModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()

    print("Загрузка модели")
    model.load_weights(checkpoint_path)
    print("Анализ точности модели")
    test_loss, test_acc = model.evaluate(ds, verbose=1, steps=TEST_STEPS)
    print('\nТочность на проверочных данных:', test_acc)


def Predictions_2(path):
    """Вытаскивает по путю картинки для предсказания и предсказывает."""
    path = pathlib.Path(path)
    all_image_paths = list(path.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    print(all_image_paths)

    for path in all_image_paths:
        print(path)
        print(path, Predict(path))

def Predict(path_img):
    model = InitModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()

    print("Загрузка модели")
    # checkpoint_path = "Weights_20/cp.ckpt"
    model.load_weights(checkpoint_path)

# ds_predict = InitDataSet(path_img)

    ds_predict = tf.data.Dataset.from_tensor_slices(path_img)
    ds_predict = ds_predict.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast("none", tf.int64))
    ds_predict = tf.data.Dataset.zip((ds_predict, label_ds))
    # ds_predict = ds_predict.cache()

    ds = ds_predict.shuffle(buffer_size=1)
    ds = ds.repeat()
    ds = ds.batch(1)
    # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    predictions = model.predict(ds, steps=1)

    output = []
    for p in predictions:
        output.append(LABEL_NAMES[np.argmax(p)])
    return output


#     path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
#     image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
#     label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
#     image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
#     image_label_ds = image_label_ds.cache()
#     return image_label_ds

def Predictions(path, count):
    model = InitModel()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()

    print("Загрузка модели")
    checkpoint_path = "Weights_20/cp.ckpt"
    model.load_weights(checkpoint_path)

    ds_predict = InitDataSet(path)
    ds = ds_predict.shuffle(buffer_size=CountImg(path))
    ds = ds.repeat()
    ds = ds.batch(count)
    # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    predictions = model.predict(ds, steps=1)

    output = []
    for p in predictions:
        # print(p)
        # print(np.argmax(p))
        # print(LABEL_NAMES[np.argmax(p)])
        output.append(LABEL_NAMES[np.argmax(p)])
    return output


def DataPreporationForAugmentation():
    path = pathlib.Path(PATH_TO_DATA)
    data = list(path.glob("*"))
    count = 0

    for item in data:
        temp = list(item.glob("*"))
        if len(temp) > count:
            count = len(temp)

    for item in data:
        temp = list(item.glob("*"))
        coef = int(count / len(temp))
        print(str(item))
        if coef <= 1:
            continue

        if coef > 360:
            coef = 360

        print(count, len(temp), coef)
        DataAugmentation(str(item), coef)

    return


def DataAugmentation(path_str, count):
    path = pathlib.Path(path_str)
    data = list(path.glob("*"))

    for file in data:
        name = str(file).split("\\")
        name = name[len(name) - 1].split(".")
        name = name[0]
        image = cv2.imread(str(file))
        image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)

        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        if count > 360:
            step = 1
        else:
            step = int(360 / count)

        for angle in range(1, 359, step):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            img = np.copy(rotated)
            black = np.where((img[:, :, 0] <= 0) & (img[:, :, 1] <= 0) & (img[:, :, 2] <= 0))
            img[black] = (255, 255, 255)
            cv2.imwrite(path_str + "/a" + name + str(angle) + ".bmp", img)
    return


def VisualImges(path):
    path = pathlib.Path(path)
    all_image_paths = list(path.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)
    label_names = sorted(item.name for item in path.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    img = np.reshape(image_ds)
    logdir = "logs/train_data/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
        tf.summary.image("Training data", img, step=0)


def additional_train(path):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir = "C:\_Programming\BloodClassifier\Weights_20"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    image_label_ds = InitDataSet(path)
    ds = image_label_ds.shuffle(buffer_size=CountImg(path))
    print(CountImg(path))
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` позволяет датасету извлекать пакеты в фоновом режиме, во время обучения модели.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    model = InitModel()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()
    print("Загрузка модели")
    checkpoint_path = "Weights_20/cp.ckpt"
    model.load_weights(checkpoint_path)

    print(CountImg(path) / BATCH_SIZE)
    steps_per_epoch = tf.math.ceil(CountImg(path) / BATCH_SIZE).numpy()

    model.fit(ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[tensorboard_callback, cp_callback],
              verbose=1)
    print("обучение завершенно")


# tf.Session(config=tf.ConfigProto(allow_growth=True))
# VisualImges(r"C:\_Programming\DataSets\temp\TrainDataSet")
# StartTrain(r"C:\_Programming\DataSets\temp\TrainDataSet_20")
# additional_train(r"C:\_Programming\DataSets\temp\TestDataSet_20")
# print(h)
# Visualisation(history)

# checkpoint_path = "Weights_90/cp.ckpt"
# TEST_STEPS = 1000
# TestNeiron(r"C:\_Programming\DataSets\temp\TestDataSet_90")

# DevideData(10, PATH_TO_DATA)

# for i in range(90, 100, 10):
#     PATH_TO_TRAIN = "\TrainDataSet_" + str(i)
#     PATH_TO_TEST = "\TestDataSet_" + str(i)
#     DevideData(i, PATH_TO_DATA)
# ClearTempData()

#
# o = Predictions_2(r"C:\_Programming\DataSets\Predict")

# DataAugmentation(r"C:\_Programming\DataSets\First\Segmented neutrophil")
# DataPreporationForAugmentation()

# DevideData(20, r"C:\_Programming\DataSets\Old\SortedDataJPGOLD")



# for i in range(20, 0, -10):
#     print(i)
#     log_dir = "logs/fit/" + datetime.datetime.now().strftime("train_"+str(i))
#     checkpoint_path = "Weights_"+str(i)+"/cp.ckpt"
#     checkpoint_dir = os.path.dirname(checkpoint_path)
#     # Создаем коллбек сохраняющий веса модели
#     cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                      save_weights_only=True,
#                                                      verbose=1)
#     StartTrain(r"C:\_Programming\DataSets\temp\TrainDataSet_"+str(i),r"C:\_Programming\DataSets\temp\TestDataSet_"+str(i), log_dir)
