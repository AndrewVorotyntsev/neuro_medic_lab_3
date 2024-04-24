import os

import numpy as np
from keras import Sequential
from keras.src.applications.resnet import ResNet50
from keras.src.layers import Flatten, Dense, Dropout, MaxPooling2D
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from keras.src.utils import load_img, img_to_array

import datetime

current_dir = os.path.dirname(__file__)

train_dir = os.path.join(current_dir, "train")
test_dir = os.path.join(current_dir, "test")

resolution = (224, 224)
# TODO:
#  использовать нативное разрешение
#  п.3)
#    rescale
#  поиграться со слоями

# Создание генераторов для загрузки изображений
# Создание генераторов для загрузки изображений
train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.4)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=resolution,
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=resolution,
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Загрузка предобученной модели ResNet50 без полносвязанного слоя
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(resolution + (3,)))

# Добавление нового полносвязанного слоя
model = Sequential()
model.add(base_model)
model.add(Flatten())
# model.add(Dense(128, activation='tanh'))
# model.add(Dense(256, activation='relu'))
# model.add(MaxPooling2D(4, 4), 4)
# model.add(Dropout(rate=0.5))
model.add(Dense(12, activation='softmax'))  # 12 - количество классов

# Замораживаем веса предобученной части модели
for layer in base_model.layers:
    layer.trainable = False

optimizer = Adam(learning_rate=0.0001)

# Компиляция модели
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'],)

# Обучение модели
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples,
    epochs=10)

test_files = os.listdir(test_dir)

class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
# Открытие файла для записи результатов классификации
log_path = '{date:%Y-%m-%d_%H_%M_%S}.txt'.format(date=datetime.datetime.now())
with open(log_path, 'w') as f:
    f.write("File:    Result of classification:\n")
    # Применение модели к каждому изображению из папки test
    for filename in test_files:
        img_path = os.path.join(test_dir, filename)
        img = load_img(img_path, target_size=resolution)
        img_array = img_to_array(img)
        img_array = img_array.reshape(((1,) + resolution + (3,)))
        img_array = img_array / 255.0  # Масштабирование значений пикселей до [0, 1]

        # Получение предсказания для изображения
        prediction = model.predict(img_array)

        # Преобразование предсказания в метку класса
        class_index = np.argmax(prediction)
        class_label = class_names[class_index]  # Предполагается, что у вас есть список с названиями классов

        # Запись результатов в файл
        f.write(f"{filename}\t{class_label}\n")
        print(f"{filename}\t{class_label}\n")
