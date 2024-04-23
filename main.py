import os

import numpy as np
from keras import Sequential
from keras.src.applications.resnet import ResNet50
from keras.src.layers import Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import load_img, img_to_array

current_dir = os.path.dirname(__file__)

train_dir = os.path.join(current_dir, "train")
test_dir = os.path.join(current_dir, "test")

resolution = (1440, 600)
# TODO:
#  использовать нативное разрешение
#  п.3)
#    rescale
#  поиграться со слоями

# Создание генераторов для загрузки изображений
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=resolution,
    batch_size=32,
    interpolation='nearest',
    class_mode='categorical',
    subset='training')

# validation_generator = train_datagen.flow_from_directory(
#     test_dir,
#     target_size=resolution,
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation')

# Загрузка предобученной модели ResNet50 без полносвязанного слоя
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(resolution + (3,)))

# Добавление нового полносвязанного слоя
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))  # Добавление полносвязанного слоя с 256 нейронами
model.add(Dense(12, activation='softmax'))  # 12 - количество классов в вашем наборе данных

# Замораживаем веса предобученной части модели
for layer in base_model.layers:
    layer.trainable = False

# Компиляция модели
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples,
    epochs=5)

test_files = os.listdir(test_dir)

class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
# Открытие файла для записи результатов классификации
with open('classification_results.txt', 'w') as f:
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
