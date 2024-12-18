import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, Sequential
import numpy as np

# Параметры
batch_size = 32
epochs = 200
img_width = 128
img_height = 128

train_dataset_dir = 'data/train'
validation_dataset_dir = 'data/validation'
test_dataset_dir = 'data/test'

# Загрузка тренировочного датасета с разделением на train и validation
train_ds = tf.keras.utils.image_dataset_from_directory(
	train_dataset_dir,
	subset="training", 
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
	validation_dataset_dir,
	subset="validation",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

# Загрузка тестового датасета
test_ds = tf.keras.utils.image_dataset_from_directory(
	test_dataset_dir,
	image_size=(img_height, img_width),
	batch_size=batch_size)

class_names = train_ds.class_names
print(f"Названия классов: {class_names}")

# Оптимизация производительности с помощью кэширования и предварительной выборки
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Создание модели
num_classes = len(class_names)

# Добавим class weights для баланса классов
class_weights = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 1.0
}

# Сначала проверим баланс классов
for images, labels in train_ds:
    unique, counts = np.unique(labels, return_counts=True)
    print("Распределение классов в датасете:", dict(zip(unique, counts)))
    break

# Настраиваем более сбалансированную модель
model = Sequential([
	# Нормализация входных данных
	layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
	
	# Более интенсивная аугментация для лучшей генерализации
	layers.RandomFlip("horizontal_and_vertical"),
	layers.RandomRotation(0.1),
	layers.RandomZoom(0.2),
	layers.RandomTranslation(0.05, 0.05),
	layers.RandomBrightness(0.01),
	
	# Первый блок свертки
	layers.Conv2D(32, 5, padding='same'),  # Увеличиваем размер ядра
	layers.BatchNormalization(),
	layers.Activation('relu'),
	layers.MaxPooling2D(pool_size=(2, 2)),
	layers.Dropout(0.25),
	
	# Второй блок свертки
	layers.Conv2D(64, 3, padding='same'),
	layers.BatchNormalization(),
	layers.Activation('relu'),
	layers.MaxPooling2D(pool_size=(2, 2)),
	layers.Dropout(0.25),
	
	# Третий блок свертки
	layers.Conv2D(128, 3, padding='same'),
	layers.BatchNormalization(),
	layers.Activation('relu'),
	layers.MaxPooling2D(pool_size=(2, 2)),
	layers.Dropout(0.25),
	
	layers.Flatten(),
	layers.Dense(256),
	layers.BatchNormalization(),
	layers.Activation('relu'),
	layers.Dropout(0.5),
	layers.Dense(num_classes, activation='softmax'),
	
    layers.Flatten(),
	layers.Dense(512),
	layers.BatchNormalization(),
	layers.Activation('relu'),
	layers.Dropout(0.5),
	layers.Dense(num_classes, activation='softmax'),
])

# Настраиваем оптимизатор с L2 регуляризацией
optimizer = tf.keras.optimizers.Adam(
	learning_rate=0.0003,  # Уменьшаем learning rate
	beta_1=0.9,
	beta_2=0.999,
	epsilon=1e-07,
	weight_decay=1e-5  # Добавляем L2 регуляризацию
)

model.compile(
	optimizer=optimizer,
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
	metrics=['accuracy']
)

# Настраиваем callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
	monitor='val_loss',
	patience=15,
	restore_best_weights=True,
	min_delta=0.001
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
	monitor='val_loss',
	factor=0.2,
	patience=7,
	min_lr=0.00001,
	verbose=1
)

# Обучение модели
history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=epochs,
	callbacks=[early_stopping, reduce_lr],
	class_weight=class_weights
)

# После обучения добавим вывод распределения предсказаний
def evaluate_class_distribution(model, dataset):
	predictions = []
	for images, _ in dataset:
		pred = model.predict(images)
		predictions.extend(pred)
	
	predictions = np.array(predictions)
	class_distribution = np.mean(predictions, axis=0)
	print("\nСреднее распределение предсказаний по классам:")
	for i, prob in enumerate(class_distribution):
		print(f"Класс {i}: {prob:.4f}")

# Оцениваем распределение на валидационном наборе
evaluate_class_distribution(model, val_ds)

# Оценка на тестовом наборе
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nТочность на тестовом наборе: {test_accuracy:.4f}")

# Сохранение
model.save('fingers_model.keras')
print("\nМодель сохранена как 'fingers_model.keras'")

# Построение графиков обучения
plt.figure(figsize=(12, 4))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность на обучающем наборе')
plt.plot(history.history['val_accuracy'], label='��очность на проверочном наборе')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.title('График точности')
plt.legend()

# График функции потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери на обучающем наборе')
plt.plot(history.history['val_loss'], label='Потери на проверочном наборе')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.title('График функции потерь')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()
