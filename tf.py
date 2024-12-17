import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from tensorflow.keras.regularizers import l2

# Включаем оптимизацию для Apple Silicon
tf.config.set_visible_devices([], 'GPU')  # Отключаем GPU для использования Apple Metal
tf.config.list_physical_devices('GPU')  # Проверяем доступные GPU

# Параметры
IMG_SIZE = 128
BATCH_SIZE = 32  # Увеличиваем размер батча для M1
EPOCHS = 15

# Загрузка и подготовка данных
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/train',
    labels='inferred',
    label_mode='int',
    class_names=['0', '1', '2', '3', '4', '5'],
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Изменяем способ разделения данных
train_size = int(0.7 * len(train_ds))  # Уменьшаем до 70%
val_size = int(0.15 * len(train_ds))   # 15% на валидацию
test_size = len(train_ds) - train_size - val_size  # 15% на тест

val_ds = train_ds.skip(train_size).take(val_size)
test_ds = train_ds.skip(train_size + val_size)
train_ds = train_ds.take(train_size)

# Оптимизация производительности для M1
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Нормализация данных с использованием mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')  # Включаем смешанную точность
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Создание модели с оптимизацией для M1
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Слои аугментации
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    
    # Оптимизированная архитектура для M1
    layers.Conv2D(64, 3, padding='same', activation='relu', use_bias=False),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    layers.Conv2D(128, 3, padding='same', activation='relu', use_bias=False),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    layers.Conv2D(256, 3, padding='same', activation='relu', use_bias=False),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(6, activation='softmax')
])

# Оптимизатор с поддержкой mixed precision
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

# Компиляция модели
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
    jit_compile=True  # Включаем XLA компиляцию для M1
)

# Ранняя остановка для предотвращения переобучения
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Уменьшение скорости обучения при плато
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

# Обучение модели
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# Оценка модели на тестовом наборе
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\nТочность на тестовом наборе: {test_accuracy:.2f}")

# Сохранение модели
model.save('fingers_model.keras')
print("\nМодель сохранена как 'fingers_model.keras'")

# Построение графиков обучения
plt.figure(figsize=(12, 4))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность на обучающем наборе')
plt.plot(history.history['val_accuracy'], label='Точность на проверочном наборе')
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
plt.show()
