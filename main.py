import tensorflow as tf
import numpy as np
import cv2

# Загрузка модели
model = tf.keras.models.load_model('fingers_model.keras')

def predict_frame(frame):
    # Сохраняем оригинальный размер для отображения
    original_frame = frame.copy()
    
    # Предобработка кадра
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128))
    img_array = tf.keras.utils.img_to_array(frame)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    # Получение предсказания
    predictions = model.predict(img_array)
    # Применяем softmax для получения вероятностей
    score = tf.nn.softmax(predictions[0])
    
    # Получение названий классов
    class_names = ['0', '1', '2', '3', '4', '5']
    
    print("Это изображение, скорее всего, относится к классу {} с вероятностью {:.2f}%"
          .format(class_names[np.argmax(score)], 100 * np.max(score)))
    
    # Вывод вероятностей для всех классов
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {100 * score[i]:.2f}%")
        
    return original_frame

# Тест на одиночном изображении
print("\nТестирование на изображении test.png:")
test_image = cv2.imread('test.png')
if test_image is not None:
    predict_frame(test_image)
    cv2.imshow('Test Image', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Ошибка при загрузке test.png")

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)

# Установка размера кадра как в обучающем наборе (180x180)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка при получении кадра")
        break
        
    # Предсказание для текущего кадра и получение обработанного изображения
    processed_frame = predict_frame(frame)
    
    # Отображение кадра
    cv2.imshow('Webcam', processed_frame)
    
    # Выход при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
