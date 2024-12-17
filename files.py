import os
import shutil

def organize_test_images():
    # Пути к директориям
    base_dir = os.path.dirname(__file__)
    test_source_dir = os.path.join(base_dir, 'dataset', 'train')
    
    # Создаем временную директорию для хранения оригинальных файлов
    temp_dir = os.path.join(test_source_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Перемещаем все файлы во временную директорию
    files = [f for f in os.listdir(test_source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for file in files:
        source_path = os.path.join(test_source_dir, file)
        temp_path = os.path.join(temp_dir, file)
        shutil.move(source_path, temp_path)
    
    # Создаем подпапки для каждого класса (0-5 пальцев)
    for i in range(6):
        os.makedirs(os.path.join(test_source_dir, str(i)), exist_ok=True)
    
    # Распределяем файлы по папкам
    for filename in files:
        # Извлекаем количество пальцев из имени файла
        if 'L' in filename:
            num_fingers = filename.split('L')[0][-1]
        else:
            num_fingers = filename.split('R')[0][-1]
        
        # Перемещаем файл в соответствующую поддиректорию
        source_path = os.path.join(temp_dir, filename)
        target_path = os.path.join(test_source_dir, str(num_fingers), filename)
        shutil.move(source_path, target_path)
    
    # Удаляем временную директорию
    os.rmdir(temp_dir)
    
    print(f"Организация тестовых файлов завершена. Всего файлов: {len(files)}")

if __name__ == "__main__":
    organize_test_images()