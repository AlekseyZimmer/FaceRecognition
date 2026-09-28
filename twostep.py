# Распознавание лиц через LBPH (OpenCV 4.x + picamera2)
import cv2
import sys
import os
import time
import numpy as np

# --- Проверка наличия contrib-модуля face ---
if not hasattr(cv2, 'face'):
    print("Ошибка: модуль cv2.face не найден.")
    print("Установите contrib-версию:")
    print("  pip3 install opencv-contrib-python")
    print("или (системная): sudo apt install python3-opencv libopencv-contrib-dev")
    sys.exit(1)

from picamera2 import Picamera2

# --- Поиск Haar-каскада ---
def find_haar(name="haarcascade_frontalface_default.xml"):
    candidates = []
    if hasattr(cv2, 'data'):
        candidates.append(os.path.join(cv2.data.haarcascades, name))
    candidates += [
        f"/usr/share/opencv4/haarcascades/{name}",
        f"/usr/share/opencv/haarcascades/{name}",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

# Используем свой file.xml, если он есть рядом; иначе — встроенный
haar_file = 'file.xml'
if not os.path.isfile(haar_file):
    haar_file = find_haar()
if not haar_file:
    print("Не найден каскад (ни file.xml, ни haarcascade_frontalface_default.xml)")
    sys.exit(1)
print(f"Каскад: {haar_file}")

datasets = 'datasets'
size = 4   # оставлен для совместимости, в OpenCV 4 не используется

print('Recognizing Face. Please be in sufficient light...')

# --- Часть 1: загрузка датасета и обучение LBPH ---
images, labels, names, id_counter = [], [], {}, 0

if not os.path.isdir(datasets):
    print(f"Папка '{datasets}' не найдена. Сначала запустите сбор датасета.")
    sys.exit(1)

for subdir in sorted(os.listdir(datasets)):
    subjectpath = os.path.join(datasets, subdir)
    if not os.path.isdir(subjectpath):
        continue

    names[id_counter] = subdir
    count_for_person = 0
    for filename in sorted(os.listdir(subjectpath)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm')):
            continue
        img = cv2.imread(os.path.join(subjectpath, filename), 0)
        if img is None:
            continue
        images.append(img)
        labels.append(id_counter)
        count_for_person += 1

    print(f"  {subdir}: {count_for_person} изображений")
    id_counter += 1

if len(images) == 0:
    print("Датасет пуст — нечего обучать.")
    sys.exit(1)

print(f"Всего: {len(images)} изображений, {len(names)} человек")

# LBPH требует одинаковый размер для всех изображений
(width, height) = (130, 100)
images = [cv2.resize(img, (width, height)) for img in images]
images = np.array(images)
labels = np.array(labels)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
print("Модель обучена.")

# --- Часть 2: распознавание в реальном времени ---
face_cascade = cv2.CascadeClassifier(haar_file)
if face_cascade.empty():
    print("Каскад не загрузился.")
    sys.exit(1)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)   # прогрев автоэкспозиции

# Порог уверенности LBPH: меньше — лучше. Обычно 50–90.
# Значение нужно подбирать под освещение и качество датасета.
CONFIDENCE_THRESHOLD = 90

logged_names = set()
try:
    while True:
        frame_rgb = picam2.capture_array()
        im = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))

            label, confidence = model.predict(face_resize)

            if confidence < CONFIDENCE_THRESHOLD:
                name = names.get(label, "Unknown")
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
                text = f"{name} - {confidence:.0f}"
                cv2.putText(im, text, (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

                if name not in logged_names:
                    logged_names.add(name)
                    print(f"Log: Зашел: {name} (confidence={confidence:.0f})")
            else:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(im, 'Not in DS', (x - 10, y - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:   # ESC
            break

except KeyboardInterrupt:
    print("\nПрервано пользователем.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Готово.")
