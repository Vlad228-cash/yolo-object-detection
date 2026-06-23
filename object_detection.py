import cv2 
import numpy as np 
import requests
import os
from collections import Counter
from matplotlib import pyplot as plt 

def download_file(url, filename):
    if not os.path.exists(filename): 
        print(f"Скачивание: {filename}")
        response = requests.get(url, stream=True) 
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chuck in response.iter_content(chunk_size= 8192):
                if chuck:
                    f.write(chuck)
        print(f"Готово: {filename}")
    else:
        print(f"Уже существует: {filename}")

urls = {
    "ctg" : "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "weights" : "https://pjreddie.com/media/files/yolov3.weights",
    "names" : "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}

download_file(urls["ctg"],"yolov3.cfg")
download_file(urls["weights"],"yolov3.weights")
download_file(urls["names"],"coco.names")

with open("coco.names", "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

path = input("Введи путь к изображению или URL: ")
if path.startswith("https"):
    response = requests.get(path)
    response.raise_for_status()
    img_array = np.asarray(bytearray(response.content), dtype= np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
else:
    img = cv2.imread(path)

if img is None:
    print("Ошибка: изображение не найдено")

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") 
layer_names = net.getLayerNames() 

output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers().flatten()]
height, width, _ = img.shape 
blob = cv2.dnn.blobFromImage(img, 1/ 255.0, (416,416),swapRB=True,crop=False) 
net.setInput(blob) 
outputs = net.forward(output_layers)

boxes = []
confidances = []
class_ids = []

confidance_threshold = 0.3
nms_threshold = 0.4 

for output in outputs:
    for datection in output:
        scores = datection[5:]
        class_id = np.argmax(scores)
        confidance = float(scores[class_id])
        if confidance > confidance_threshold:
            center_x = int(datection[0] * width)
            center_y = int(datection[1] * height)
            w = int(datection[2] * width)
            h = int(datection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x,y,w,h])
            confidances.append(confidance)
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidances, confidance_threshold,nms_threshold)

np.random.seed(42)
colors = np.random.randint(0,255,size=(len(classes),3),dtype="np.int8")

result_img = img.copy()
detected_labels = []

if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidance = confidances[i]
        detected_labels.append(label)
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(result_img, (x,y), (x + w,y + h),color,2)
        text = f"{label}: {confidance:.2f}"
        cv2.putText(result_img, text, (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_COMPLEX,0.6, color, 2)
    
result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB) 

plt.figure(figsize=(12,8))
plt.imshow(result_img)
plt.title("YOLOV детекция объектов")
plt.axis("off") 
plt.show()

print("Найдено объектов:", len(indices) if len(indices) > 0 else 0)

if detected_labels:
    counts = Counter(detected_labels)
    print("\nКоличество объектов по классам:")
    for label, count in counts.items():
        print(f"- {label}: {count}")
else:
    print("Объекты не найдены")