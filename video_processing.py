import cv2
import os
import math
import time
import numpy as np
import cvzone
from ultralytics import YOLO
from sort import Sort

def calculate_speed(carPositions, frame_timestamp, frame_rate):
    for id, carPosition in carPositions.items():
        if 'prev_position' in carPosition:
            prev_position = carPosition['prev_position']
            curr_position = carPosition['position']
            distance = math.sqrt((curr_position[0] - prev_position[0]) ** 2 + (curr_position[1] - prev_position[1]) ** 2)
            speed_mps = distance / (1 / frame_rate)
            speed_kmph = speed_mps * 3.6
            carPositions[id]['speed'] = speed_kmph
            carPositions[id]['timestamp'] = frame_timestamp
            carPositions[id]['prev_position'] = curr_position
    return carPositions

def process_video(video_file_location, speed_checkbox_value, classname_checkbox_value, outgoing_checkbox_value,
                  oncoming_checkbox_value):
    cap = cv2.VideoCapture(video_file_location)
    model = YOLO("../Yolo-weights/yolov8n.pt")

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
                  "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                  "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    current_directory = os.path.dirname(os.path.abspath(__file__))
    mask_path = os.path.join(current_directory, "mask.png")
    mask = cv2.imread(mask_path)

    sort = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    carPositions = {}
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    while True:
        success, img = cap.read()
        imgRegion = cv2.bitwise_and(img, mask)
        graphics_path = os.path.join(current_directory, "graphics.png")
        imgGraphics = cv2.imread(graphics_path, cv2.IMREAD_UNCHANGED)
        img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

        results = model(imgRegion, stream=True)

        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box.xyxy[0], int(box.cls[0]), box.conf[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                if classname_checkbox_value == "":
                    if speed_checkbox_value:
                        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(255, 0, 0))
                    else:
                        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(0, 255, 0))
                else:
                    if classNames[int(cls)] == classname_checkbox_value:
                        if speed_checkbox_value:
                            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(255, 0, 0))
                        else:
                            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(0, 255, 0))

                if speed_checkbox_value:
                    cx, cy = x1 + ((x2 - x1) // 2), y1 + ((y2 - y1) // 2)
                    cvzone.putTextRect(img, f'{classNames[int(cls)]}', (max(0, x1), max(35, y1)), scale=1,
                                       thickness=1, offset=3)
                    cvzone.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                    if classNames[int(cls)] == "car" or classNames[int(cls)] == "truck":
                        detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        resultsTracker = sort.update(detections)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + (w // 2), y1 + (h // 2)

            if outgoing_checkbox_value:
                if cy > 0 and cy < 1000:
                    if id in carPositions:
                        if "outgoing" in carPositions[id]:
                            cvzone.putTextRect(img, f'Outgoing: {len(carPositions)}', (50, 50), scale=2,
                                               thickness=2, offset=10)
                    else:
                        carPositions[id] = {"position": (cx, cy), "timestamp": time.time(), "prev_position": (cx, cy)}
                else:
                    if id in carPositions:
                        if "outgoing" not in carPositions[id]:
                            carPositions[id]["outgoing"] = True
                            cv2.line(img, (cx, cy), (cx, cy), (0, 0, 255), 5)
            elif oncoming_checkbox_value:
                if cy > 0 and cy < 1000:
                    if id in carPositions:
                        if "oncoming" in carPositions[id]:
                            cvzone.putTextRect(img, f'Oncoming: {len(carPositions)}', (50, 50), scale=2,
                                               thickness=2, offset=10)
                    else:
                        carPositions[id] = {"position": (cx, cy), "timestamp": time.time(), "prev_position": (cx, cy)}
                else:
                    if id in carPositions:
                        if "oncoming" not in carPositions[id]:
                            carPositions[id]["oncoming"] = True
                            cv2.line(img, (cx, cy), (cx, cy), (0, 0, 255), 5)

            # Calculate speed
            frame_timestamp = time.time()
            carPositions = calculate_speed(carPositions, frame_timestamp, frame_rate)

            if speed_checkbox_value:
                if id in carPositions:
                    speed = carPositions[id]["speed"]
                    cvzone.putTextRect(img, f'{speed} km/h', (max(0, x1), max(35, y1)), scale=1,
                                      thickness=1, offset=3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)