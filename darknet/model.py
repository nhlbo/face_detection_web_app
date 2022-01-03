import base64
import time
from binascii import b2a_base64

import cv2
import numpy as np

net = None
LABELS = None
COLORS = None

net_2 = None
LABELS_2 = None
COLORS_2 = None

LABELS_FILE = 'darknet/data/coco.names'
CONFIG_FILE = 'darknet/cfg/yolov3_320x320.cfg'
WEIGHTS_FILE = 'darknet/weights/yolov3.weights'

LABELS_FILE_2 = 'darknet/data/coco.names'
CONFIG_FILE_2 = 'darknet/cfg/yolov3_320x320.cfg'
WEIGHTS_FILE_2 = 'darknet/weights/yolov3.weights'


def load_model():
    global net, LABELS, COLORS, net_2, LABELS_2, COLORS_2
    np.random.seed(4)
    LABELS = open(LABELS_FILE).read().strip().split("\n")
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

    LABELS_2 = open(LABELS_FILE_2).read().strip().split("\n")
    COLORS_2 = np.random.randint(0, 255, size=(len(LABELS_2), 3), dtype="uint8")
    net_2 = cv2.dnn.readNetFromDarknet(CONFIG_FILE_2, WEIGHTS_FILE_2)


def detect(img, confidence_threshold):
    img_base64 = ''
    for chunk in img.chunks():
        img_base64 += b2a_base64(chunk).decode().strip()

    im_bytes = base64.b64decode(img_base64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold,
                            confidence_threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text.decode(), end - start
