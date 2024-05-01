import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO
net = cv2.dnn.readNetFromDarknet("C:/Users/Acer/Desktop/Object Detection using YOLO & Tensorflow/ObjectDetection/yolov3.cfg", "C:/Users/Acer/Desktop/Object Detection using YOLO & Tensorflow/ObjectDetection/yolov3.weights")

classes = []
with open("C:/Users/Acer/Desktop/Object Detection using YOLO & Tensorflow/ObjectDetection/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# List of image paths
image_paths = [
    r"C:/Users/Acer/Desktop/Object Detection using YOLO & Tensorflow/ObjectDetection/images/example1.jpg",
    r"C:/Users/Acer/Desktop/Object Detection using YOLO & Tensorflow/ObjectDetection/images/example2.jpg",
    r"C:/Users/Acer/Desktop/Object Detection using YOLO & Tensorflow/ObjectDetection/images/example3.jpg"
]

for image_path in image_paths:
    # Load image
    img = cv2.imread(image_path)

    if img is not None:
        # Resize image
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        # Show the detected objects
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
        plt.show()
    else:
        print(f"Error: Unable to load image '{image_path}'.")