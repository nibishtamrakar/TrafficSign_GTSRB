import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.70  # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRAINED MODEL
model = keras.models.load_model('my_model14.h5')

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def getClassName(classRes):
    labels = {0: 'Speed limit(20km/h)',
              1: 'Speed limit(30km/h)',
              2: 'Speed limit(50km/h)',
              3: 'Speed limit(60km/h)',
              4: 'Speed limit(70km/h)',
              5: 'Speed limit(80km/h)',
              6: 'End of speed limit(80km/h)',
              7: 'Speed limit(100km/h)',
              8: 'Speed limit(120km/h)',
              9: 'No passing',
              10: 'No passing for vehicles over 3.5 metric tons',
              11: 'Right-of-way at the next intersection',
              12: 'Priority road',
              13: 'Yield',
              14: 'Stop',
              15: 'No vehicles',
              16: 'Vehicles over 3.5 metric tons prohibited',
              17: 'No entry',
              18: 'General caution',
              19: 'Dangerous curve to the left',
              20: 'Dangerous curve to the right',
              21: 'Double curve',
              22: 'Bumpy road',
              23: 'Slippery road',
              24: 'Road narrows on the right',
              25: 'Road work',
              26: 'Traffic signals',
              27: 'Pedestrians',
              28: 'Children crossing',
              29: 'Bicycles crossing',
              30: 'Beware of ice/snow',
              31: 'Wild animals crossing',
              32: 'End of all speed and passing limits',
              33: 'Turn right ahead',
              34: 'Turn left ahead',
              35: 'Ahead only',
              36: 'Go straight or right',
              37: 'Go straight or left',
              38: 'Keep right',
              39: 'Keep left',
              40: 'Roundabout mandatory',
              41: 'End of no passing',
              42: 'End of no passing by vehicles over 3.5 metric tons'}
    return labels[classRes]

while True:
    # READ IMAGE
    success, imgOrignal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_step(img)
    myClassName = getClassName(np.argmax(classIndex))

    probabilityValue = np.amax(predictions)

    # BOUNDING BOX DETECTION (simple color-based approach for red signs)
    hsv = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_prob = 0
    best_bbox = None
    best_class_name = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Threshold to remove small areas
            x, y, w, h = cv2.boundingRect(cnt)
            cropped_img = imgOrignal[y:y+h, x:x+w]
            if cropped_img.size == 0:
                continue
            cropped_img = cv2.resize(cropped_img, (32, 32))
            cropped_img = preprocessing(cropped_img)
            cropped_img = cropped_img.reshape(1, 32, 32, 1)

            pred = model.predict(cropped_img)
            prob = np.amax(pred)
            class_idx = model.predict_step(cropped_img)
            class_name = getClassName(np.argmax(class_idx))

            if prob > max_prob:
                max_prob = prob
                best_bbox = (x, y, w, h)
                best_class_name = class_name

    if max_prob > threshold and best_bbox is not None:
        x, y, w, h = best_bbox
        cv2.rectangle(imgOrignal, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(imgOrignal, best_class_name, (x, y-10), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(max_prob * 100, 2)) + "%",
                    (x, y + h + 20), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
