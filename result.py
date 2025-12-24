import cv2
import cvlib as cv
import numpy as npfrom tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image


# %%
model = load_model('model.h5')
model.summary()

# open webcam (웹캠 열기)
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

# loop through frames
while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()
    #영상 전처리
    for sigma in range(1, 4):
        dst = cv2.GaussianBlur(frame, (0, 0), sigma)

    if not status:
        print("Could not read frame")
        exit()

    #얼굴 인식 적용
    face, confidence = cv.detect_face(frame)
    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:


            face_region = frame[startY:endY, startX:endX]
            face_region1 = cv2.resize(face_region, (224, 224), interpolation=cv2.INTER_AREA)


            img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            x = img_to_array(face_region1)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            prediction = model.predict(x)
            predicted_class = np.argmax(prediction[0])  # 예측된 클래스 0, 1, 2
            print(prediction[0])
            print(predicted_class)

            if predicted_class == 0 and (prediction[0][0] * 100) > 0.5:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "LSW ({:.2f}%)".format((prediction[0][0]) * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if predicted_class == 1 and (prediction[0][1] * 100) > 0.5:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "KUM ({:.2f}%)".format((prediction[0][1]) * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if predicted_class == 2 and (prediction[0][2] * 100) > 0.5:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "JSJ ({:.2f}%)".format((prediction[0][2]) * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    # display output
    cv2.imshow("classify", frame)
    cv2.imshow("blur", dst)


    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()