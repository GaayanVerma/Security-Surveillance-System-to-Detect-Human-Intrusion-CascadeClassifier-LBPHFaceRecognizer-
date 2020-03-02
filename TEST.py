from tkinter import *
from tkinter import ttk
import threading
from PIL import ImageTk
root = Tk()
topframe=Frame(root)

root.title('Real Time Security Surveillance')
topframe.pack(side=BOTTOM)


canvas = Canvas(topframe, bg="black", width=700, height=463)
canvas.pack()

photoimage = ImageTk.PhotoImage(file="IMG1.jpg")
canvas.create_image(352, 233, image=photoimage)



L=Label(topframe,text="Information of Detected",font=("Arial Bold", 30));





def recog():


    import cv2
    import numpy as np
    import os
    import pandas as pd
    from cv2.cv2 import CascadeClassifier

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')  # load trained model
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade: CascadeClassifier = cv2.CascadeClassifier(cascadePath);
    df = pd.read_csv("ds.csv")

    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 2
    names = ['', 'gaayan', 'aksshay']
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if (confidence < 100):
                L.config(text=df.iloc[id - 1])
                print(df.iloc[id - 1])
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            L.config(text='Information of Detected')
            break


    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()



def dataset():
    import cv2
    import os

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)


    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face_id = input('\n enter user id end press <return> ==>  ')

    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    count = 0


    while (True):

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1


            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 30:
            break


    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

def training():
    import cv2
    import numpy as np
    from PIL import Image
    import os


    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        return faceSamples, ids

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))


    recognizer.write('trainer/trainer.yml')


    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

def thstart():
    thread = threading.Thread(target=recog, args=())
    thread.start()



L.pack(side=TOP)
button1=Button(topframe, text='FaceCapture',font=("Arial Bold", 30), command= dataset)
button1.pack(side=LEFT,fill=Y)
button2=Button(topframe, text='FaceTrain',font=("Arial Bold", 30), command=training)
button2.pack(side=LEFT,fill=Y)
button3=Button(topframe, text='RecogniseFace',font=("Arial Bold", 30), command= thstart)
button3.pack(side=LEFT,fill=Y)

root.mainloop()
