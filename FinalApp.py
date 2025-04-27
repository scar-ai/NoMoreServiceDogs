from PyQt5 import QtCore, QtGui, QtWidgets

import time

import torch
import onnxruntime

import numpy as np
from PIL import Image, ImageDraw

import pyttsx3
import cv2
from torchvision import transforms
from PIL import Image
from environnementModel.dataset import DailyPlacesDataset


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'screen', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]



print("Loading ONNX session...")
onnx_session_envirmt = onnxruntime.InferenceSession(r"environnementModel\model\quantized.onnx", providers=["VitisAIExecutionProvider"],
                                            provider_options=[{"config_file":"vaip_config.json"}])

print("Loading ONNX session...")
onnx_session_objts = onnxruntime.InferenceSession(r"objectsModel\model\quantized.onnx", providers=["VitisAIExecutionProvider"],
                                            provider_options=[{"config_file":"vaip_config.json"}])


print("Aquiring classes")
dataset = DailyPlacesDataset(root_dir="dataset/Places")
labels = dataset.getClasses()

labels = [label.replace('Places__', '').replace('-', ' ') for label in labels]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

istts = False
print('Done, ready to use.')


def text_to_speech(text):
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
        engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0')  # Choose a robotic voice (Mac OS X specific voice)

        engine.say(text)
        engine.runAndWait()


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(290, 591)
        MainWindow.setMinimumSize(QtCore.QSize(290, 591))
        MainWindow.setMaximumSize(QtCore.QSize(290, 591))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(290, 550))
        self.centralwidget.setMaximumSize(QtCore.QSize(290, 550))
        self.centralwidget.setObjectName("centralwidget")
        self.Button_Dec = QtWidgets.QPushButton(self.centralwidget)
        self.Button_Dec.setGeometry(QtCore.QRect(24, 362, 241, 171))
        font = QtGui.QFont()
        font.setFamily("Noto Serif Lao")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.Button_Dec.setFont(font)
        self.Button_Dec.setObjectName("Button_Dec")
        self.Button_Dec.clicked.connect(self.DetectionRun)
        self.Line_Sep1 = QtWidgets.QFrame(self.centralwidget)
        self.Line_Sep1.setGeometry(QtCore.QRect(20, 340, 251, 20))
        self.Line_Sep1.setFrameShape(QtWidgets.QFrame.HLine)
        self.Line_Sep1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.Line_Sep1.setObjectName("Line_Sep1")
        self.Label_Detected = QtWidgets.QLabel(self.centralwidget)
        self.Label_Detected.setGeometry(QtCore.QRect(30, 30, 221, 20))
        font = QtGui.QFont()
        font.setFamily("Noto Kufi Arabic")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.Label_Detected.setFont(font)
        self.Label_Detected.setAlignment(QtCore.Qt.AlignCenter)
        self.Label_Detected.setObjectName("Label_Detected")
        self.Label_Pred = QtWidgets.QLabel(self.centralwidget)
        self.Label_Pred.setGeometry(QtCore.QRect(20, 60, 241, 101))
        font = QtGui.QFont()
        font.setFamily("OpenSymbol")
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.Label_Pred.setFont(font)
        self.Label_Pred.setAlignment(QtCore.Qt.AlignCenter)
        self.Label_Pred.setObjectName("Label_Pred")
        self.Line_Sep3 = QtWidgets.QFrame(self.centralwidget)
        self.Line_Sep3.setGeometry(QtCore.QRect(20, 200, 251, 20))
        self.Line_Sep3.setFrameShape(QtWidgets.QFrame.HLine)
        self.Line_Sep3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.Line_Sep3.setObjectName("Line_Sep3")
        self.Check_tts = QtWidgets.QCheckBox(self.centralwidget)
        self.Check_tts.setGeometry(QtCore.QRect(40, 260, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Sans Serif Collection")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.Check_tts.setFont(font)
        self.Check_tts.setObjectName("Check_tts")
        self.Check_tts.stateChanged.connect(self.TTSactivation)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 290, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


            
    def TTSactivation(self, state):
        if state == 2:
            global istts
            istts = True
        else:
            istts = False
            
    def DetectionRun(self):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame")
            exit()

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        #converting image to torch tensor
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        image_1 = transform(pil_image.resize((224,224))).type(torch.float32).reshape(1, 3, 224, 224).numpy()
        
        image_1 = {"input": image_1}
        t1= time.time()
        evnt_outs = onnx_session_envirmt.run(None, image_1)
        t2= time.time()
        t3= t2-t1
        labels = dataset.getClasses()

        labels = [label.replace('Places__', '').replace('-', ' ') for label in labels]
        evnt_outs = torch.from_numpy(evnt_outs[0]).squeeze(0).argmax(0).item()
        self.Label_Pred.setText(labels[evnt_outs])




        image_2 = pil_image.convert("RGB").resize((1920, 1080))
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = transform(image_2).unsqueeze(0).numpy()

        image2 = {"input": image_tensor}
        t1= time.time()
        objt_outs = onnx_session_objts.run(None, image2)
        t2 = time.time()
        t4 = t2-t1
        print(f'{round(t3+t4, 2)}s')


        draw = ImageDraw.Draw(image_2)

        boxes = objt_outs[0]
        labels_1 = objt_outs[1]
        scores = objt_outs[2]


        objects_dtected = []
        print('------------------------------------------------')
        for box, label, score in zip(boxes, labels_1, scores):
            if score > 0.6:
                objects_dtected.append(f'{COCO_INSTANCE_CATEGORY_NAMES[label]}')
                print(COCO_INSTANCE_CATEGORY_NAMES[label], "-", score)
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
                draw.text((box[0], box[1]),
                          f"Class: {COCO_INSTANCE_CATEGORY_NAMES[label]}\nScore: {round(score.item(), 2)}", fill="red")
        image_2.save("output.jpg")
        image_2.show()
        
        if istts:
            print(labels[evnt_outs])
            finalspeech = ""
            pobj =[]
            for object in objects_dtected:
                if object not in pobj:
                    obj_count = objects_dtected.count(object)
                    if obj_count>1:
                        finalspeech = finalspeech+f"{obj_count} {object}, "
                    else: 
                        finalspeech = finalspeech+f"{object}, "
                    pobj.append(object)
    
            text_to_speech(f'You are in {labels[evnt_outs]} with {finalspeech}')
            print(f'You are in {labels[evnt_outs]} with {finalspeech}')
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "No More Service Dogs"))
        self.Button_Dec.setText(_translate("MainWindow", "RUN DETECTION"))
        self.Label_Detected.setText(_translate("MainWindow", "Detected:"))
        self.Label_Pred.setText(_translate("MainWindow", "Prediction"))
        self.Check_tts.setText(_translate("MainWindow", "Enable Text-To-Speech"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())