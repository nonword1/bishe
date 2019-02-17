import cv2
import numpy as np
import os

def getlabel(imgfile,writefile):
    img = cv2.imread(imgfile, 0)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=-1)
    cv2.imwrite(writefile, img)

path = "training_set"
files = os.listdir(path)
ss = []
for file in files:
    if "Annotation" in file:
        ss.append(file)
print(ss)
for s in ss:
    pre = s.split(".", 1)[0]
    imgfile = "training_set/" + s
    writefile = "training_labels/"+pre+"_label.png"
    getlabel(imgfile, writefile)









