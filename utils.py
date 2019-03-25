import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import os
import matplotlib.pyplot as plt

def get_PA(pred_batch, label_batch,th):
    '''get pixel accuracy'''
    label_batch = label_batch.cpu().data.numpy()
    pred_batch = pred_batch.cpu().data.numpy()
    N = label_batch.shape[0]
    PA = 0
    for n in range(N):
        pii = 0
        pred = pred_batch[n].flatten()
        label = label_batch[n].flatten()
        pred = pred > th
        pred = np.uint8(pred * 1)
        for i in range(len(label)):
            if pred[i]==label[i]:
                pii += 1
        pa = pii / len(label)
        PA += pa
    PA = PA / N
    return PA


def get_mean_std(path="training_set"):
    files = os.listdir(path)
    ss = []
    imgs = np.zeros([800, 544, 1])
    for file in files:
        if "Annotation" not in file:
            ss.append(file)
    for s in ss:
        img_path = path + "/" + s

        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (544, 800))
        img = img[:, :, np.newaxis]

        imgs = np.concatenate((imgs, img), axis=2)
    imgs = imgs.astype(np.float32) / 255
    pixels = imgs.ravel()
    means = np.mean(pixels)
    std = np.std(pixels)

    print("Mean:{}; std:{}".format(means, std))


def fit_elipse(img):
    edge = cv2.Canny(img, 10, 150)
    y, x = np.nonzero(edge)
    edge_list = np.array([[_x, _y] for _x,_y in zip(x, y)])

    elipse = cv2.fitEllipse(edge_list)
    return elipse

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


if __name__=="__main__":
    path = "training_set/000_HC_Annotation.png"
    img = cv2.imread(path, 0)
    el = fit_elipse(img)
    print(el)
    img_clone = img.copy()
    cv2.ellipse(img_clone, el,255,1)
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.imshow(img_clone-img)
    plt.pause(10)

