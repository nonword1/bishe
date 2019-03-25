import os
import torch
import torchvision
import numpy as np
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
from model import UNet
import utils
import cv2
class myDataSet(data.Dataset):
    def __init__(self, imagepath, labelpath, transform=None, joint_trans=None, target_trans=None):
        super(myDataSet, self).__init__()
        self.imagepath = imagepath
        self.labelpath = labelpath
        self.transform = transform
        self.joint_trans = joint_trans
        self.target_trans = target_trans
        files = os.listdir(imagepath)
        self.img_id = []
        for file in files:
            if "Annotation" not in file:
                pre = file.split(".", 1)[0]
                self.img_id.append(pre)
        self.files = []
        for name in self.img_id:
            img_file = os.path.join(self.imagepath, "%s.png" % name)
            label_file = os.path.join(self.labelpath, "%s_Annotation_label.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        datafiles = self.files[item]
        name = datafiles["name"]
        image = Image.open(datafiles["img"]).convert("L")
        label = Image.open(datafiles["label"]).convert("L")
        label = label.point(lambda x: 255 if x > 127 else 0)
        label = label.convert('1')
        if self.joint_trans is not None:
            image = self.joint_trans(image)
            label = self.joint_trans(label)
        if self.transform is not None:
            img = self.transform(image)
        if self.target_trans is not None:
            label = self.transform(label)
        return img, label


class testDataSet(data.Dataset):
    def __init__(self, imagepath, transform=None):
        super(testDataSet, self).__init__()
        self.imagepath = imagepath
        self.transform = transform
        files = os.listdir(imagepath)
        self.img_id = []
        for file in files:
            if "Annotation" not in file:
                pre = file.split(".", 1)[0]
                self.img_id.append(pre)
        self.files = []
        for name in self.img_id:
            img_file = os.path.join(self.imagepath, "%s.png" % name)
            self.files.append({
                "img": img_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        datafiles = self.files[item]
        name = datafiles["name"]
        image = Image.open(datafiles["img"]).convert("L")
        if self.transform is not None:
            img = self.transform(image)
        return img, name


if __name__ == '__main__':
    imagepath = "val_set"
    labelpath = "val_labels"
    transform = T.Compose([
        T.ToTensor()
    ])
    target_trans = T.Compose([
        T.ToTensor()
    ])
    joint_trans = T.Compose([
        T.Resize((544, 800))
    ])
    dst = myDataSet(imagepath, labelpath, transform, joint_trans, target_trans)
    trainloader = data.DataLoader(dst, batch_size=1)
    model = UNet(BN=True)
    print("load the checkpoint : UNet_2.tar")
    save_file = "UNet_2.tar"
    checkpoint = torch.load(save_file, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    PA = 0
    dirn = "val_result"
    if not os.path.exists(dirn):
        os.mkdir(dirn)
    for i, da in enumerate(trainloader):
        imgs, labels = da
        out = model(imgs)
        a, b, c, d = imgs.size()
        imgs, labels, out = imgs.squeeze(0), labels.squeeze(0), out.squeeze(0)
        # img = T.ToPILImage()(imgs)
        # label = T.ToPILImage()(labels)
        # out = T.ToPILImage()(out)
        # plt.subplot(3, 1, 1)
        # plt.imshow(img)
        # plt.subplot(3, 1, 2)
        # plt.imshow(label)
        # plt.subplot(3, 1, 3)
        # plt.imshow(out)
        # plt.pause(10)




