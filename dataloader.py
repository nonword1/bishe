import os
import torch
import torchvision
import numpy as np
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
class myDataSet(data.Dataset):
    def __init__(self, imagepath, labelpath, transform=None):
        super(myDataSet, self).__init__()
        self.imagepath = imagepath
        self.labelpath = labelpath
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

        if self.transform is not None:
            img = self.transform(image)
            label = self.transform(label)
        return img, label

if __name__ == '__main__':
    imagepath = "training_set"
    labelpath = "training_labels"
    transform = T.Compose([
        T.ToTensor()
    ])
    dst = myDataSet(imagepath, labelpath, transform)
    trainloader = data.DataLoader(dst, batch_size=1)

    for i, da in enumerate(trainloader):
        imgs, labels = da
        if i%1 == 0:
            imgs, labels = imgs.squeeze(0), labels.squeeze(0)
            img = T.ToPILImage()(imgs)
            label = T.ToPILImage()(labels)
            plt.subplot(2,1,1)
            plt.imshow(img)
            plt.subplot(2,1,2)
            plt.imshow(label)
            plt.pause(1)

