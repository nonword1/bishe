from dataloader import testDataSet
from model import UNet
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
from collections import OrderedDict
import torch.optim as optim
import torchvision.transforms as T
import torch.utils.data as data
import matplotlib.pyplot as plt

if __name__=="__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataroot", default="val_set", help="root of data in training")
    parse.add_argument("--batchsize", type=int, default=1, help="data batch size")
    parse.add_argument("--total_epoch", type=int, default=50, help="total epoch for training")
    parse.add_argument("--cuda", action='store_true', default=False, help='enables cuda')
    parse.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parse.add_argument("--BN", action="store_true", default=True, help="use Batch Normal")
    parse.add_argument('--save_file', default='UNet_2.tar', type=str, help='output checkpoint filename')
    parse.add_argument("--mGPU", default=False, help="use multi-gpus or not")
    #parse.add_argument("resume", default="", type=str, help='path to latest checkpoint')
    args = parse.parse_args()
    print(args)


    model = UNet(BN=args.BN)
    if args.cuda:
        model = model.cuda()
    checkpoint = torch.load(args.save_file, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    transforms = T.Compose([
        T.Resize((544, 800)),
        T.ToTensor()
    ])

    test_set = testDataSet(args.dataroot, transforms)
    testloader = data.DataLoader(test_set, args.batchsize, shuffle=False)

    dirn = "val_result"
    if not os.path.exists(dirn):
        os.mkdir(dirn)

    for i, da in enumerate(testloader):
        img, name = da
        img_name = name[0]+".png"
        if args.cuda:
            img = img.cuda()
        out = model(img)
        out = out.squeeze(0)
        prediction = T.ToPILImage()(out)
        path = os.path.join(dirn, img_name)
        prediction.save(path)


