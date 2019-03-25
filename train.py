from dataloader import myDataSet
from model import UNet
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import torch.optim as optim
import torchvision.transforms as T
import torch.utils.data as data
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
import utils
from collections import OrderedDict
def train(data_loader, model, epoch, optim, loss_fn, arg):
    if arg.cuda:
        loss_fn = loss_fn.cuda()
    loss_sum = 0

    for i, (x, labels) in enumerate(data_loader):

        x, labels = Variable(x), Variable(labels)
        if arg.cuda:
            x = x.cuda()
            labels = labels.cuda()

        optim.zero_grad()
        out = model(x)
        N = x.shape[0]
        ########
        #out = out.permute(0, 2, 3, 1)
        #m = out.shape[0]
        #out = out.resize(m*800*544, 1)
        #labels = labels.resize(m*800*544)
        ########
        loss = loss_fn(out, labels)
        loss.backward()
       # print(loss.data)
        loss_sum += loss.data
        
        optim.step()
        if i%(len(data_loader)/10)==0:
            #PA = utils.get_PA(out, labels, 0.5)
            print("epoch {}/{}, batch_num {}/{}, loss {}".format(epoch, arg.total_epoch, i, len(data_loader), loss.data))



    print("epoch {}, loss {}".format(epoch, loss_sum.data/len(data_loader)))
    if args.mGPU:
        state = {"epoch": epoch + 1,
                 "state_dict": model.module.state_dict(),
                 "loss": loss_sum.data / len(data_loader)}
    else:
        state={"epoch":epoch+1,
               "state_dict": model.state_dict(),
               "loss": loss_sum.data/len(data_loader)}
    torch.save(state, arg.save_file)


if __name__=="__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--dataroot", default="training_set", help="root of data in training")
    parse.add_argument("--labelroot", default="training_labels", help="path of labels in training")
    parse.add_argument("--batchsize", type=int, default=2, help="data batch size")
    parse.add_argument("--total_epoch", type=int, default=400, help="total epoch for training")
    parse.add_argument("--cuda", action='store_true',  default=True, help='enables cuda')
    parse.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parse.add_argument("--BN", action="store_true", default=True, help="use Batch Normal")
    parse.add_argument('--save_file', default='UNet_3.tar', type=str, help='output checkpoint filename')
    parse.add_argument("--resume", default="", type=str, help='path to latest checkpoint')
    parse.add_argument("--mGPU", default=False, help="use multi-gpus or not")
    args = parse.parse_args()
    print(args)

    #transform
    transform = T.Compose([
        T.ToTensor(),
        #T.Normalize(0.15517, 0.17290)
    ])
    target_trans = T.Compose([
        T.ToTensor()
    ])
    joint_trans = T.Compose([
        T.Resize((544, 800)),
        T.RandomHorizontalFlip(),
    ])
    #dataset
    dataset = myDataSet(args.dataroot, args.labelroot, transform, joint_trans, target_trans)
    loader = data.DataLoader(dataset, args.batchsize, True, num_workers=0)
    #model
    model = UNet(BN=args.BN)
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=>  load checkpoint : {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["state_dict"])
            print("start epoch:{}, loss:{}".format(checkpoint["epoch"], checkpoint["loss"]))
            start_epoch += checkpoint["epoch"]
    if args.cuda:
        model = model.cuda()
    if args.mGPU:
        model = DataParallel(model)
        #cudnn.benchmark = True
    #optimizer = optim.Adagrad(model.parameters(), args.lr)
    optimizer = optim.SGD([
        {'params':[param for name, param in model.named_parameters() if name[-4:]=='bias'],
         'lr': 2*args.lr},
        {'params':[param for name, param in model.named_parameters() if name[-4:]!='bias'],
         'lr': args.lr, 'weight_decay':5e-4}
    ], momentum=0.9)
    model.train()

    #train
    lr = args.lr
    loss_fn = nn.MSELoss()
    #loss_fn = utils.CrossEntropyLoss2d()
    for epoch in range(start_epoch, start_epoch+args.total_epoch):
        train(loader, model, epoch, optimizer, loss_fn, args)
        #
        if (epoch+1)%50==0 and (lr>1e-6):
            lr /= 10
            print("========reset lr to %.8f======" % lr)
            for parameter in optimizer.param_groups:
                parameter["lr"]=lr


