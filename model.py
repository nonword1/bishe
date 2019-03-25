import torch
import torch.nn as nn


def encoder_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, BN=False, dropout=False):
    if BN is True:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
    else:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.ReLU(),
        ]
    if dropout is True:
        layers.append(nn.Dropout())
    return nn.Sequential(*layers)


def decoder_block(in_channels, middle_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, BN=False):
    if BN is True:
        return nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(),
            nn.Conv2d(middle_channels, middle_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size, stride, padding, bias=bias),
            nn.ReLU(),
            nn.Conv2d(middle_channels, middle_channels, kernel_size, stride, padding, bias=bias),
            nn.ReLU(),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
        )

class UNet(nn.Module):
    def __init__(self, BN=False):
        super(UNet, self).__init__()
        self.conv1 = encoder_block(1, 64, BN=BN)
        self.conv2 = encoder_block(64, 128, BN=BN)
        self.conv3 = encoder_block(128, 256, BN=BN)
        self.conv4 = encoder_block(256, 512, BN=BN, dropout=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = decoder_block(512, 1024, 512, BN=BN)
        self.dconv4 = decoder_block(1024, 512, 256, BN=BN)
        self.dconv3 = decoder_block(512, 256, 128, BN=BN)
        self.dconv2 = decoder_block(256, 128, 64, BN=BN)
        self.dconv1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d) :
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0.0)
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.maxpool(conv1))
        conv3 = self.conv3(self.maxpool(conv2))
        conv4 = self.conv4(self.maxpool(conv3))
        center = self.center(self.maxpool(conv4))
        dconv4 = self.dconv4(torch.cat((center, conv4), 1))
        dconv3 = self.dconv3(torch.cat((dconv4, conv3), 1))
        dconv2 = self.dconv2(torch.cat((dconv3, conv2), 1))
        dconv1 = self.dconv1(torch.cat((dconv2, conv1), 1))
        return dconv1


class Discriminator(nn.Module):
    def __init__(self, BN=True):
        super(Discriminator, self).__init__()
        self.conv1_1 = encoder_block(3, 64, BN=BN)
        self.conv1_2 = encoder_block(1, 64, BN=BN)
        self.conv2 = encoder_block(128, 256, BN=BN)
        self.conv3 = encoder_block(256, 512, BN=BN)
        self.fc = nn.Linear()
        self.maxpool = nn.MaxPool2d(2, 2)


    def forward(self, origin, seg):
        origin = self.conv1_1(origin)
        seg = self.conv1_2(seg)
        x = torch.cat(origin, seg)
        x = self.conv2(self.maxpool(x))
        x = self.conv3(self.maxpool(x))
        x = x.fc(x)
        return x




if __name__=="__main__":
    model = UNet()
    for module in model.modules():
        print(module)

