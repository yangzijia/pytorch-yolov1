from torch import nn
import torch.nn.functional as F
import torch

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
        [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    s = 1
    first_flag = True
    for v in cfg:
        s = 1
        if v == 64 and first_flag:
            s = 2
            first_flag = False
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels,
                               v,
                               kernel_size=3,
                               stride=s,
                               padding=1)
            if batch_norm:
                layers += [
                    conv2d,
                    nn.BatchNorm2d(v),
                    nn.LeakyReLU(inplace=True)
                ]
            else:
                layers += [conv2d, nn.LeakyReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.Dropout(),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class YoLo(nn.Module):
    def __init__(self, features, classes_num, bbox_num):
        super(YoLo, self).__init__()
        self.features = features
        self.classes_num = classes_num
        self.bbox_num = bbox_num
        self.classify = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                      nn.Dropout(), nn.LeakyReLU(inplace=True),
                                      nn.Linear(4096, 7 * 7 * (5 * self.bbox_num + self.classes_num)))

    def forward(self, x):
        # print(self.features)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        x = torch.sigmoid(x)
        return x.view(-1, 7, 7, 5 * self.bbox_num + self.classes_num)


def yolov1_model(classes_num, bbox_num, pre_weights_path=None):
    # "weights/vgg16_bn-6c64b313.pth"
    vgg = VGG(make_layers(cfg['D'], batch_norm=True))
    if pre_weights_path:
        vgg.load_state_dict(torch.load(pre_weights_path))
    net = YoLo(vgg.features, classes_num, bbox_num)
    return net


if __name__ == "__main__":
    net = yolov1_model(2, 2)

    data = torch.rand((8, 3, 448, 448))
    rst = net(data)
    print(rst.shape)
