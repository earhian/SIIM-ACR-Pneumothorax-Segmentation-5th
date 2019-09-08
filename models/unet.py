from models.Aspp import *
from models.modelzoo import *
from models.utils import *


class lager_kernel_block(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(lager_kernel_block, self).__init__()
        self.conv_left = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(mid_c, mid_c, kernel_size=(1, 7), padding=(0, 3)),
        )
        self.conv_right = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(mid_c, mid_c, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.conv = nn.Conv2d(mid_c, out_c, kernel_size=1)

    def forward(self, x):
        x_left = self.conv_left(x)
        x_right = self.conv_left(x)
        x = x_left + x_right
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, model_name):
        super(Unet, self).__init__()
        self.model_name = model_name
        self.down = True
        if model_name == 'seresnext50':
            self.basemodel = se_resnext50_32x4d()
            self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
            inplanes = 64
            layer0_modules = [
                ('conv1', nn.Conv2d(1, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
            layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                        ceil_mode=True)))
            self.basemodel.layer0 = nn.Sequential(OrderedDict(layer0_modules))

            self.down1 = nn.Conv2d(256, self.planes[0], kernel_size=1)
            self.down2 = nn.Conv2d(512, self.planes[1], kernel_size=1)
            self.down3 = nn.Conv2d(1024, self.planes[2], kernel_size=1)
            self.down4 = nn.Conv2d(2048, self.planes[3], kernel_size=1)
        if model_name == 'seresnext26':
            self.basemodel = seresnext26_32x4d(pretrained=True)
            self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
            self.startconv = nn.Conv2d(1, 3, kernel_size=1)

            self.down1 = nn.Conv2d(256, self.planes[0], kernel_size=1)
            self.down2 = nn.Conv2d(512, self.planes[1], kernel_size=1)
            self.down3 = nn.Conv2d(1024, self.planes[2], kernel_size=1)
            self.down4 = nn.Conv2d(2048, self.planes[3], kernel_size=1)
        if model_name == 'seresnext101':
            self.basemodel = se_resnext101_32x4d()
            self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
            inplanes = 64
            layer0_modules = [
                ('conv1', nn.Conv2d(1, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
            layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                        ceil_mode=True)))
            self.basemodel.layer0 = nn.Sequential(OrderedDict(layer0_modules))

            self.down1 = nn.Conv2d(256, self.planes[0], kernel_size=1)
            self.down2 = nn.Conv2d(512, self.planes[1], kernel_size=1)
            self.down3 = nn.Conv2d(1024, self.planes[2], kernel_size=1)
            self.down4 = nn.Conv2d(2048, self.planes[3], kernel_size=1)
        if model_name == 'resnet34':
            self.basemodel = resnet34(True)
            self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
        
        if model_name == 'dpn68':
            self.startconv = nn.Conv2d(1, 3, kernel_size=1)
            self.basemodel = dpn68(pretrained=True)

            self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
            self.down1 = nn.Conv2d(144, self.planes[0], kernel_size=1)
            self.down2 = nn.Conv2d(320, self.planes[1], kernel_size=1)
            self.down3 = nn.Conv2d(704, self.planes[2], kernel_size=1)
            self.down4 = nn.Conv2d(832, self.planes[3], kernel_size=1)
        if model_name == 'efficientnet-b5':
            self.startconv = nn.Conv2d(1, 3, kernel_size=1)
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b5')

            self.planes = [256 // 4, 512 // 4, 1024 // 4, 2048 // 4]
            self.down1 = nn.Conv2d(40, self.planes[0], kernel_size=1)
            self.down2 = nn.Conv2d(64, self.planes[1], kernel_size=1)
            self.down3 = nn.Conv2d(176, self.planes[2], kernel_size=1)
            self.down4 = nn.Conv2d(512, self.planes[3], kernel_size=1)

        if model_name == 'efficientnet-b3':
            self.startconv = nn.Conv2d(1, 3, kernel_size=1)
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b3')

            self.planes = [32, 48, 136, 384]
            self.down = False

        if model_name == 'efficientnet-b2':
            self.startconv = nn.Conv2d(1, 3, kernel_size=1)
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b2')

            self.planes = [24, 48, 120, 352]
            self.down = False

        if model_name == 'efficientnet-b1':
            self.startconv = nn.Conv2d(1, 3, kernel_size=1)
            self.basemodel = EfficientNet.from_pretrained('efficientnet-b1')

            self.planes = [24, 40, 112, 320]
            self.down = False

        self.center = ASPP(self.planes[3], self.planes[2])
        self.fc_op = nn.Sequential(
            nn.Conv2d(self.planes[2], 64, kernel_size=1),
            nn.AdaptiveAvgPool2d(1))

        self.fc = nn.Linear(64, 1)
        self.UP4 = UpBlock(self.planes[2], 64, 64)
        self.UP3 = UpBlock(self.planes[2] + 64, 64, 64)
        self.UP2 = UpBlock(self.planes[1] + 64, 64, 64)
        self.UP1 = UpBlock(self.planes[0] + 64, 64, 64)
        self.final = nn.Sequential(
            nn.Conv2d(64 * 4, self.planes[0] // 2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.planes[0] // 2),

            nn.Conv2d(self.planes[0] // 2, self.planes[0] // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.planes[0] // 2),

            nn.UpsamplingBilinear2d(size=(SIZE, SIZE)),

            nn.Conv2d(self.planes[0] // 2, NUM_CLASSES, kernel_size=1)
        )

    def forward(self, x):
        if self.model_name in ['dpn68', 'seresnext26', 'efficientnet-b5', 'efficientnet-b3', 'efficientnet-b2', 'efficientnet-b1']:
            x = self.startconv(x)
        x1, x2, x3, x4 = self.basemodel(x)
        if self.down:
            x1 = self.down1(x1)
            x2 = self.down2(x2)
            x3 = self.down3(x3)
            x4 = self.down4(x4)
        x4 = self.center(x4)
        fc_feat = self.fc_op(x4)
        fc = fc_feat.view(fc_feat.size(0), -1)
        fc = self.fc(fc)
        x4 = self.UP4(x4)
        x3 = self.UP3(torch.cat([x3, x4], 1))
        x2 = self.UP2(torch.cat([x2, x3], 1))
        x1 = self.UP1(torch.cat([x1, x2], 1))
        h, w = x1.size()[2:]
        x = torch.cat(
            [
                # F.upsample_bilinear(fc_feat, size=(h, w)),
                F.upsample_bilinear(x4, size=(h, w)),
                F.upsample_bilinear(x3, size=(h, w)),
                F.upsample_bilinear(x2, size=(h, w)),
                x1
            ],
            1
        )
        return self.final(x), fc
