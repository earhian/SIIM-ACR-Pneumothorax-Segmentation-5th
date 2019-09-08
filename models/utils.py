
from common import *
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(-1, 1)
        target = target.view(-1, 1)

        pt = torch.sigmoid(input)
        pt = 1 - (pt - target.float()).abs()
        logpt = pt.log()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.long().data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class UpBlock(nn.Module):
    def __init__(self, in_c, mid_c, out_c):
        super(UpBlock, self).__init__()
        self.conv = nn.Sequential(
            SEModule(in_c, reduction=16),
            nn.Conv2d(in_c, mid_c, kernel_size=1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
            nn.Conv2d(mid_c, out_c, kernel_size=1),
            nn.ReLU()
        )
        # todo add seblock!

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    input = torch.rand((4, 1, 128))
    target = torch.ones((4, 1, 128))
    loss = FocalLoss(2, 0.2)(input, target)
    print(loss.item())
