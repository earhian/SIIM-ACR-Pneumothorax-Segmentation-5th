from models.unet import *

class model_iMet(nn.Module):
    def __init__(self, model_name):
        super(model_iMet, self).__init__()

        self.model_name = model_name
        self.model = Unet(model_name)


    def forward(self, x):
        return self.model(x)

    def freeze(self, mode='fc'):
        print('freeze...')
        for p in self.model.basemodel.parameters():
            p.requires_grad = False
        for p in self.model.basemodel.layer2.parameters():
            p.requires_grad = True
        for p in self.model.basemodel.layer3.parameters():
            p.requires_grad = True
        for p in self.model.basemodel.layer4.parameters():
            p.requires_grad = True
        pass

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_loss(self, outs, fc=None, labels=None):
        if fc is None:
            loss = SIZE * nn.BCEWithLogitsLoss(reduce=True)(outs, labels) + SoftDiceLoss()(outs, labels)
            # loss = lovasz_hinge(outs, labels)
        else:
            b = len(outs)
            loss = SIZE * nn.BCEWithLogitsLoss(reduce=True)(outs, labels)
            labels_fc = (labels.view(b, -1).sum(-1) > 0).float().view(b, 1)
            loss += nn.BCEWithLogitsLoss(reduce=True)(fc, labels_fc)
            # loss += 32 * FocalLoss(2.0)(fc, labels_fc)
        return loss

    def get_con_loss(self, outs_hard, fc_hard, outs_simple, fc_simple, gamma=1.0):
        loss = nn.MSELoss()(outs_hard.sigmoid(), outs_simple.sigmoid().detach()) + 0.001 * nn.MSELoss()(fc_hard.sigmoid(), fc_simple.sigmoid().detach())
        return loss * gamma

    # def con_loss(self, outs, fc, outs_ema, fc_ema):
    #     return nn.MSELoss()(outs.sigmoid(), outs_ema.sigmoid().detach()) + 0.1 * nn.MSELoss()(fc.sigmoid(), fc_ema.sigmoid().detach())

    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            try:
                state_dict[key] = pretrain_state_dict[key]
            except:
                print(key)
        self.load_state_dict(state_dict)

# if __name__ == '__main__':
# import torch
# from models.modelzoo.efficientNet import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b1')
# x = torch.rand((4, 3, 224, 224))
# _ = model(x)
