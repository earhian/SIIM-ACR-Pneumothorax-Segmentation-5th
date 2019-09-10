import datetime
import time
from timeit import default_timer as timer
import os, argparse
import torch.utils.data.sampler as torchSampler
import torchvision.utils as vutils
# image_size = 128
from tensorboardX import SummaryWriter

from dataSet.reader import *
from dataSet.transform import *
from models.model import *
from utils.file import *
from utils.metric import *

torch.backends.cudnn.benchmark = True
writer = SummaryWriter('')

lam = 0


def transform_train(image, mask):
    image, mask = random_scale(image, mask, center_p=0.5)
    if random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if 1:
        image, mask = random_erase(image, mask, p=0.5)
    if 1:
        image, mask = random_angle_rotate(image, mask, angles=(-25, 25))
    if random.random() < 0.5:
        alpha = random.uniform(0.95, 1.05)
        image = do_brightness_multiply(image, alpha)

    image = torch.from_numpy(image).float().div(255)
    image = image.unsqueeze(0)
    mask = (torch.from_numpy(mask).float().div(255) > 0.5).float()
    mask = mask.unsqueeze(0)
    return image, mask

def transform_test(image):
    mask = np.zeros_like(image)
    image, mask = random_scale(image, mask, center_p=0.5)
    if random.random() < 0.5:
        image = np.fliplr(image)
    if 1:
        image, mask = random_erase(image, mask, p=0.5)
    if 1:
        image, mask = random_angle_rotate(image, mask, angles=(-25, 25))
    image_hard = image.copy()
    image_simple = image.copy()
    if random.random() < 1:
        alpha = random.uniform(0.9, 1.1)
        image_hard = do_brightness_multiply(image_hard, alpha)
    # if random.random() < 0.8:
    #     alpha = random.uniform(0.05, 0.2)
    #     image_hard = do_gaussian_noise(image_hard, alpha)

    image_hard = torch.from_numpy(image_hard).float().div(255)
    image_hard = image_hard.unsqueeze(0)
    image_simple = torch.from_numpy(image_simple).float().div(255)
    image_simple = image_simple.unsqueeze(0)

    return image_hard, image_simple


def transform_valid(image, mask):
    image, mask = random_scale(image, mask, size_=SIZE)

    image = torch.from_numpy(image).float().div(255)
    image = image.unsqueeze(0)
    mask = (torch.from_numpy(mask).float().div(255) > 0.5).float()
    mask = mask.unsqueeze(0)

    return image, mask


def drawing(images, results1, results2, mode='train'):
    images = images.data.cpu()[:16]
    results1 = (results1 > 0.5).float()[:16]
    results2 = (results2 > 0.5).float()[:16]
    images = vutils.make_grid(images, normalize=True, scale_each=True)
    results1 = vutils.make_grid(results1, normalize=True, scale_each=True)
    results2 = vutils.make_grid(results2, normalize=True, scale_each=True)
    writer.add_image('image/{}'.format(mode), torch.cat([images * 0.5 + results1 * 0.5, images * 0.5 + results2 * 0.5], 1))


def eval(model, dataLoader_valid):
    with torch.no_grad():
        model.eval()
        model.mode = 'valid'
        valid_loss, index_valid = 0, 0
        all_results = []
        all_masks = []
        all_fcs = []
        all_fc_labels = []

        for valid_data in dataLoader_valid:
            images, masks = valid_data
            images = images.cuda()
            masks = masks.cuda()
            b = images.size(0)
            outs, fc = data_parallel(model, images)
            model.loss = model.get_loss(outs, fc, masks)
            outs = torch.sigmoid(outs).cpu()
            fc = torch.sigmoid(fc).cpu()
            masks = masks.cpu()
            # all_results.append(outs)
            # all_masks.append(masks)
            all_results.append(F.upsample_bilinear(outs, (256, 256)))
            all_masks.append((F.upsample_bilinear(masks, (256, 256)) > 0.5).float())
            all_fcs.append(fc)
            all_fc_labels.append((masks.view(b, -1).sum(-1) > 0).float().view(b, 1))
            if index_valid == 0:
                drawing(images, outs, masks, mode='valid')
            b = len(masks)
            valid_loss += model.loss.item() * b
            index_valid += b

        valid_loss = valid_loss / index_valid

        all_results = torch.cat(all_results, 0)
        all_masks = torch.cat(all_masks, 0)

        ts = np.linspace(0.3, 0.7, 5)
        mIoUs = []
        mnegIoUs = []
        mposIoUs = []

        for t in ts:
            s, s_neg, s_pos, _, _ = metric(all_results, all_masks, t)
            mIoUs.append(s)
            mnegIoUs.append(s_neg)
            mposIoUs.append(s_pos)
        
        max_iou = max(mIoUs)
        max_index = mIoUs.index(max_iou)
        best_t = ts[max_index]
        valid_iou = max_iou

        return valid_loss, valid_iou, best_t, mnegIoUs[max_index], mposIoUs[max_index],


def get_lr(epoch, lr_min=0.001, lr_max=0.01, cycle_epochs=50, mode='cos_Annealing'):
    """

    :param epoch:
    :param lr_min:
    :param lr_max:
    :param cycle_epochs:
    :param mode: in ['linear', 'sin', etc]
    :return:
    """

    if mode == 'linear':
        epoch = (epoch % cycle_epochs) / cycle_epochs
        if epoch >= 0.5:
            lr = lr_max - (lr_max - lr_min) * (2 * epoch - 1)
        else:
            lr = lr_max - (lr_max - lr_min) * (1 - 2 * epoch)
    elif mode == 'sin':
        epoch = (epoch % cycle_epochs) / cycle_epochs
        lr = math.sin(epoch * math.pi) * (lr_max - lr_min) + lr_min
    elif mode == 'cos_Annealing':
        epoch = epoch / cycle_epochs - (epoch // cycle_epochs)
        lr = - (1 - math.cos(epoch * math.pi)) * 0.5 * (lr_max - lr_min) + lr_max
    elif mode == 'normal':
        if epoch > 300:
            lr = 0.005
        elif epoch > 400:
            lr = 0.001
        else:
            lr = 0.01
    elif mode == 'warmup':
        lr = epoch / 5 * (lr_max - lr_min) + lr_min
    else:
        print('error')
        exit(0)
    return lr


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train( fold_index=0, model_name='seresnext50', checkPoint_start=0, lr=1e-2, batch_size=36):
    model = model_iMet(model_name).cuda()
    ema_model = model_iMet(model_name).cuda()
    for p in ema_model.parameters():
        p.detach()
    i = 0
    iter_smooth = 50
    iter_valid = 500
    iter_save = 500
    epoch = 0
    # if freeze:
    # if freeze_mode is not None:
    #     model.freeze(freeze_mode)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=lr, weight_decay=0.0001)
    # fold_index = 1
    resultDir = './result/{}_semi_{}'.format(model_name, fold_index)
    # resultDir = './result/{}'.format(model_name)
    ImageDir = resultDir + '/image'
    checkPoint = os.path.join(resultDir, 'checkpoint')
    os.makedirs(checkPoint, exist_ok=True)
    os.makedirs(ImageDir, exist_ok=True)
    log = Logger()
    log.open(os.path.join(resultDir, 'log_train.txt'), mode='a')
    log.write(' start_time :{} \n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    log.write(' batch_size :{} \n'.format(batch_size))
    # Image,Id
    if 1:
        folds = pd.read_csv("./input/folds_6_stage2.csv")
        train_data = folds[folds['fold'] != fold_index]
        valid_data = folds[folds['fold'] == fold_index]
        train_names = train_data['path'].tolist()
        valid_names = valid_data['path'].tolist()
    print(len(train_names), len(valid_names))
    print(len(set(train_names) & set(valid_names)))

    # exit(0)
    dst_train = Dataset_pne_external(train_names, mode='train', transform=transform_train, transform_test=transform_test)
    num_data = dst_train.__len__()
    print(num_data)

    train_rles = [len(dst_train.masks_rle[name.split('/')[-1].replace('.dcm', '')]) for name in train_names]
    num_empty = train_rles.count(0)
    factor = float(num_empty) / num_data
    print(factor)
    weights = np.ones(num_data) * factor
    for i_, rle_num in enumerate(train_rles):
        if rle_num == 0:
            weights[i_] = 1 - factor
    sampler_train = torchSampler.WeightedRandomSampler(weights=weights,
                                                       num_samples=num_data,
                                                       replacement=True)
    dataloader_train = DataLoader(dst_train, sampler=sampler_train, pin_memory=False, batch_size=batch_size,
                                  num_workers=4)

    dst_valid = Dataset_pne(valid_names, mode='valid', transform=transform_valid)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, pin_memory=False, batch_size=batch_size, num_workers=4)
    train_loss, train_iou, train_acc = 0.0, 0.0, 0.0
    valid_loss, valid_iou, valid_iou2, valid_acc = 0.0, 0.0, 0.0, 0.0
    batch_loss, batch_iou, batch_acc = 0.0, 0.0, 0.0
    # best_t = 0
    train_loss_sum = 0
    train_iou_sum = 0
    train_acc_sum = 0

    sum = 0
    skips = []
    if not checkPoint_start == 0:
        log.write('start from l_rate ={}, start from{} \n'.format(lr, checkPoint_start))
        model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start)), skip=skips)
        ema_model.load_pretrain(os.path.join(checkPoint, '%08d_ema_model.pth' % (checkPoint_start)), skip=skips)
        ckp = torch.load(os.path.join(checkPoint, '%08d_optimizer.pth' % (checkPoint_start)))
        optimizer.load_state_dict(ckp['optimizer'])
        adjust_learning_rate(optimizer, lr)
        i = checkPoint_start
        epoch = ckp['epoch']
    log.write(
        ' rate     iter   epoch  | valid   top@1    top@5    map@5  | '
        'train    top@1    top@5    map@5 |'
        ' batch    top@1    top@5    map@5 |  time          \n')
    log.write(
        '---------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    start = timer()

    start_epoch = epoch
    best_t = 0
    cycle_epoch = 0
    while i < 10000000:

        for data in dataloader_train:
            images, masks, images_hard, images_simple = data

            epoch = start_epoch + (i - checkPoint_start) * batch_size / num_data
            if epoch > 20:
                alpha = 0.999
            elif epoch > 10:
                alpha = 0.99
            elif epoch > 5:
                alpha = 0.9
            else:
                alpha = 0.5
            # lr = get_lr(epoch)
            # adjust_learning_rate(optimizer, lr)
            if i % iter_valid == 0:
                valid_loss, valid_iou, best_t, best_neg, best_pos = \
                    eval(model, dataloader_valid)
                ema_valid_loss, ema_valid_iou,  ema_best_t, ema_best_neg, ema_best_pos = \
                    eval(ema_model, dataloader_valid)
                print('\r', end='', flush=True)
                log.write(
                    '%0.5f %5.2f k %5.2f  |'
                    ' %0.3f    %0.4f    %0.2f    %0.4f    %0.4f | %0.3f    %0.4f    %0.2f | %0.3f    %0.3f    %0.3f | %0.3f | %s \n' % ( \
                        lr, i / 1000, epoch,
                        valid_loss, valid_iou, best_t, best_neg, best_pos,
                        ema_valid_loss, ema_valid_iou,ema_best_t,
                        train_loss, train_iou, train_acc,
                        batch_loss,
                        time_to_str((timer() - start) / 60)))
                time.sleep(0.01)

            if i % iter_save == 0 and not i == checkPoint_start:
                torch.save(model.state_dict(), resultDir + '/checkpoint/%08d_model.pth' % (i))
                torch.save(ema_model.state_dict(), resultDir + '/checkpoint/%08d_ema_model.pth' % (i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': i,
                    'epoch': epoch,
                    'cycle_epoch': cycle_epoch,
                    'best_t': best_t,
                }, resultDir + '/checkpoint/%08d_optimizer.pth' % (i))

            model.train()
            ema_model.eval()

            model.mode = 'train'
            images_hard = images_hard[: 4]
            images_simple = images_simple[: 4]
            b = images.size(0)
            if not b == batch_size: continue
            images = images.cuda()
            masks = masks.cuda()

            optimizer.zero_grad()

            outs, fc = data_parallel(model, images)
            loss = model.get_loss(outs, fc, masks)
            loss.backward()
            batch_loss = loss.item()

            #   for semi
            images_hard, images_simple = images_hard.cuda(), images_simple.cuda()
            model.freeze_bn()
            outs_hard, fc_hard = data_parallel(model, images_hard)
            outs_simple, fc_simple = data_parallel(ema_model, images_simple)
            #
            if epoch < 5:
                gamma = 0.001
            else:
                gamma = 2 * sigmoid_rampup(epoch, 30)
            con_loss = model.get_con_loss(outs_hard, fc_hard, outs_simple, fc_simple, gamma)
            #
            #
            con_loss.backward()
            batch_loss += con_loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
            optimizer.step()

            masks = masks.cpu()
            # batch_loss = batch_loss.item()
            outs = torch.sigmoid(outs).cpu()
            fc = torch.sigmoid(fc).cpu().view(-1)
            fc_label = (masks.view(b, -1).sum(-1) > 0).float().view(-1)
            batch_iou, _, _, _, _ = metric(outs, masks, threshold=0.6)
            batch_acc = ((fc > 0.5).float() == fc_label.float()).float().sum() / fc_label.size(0)

            sum += 1
            train_loss_sum += batch_loss
            train_iou_sum += batch_iou
            train_acc_sum += batch_acc
            update_ema_variables(model, ema_model, alpha, i)
            ema_model.train()
            data_parallel(ema_model, images)
            if (i + 1) % iter_smooth == 0:
                drawing(images_simple.cpu(), outs_hard.sigmoid().cpu(), outs_simple.sigmoid().cpu(), mode='train')
                train_loss = train_loss_sum / sum
                train_iou = train_iou_sum / sum
                train_acc = train_acc_sum / sum
                train_loss_sum = 0
                train_iou_sum = 0
                train_acc_sum = 0
                sum = 0

            print('\r%0.5f %5.2f k %5.2f  | %0.3f    %0.4f    %0.2f    %0.4f    %0.4f | %0.3f    %0.4f    %0.2f | %0.3f    %0.3f    %0.3f | %0.3f | %s  %d %d %0.2f' % ( \
                    lr, i / 1000, epoch,
                    valid_loss, valid_iou, best_t, best_neg, best_pos,
                    ema_valid_loss, ema_valid_iou, ema_best_t,
                    train_loss, train_iou, train_acc,
                    batch_loss,
                    time_to_str((timer() - start) / 60), checkPoint_start, i, gamma)
                , end='', flush=True)
            i += 1
        cycle_epoch += 1
        pass


if __name__ == '__main__':
    if 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

        parser = argparse.ArgumentParser(description='training parameters')
        parser.add_argument('--modelname', type=str, default='seresnext50',
                            help='backbone')
        parser.add_argument('--fold_index', type=int, default=0,
                            help='fold index')
        parser.add_argument('--checkPoint_start', type=int, default=0,
                            help='checkPoint_start')
        parser.add_argument('--batch_size', type=int, default=16,
                            help='batchsize')
        parser.add_argument('--lr', type=float, default=3e-4,
                            help='fold index')

        args = parser.parse_args()
        model_name = args.modelname
        checkPoint_start = args.checkPoint_start
        fold_index = args.fold_index
        lr = args.lr
        batch_size = args.batch_size
        print(model_name, checkPoint_start, fold_index, lr, batch_size)
        train(fold_index, model_name,
              checkPoint_start, lr, batch_size)
