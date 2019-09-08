import torchvision.utils as vutils
# image_size = 128
from tensorboardX import SummaryWriter
import json
from dataSet.reader import *
from models.model import *
from utils.metric import *
from common import *
import os
from tqdm import tqdm
writer = SummaryWriter('test')


def collate_test(batch):
    images = []
    names = []
    for b in batch:
        images.extend(b[0])
        names.append(b[1])
    images = torch.stack(images, 0)

    return images, names


def transform_test(image):
    images = []
    image = cv2.resize(image, (SIZE, SIZE))
    image_flip = np.fliplr(image.copy())
    image = torch.from_numpy(image).float().div(255)
    image = image.unsqueeze(0)
    image_flip = torch.from_numpy(image_flip.copy()).float().div(255)
    image_flip = image_flip.unsqueeze(0)
    return [image, image_flip]


def drawing(images, results):
    images = images[:16].data.cpu()

    results = (results[:16] > 0.5).float()
    images = vutils.make_grid(images, normalize=True, scale_each=True)
    results = vutils.make_grid(results, normalize=True, scale_each=True)
    writer.add_image('image/demo', images * 0.5 + results * 0.5)



def test(model_name, fold_index, checkPoint_start, batch_size):
    resultDir = './result/{}_semi_stage2_{}'.format(model_name, fold_index)
    checkPoint = os.path.join(resultDir, 'checkpoint')

    model = model_iMet(model_name).cuda()
    # test_names = pd.read_csv('./input/test.csv')['path'].tolist()
    test_names = os.listdir('./input/stage2/stage2_test')
    dst_valid = Dataset_pne(test_names, mode='stage2_test', transform=transform_test)
    dataLoader_test = DataLoader(dst_valid, shuffle=False, pin_memory=True, batch_size=batch_size, num_workers=4,
                                 collate_fn=collate_test)

    if not checkPoint_start == 0:
        model.load_pretrain(os.path.join(checkPoint, '%08d_model.pth' % (checkPoint_start)), skip=[])
    with torch.no_grad():
        model.eval()
        all_names = []
        all_results = []
        for valid_data in tqdm(dataLoader_test):
            images, names = valid_data
            images = images.cuda()
            outs, fc = data_parallel(model, images)
            outs = torch.sigmoid(outs).detach().cpu()
            if outs.size(-1) != 1024:
                outs = F.upsample_bilinear(outs, (1024, 1024))
            fc = torch.sigmoid(fc).detach().cpu()
            outs_raw = outs[0::2]
            outs_flip = outs[1::2]
            fc = (fc[0::2] + fc[1::2])/2
            for out_raw, out_flip, score, name in zip(outs_raw, outs_flip, fc, names):
                out_raw = (out_raw[0]).numpy()
                out_flip = np.fliplr((out_flip[0]).numpy())
                out = (out_flip + out_raw) * 0.5
                # dict_results[name] = out
                all_names.append(name)
                all_results.append(out)

        all_names = np.array(all_names)
        all_results = np.array(all_results)
        np.save(os.path.join(resultDir, '%08d_stage2_names.npy' % (checkPoint_start)), all_names)
        np.save(os.path.join(resultDir, '%08d_stage2_results.npy' % (checkPoint_start)), all_results)
    return all_names, all_results

def ensemble(results, best_t=0.6):
    count = 0
    all_names = []
    all_rles = []
    for name in results[0].keys():
        res = [res[name] for res in results]
        res = np.mean(np.stack(res, 0), 0)
        out = (res > best_t).astype(np.int)
        rle = run_length_encode(out)
        if rle == '-1':
            count += 1
        all_names.append(name)
        all_rles.append(rle)
    pd.DataFrame({"ImageId": all_names, "EncodedPixels": all_rles}).to_csv('ensemble.csv', index=None)
    print(count)

if __name__ == '__main__':
    if 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        model_names = ['seresnext50'] * 6
        batch_size = 32
        checkPoint_starts = [28500, 30500, 31000, 25500, 34500, 27500]
        fold_indexs = [0, 1, 2, 3, 4, 5]
        all_results = []
        for model_name, fold_index, checkPoint_start in zip(model_names, fold_indexs, checkPoint_starts):
            test(model_name, fold_index, checkPoint_start, batch_size)
        # ensemble(all_results, 0.5)
