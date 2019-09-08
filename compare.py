import pandas as pd
import numpy as np
import torch
from utils.utils import *
from utils.metric import *
from tqdm import tqdm
gt = pd.read_csv('input/stage2/stage_2_train.csv',engine='python')
results = pd.read_csv('ensemble_sub1.csv', engine='python')


def get_rle(data):
    # 'ImageId', u' EncodedPixels'
    ImageIds = data['ImageId'].tolist()
    EncodedPixels = data['EncodedPixels'].tolist()
    rle_dict = {}
    for ImageId, EncodedPixel in zip(ImageIds, EncodedPixels):
        if EncodedPixel.strip() == "-1":
            rle_dict[ImageId] = []
            continue
        if ImageId in rle_dict.keys():
            rle_dict[ImageId].append(EncodedPixel)
        else:
            rle_dict[ImageId] = [EncodedPixel]
    return rle_dict

gt = get_rle(gt)
results = get_rle(results)
print(len(results))
print(len(gt))
scores = []
scores_pos = []
scores_neg = []
count = 0
for k in tqdm(results.keys()):
    if k not in gt:
        continue
    res = gt[k]
    mask = results[k]
    assert len(res) <= 1
    assert len(mask) <= 1
    if len(res) == 1:
        res = rle2mask(res[0], 1024, 1024)
    else:
        res = np.zeros((1024, 1024))

    if len(mask) == 1:
        mask = rle2mask(mask[0], 1024, 1024)
    else:
        count += 1
        mask = np.zeros((1024, 1024))

    res = torch.from_numpy(res).float().div(255).unsqueeze(0)
    mask = torch.from_numpy(mask).float().div(255).unsqueeze(0)
    s, s_neg, s_pos = metric(res, mask, 0.5)[:3]
    scores.append(s)
    if mask.max() > 0:
        scores_pos.append(s_pos)
    else:
        scores_neg.append(s_neg)
print(len(scores_pos), len(scores_neg))
print(sum(scores)/len(scores))
print(sum(scores_pos)/len(scores_pos))
print(sum(scores_neg)/len(scores_neg))
print(count)
print(len(scores))