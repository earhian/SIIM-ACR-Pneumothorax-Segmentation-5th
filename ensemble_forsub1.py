import numpy as np
import pandas as pd

from utils.utils import *
# 39000, 39500, 36500, 38000
from tqdm import tqdm
import os
results = [


    ['seresnext50_semi', 0, 52000],
    ['seresnext50_semi', 1, 16000],
    ['seresnext50_semi', 2, 28500],
    ['seresnext50_semi', 3, 29500],
    ['seresnext50_semi', 4, 52000],
    # ['seresnext50_semi_v2', 1, 34000],

    ['seresnext101_semi', 0, 28000],
    ['seresnext101_semi', 1, 21500],
  ]


ws = [1] * len(results)

print(results)
assert len(results) == len(ws)
def npy2json(model_name, fold_index, cpk, temp=None, w=1):
    resultDir = './result/{}_{}'.format(model_name, fold_index)
    all_names = np.load(os.path.join(resultDir, '%08d_names.npy' % (cpk)))
    all_results = np.load(os.path.join(resultDir, '%08d_results.npy' % (cpk)))
    dict_results = {}
    if temp is not None:
        dict_results = temp
    for name, out in tqdm(zip(all_names, all_results)):
        # temp = (out > 0.5).astype(np.float)
        if out.max() < 0.45:
            out = np.zeros_like(out).astype(np.float)
        else:
            out = (out > 0.2).astype(np.float)

        if str(name) in dict_results:
            dict_results[str(name)] += out * w
        else:
            dict_results[str(name)] = out * w
    return dict_results
# def load():
#     data = pd.read_csv('8822.csv')
#     EncodedPixels = data['EncodedPixels'].tolist()
def ensemble(dict_results, best_t=0.6, num_results=1):
    count = 0
    count2 = 0
    all_names = []
    all_rles = []
    print(len(all_names), best_t)
    for name in tqdm(dict_results.keys()):
        res = dict_results[name]/float(sum(ws))
        # res = dict_results[name]
        out = (res > best_t).astype(np.int)
        if out.max() == 0:
            rle = '-1'
        else:
            out = (res > 0.2).astype(np.int)
            if out.sum() <= 64:
                rle = '-1'
                count2 += 1
            else:
                rle = run_length_encode(out)
        if rle == '-1':
            count += 1
        all_names.append(name)
        all_rles.append(rle)
    pd.DataFrame({"ImageId": all_names, "EncodedPixels": all_rles}).to_csv('ensemble_sub1.csv', index=None)
    print(count, count2)


dict_results = None
print(len(results))
for result, w in zip(results, ws):
    model_name, fold_index, cpk = result
    # dict_results = None
    dict_results = npy2json(model_name, fold_index, cpk, dict_results, w)
ensemble(dict_results, best_t=0.45, num_results=len(results))

