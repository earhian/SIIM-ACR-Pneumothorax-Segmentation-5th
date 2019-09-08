import numpy as np
import pandas as pd
# from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from common import *

data = pd.read_csv('../input/train.csv')
Id, Target = np.array(data['id'].tolist()), data['attribute_ids'].tolist()


def list2array(label):
    label_zero = np.zeros(NUM_CLASSES)
    for l in label.split(' '):
        label_zero[int(l)] = 1
    return [label_zero]


num_data = len(Target)
print('start loading')
Targets = []
single_nums = 1000
split_nums = num_data // single_nums + 1
for i in tqdm(range(split_nums)):
    Targets.extend(multi_apply(list2array, Target[i * single_nums: i * single_nums + single_nums])[0])
print('loading over')
kfold = MultilabelStratifiedShuffleSplit(n_splits=12, test_size=0.08, random_state=123)
# kfold = StratifiedKFold(n_splits=12, shuffle=True, random_state=10)
train_df_orig = data.copy()
X = Id
y = np.array(Targets)

for index, (train_index, test_index) in tqdm(enumerate(kfold.split(X, y))):  # it should only do one iteration
    print(index, "TRAIN:", len(train_index), "TEST:", len(test_index))
    train_df = train_df_orig.loc[train_df_orig.index.intersection(train_index)].copy()
    valid_df = train_df_orig.loc[train_df_orig.index.intersection(test_index)].copy()
    print(len(train_df), len(valid_df))
    print(len(set(train_index)), len(set(test_index)))
    train_df.to_csv(
        './input/train_StratifiedKFold_{}.csv'.format(index), index=None
    )
    valid_df.to_csv(
        './input/valid_StratifiedKFold_{}.csv'.format(index), index=None
    )

"""
from common import *
data = pd.read_csv('./input/train.csv')
Names = np.array(data['id'].tolist())
Ids = np.array(data['attribute_ids'].tolist())
num_names = len(Names)
all_indexs = list(range(num_names))
# valid_indexs = random.sample(all_indexs, 1103 * 5)
# train_indexs = list(set(all_indexs) ^ set(valid_indexs))
# valid_names, valid_Ids = Names[valid_indexs], Ids[valid_indexs]
# train_names, train_Ids = Names[train_indexs], Ids[train_indexs]

def split_list(num, n):
    random.seed(1325)
    list_all = list(range(num))
    lists = []
    for index in range(1, n):
        list_ = set(list_all)
        for list_item in lists:
            list_ = list_ ^ set(list_item)
        if index == n-1:
            lists.append(random.sample(list_, len(list_)))
            break
        lists.append(random.sample(list_,num//n))
    return lists
lists = split_list(num=num_names, n=12)
for index, valid_indexs in enumerate(lists):
    train_indexs = list(set(all_indexs) ^ set(valid_indexs))
    valid_names, valid_Ids = Names[valid_indexs], Ids[valid_indexs]
    train_names, train_Ids = Names[train_indexs], Ids[train_indexs]
    pd.DataFrame({'id':train_names, 'attribute_ids':train_Ids}).to_csv(
        './input/train_split_{}.csv'.format(index), index=None
    )
    pd.DataFrame({'id':valid_names, 'attribute_ids':valid_Ids}).to_csv(
        './input/valid_split_{}.csv'.format(index), index=None
    )
"""
