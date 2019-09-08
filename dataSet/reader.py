import pydicom
from torch.utils.data import Dataset, DataLoader

from utils.utils import *
from common import *
import glob
import os

# pydicom.dcmread(file_path)
class Dataset_pne(Dataset):
    def __init__(self, names, mode='train', transform=None):
        super(Dataset_pne, self).__init__()
        self.names = names
        self.mode = mode
        self.transform = transform
        if mode in ['train', 'valid']:
            self.masks_rle = self.get_rle()

    def get_rle(self):
        # 'ImageId', u' EncodedPixels'
        data = pd.read_csv('./input/stage2/stage_2_train.csv')
        ImageIds = data['ImageId'].tolist()
        EncodedPixels = data['EncodedPixels'].tolist()
        rle_dict = {}
        for ImageId, EncodedPixel in zip(ImageIds, EncodedPixels):
            if EncodedPixel == "-1":
                rle_dict[ImageId] = []
                continue
            if ImageId in rle_dict.keys():
                rle_dict[ImageId].append(EncodedPixel)
            else:
                rle_dict[ImageId] = [EncodedPixel]
        return rle_dict

    def __len__(self):
        return len(self.names)

    def load_img(self, filePath):
        return pydicom.dcmread(filePath).pixel_array

    def load_mask(self, rle, w, h):
        return rle2mask(rle, w, h)

    def load_labels(self, rles, w, h):
        mask_zeros = np.zeros((w, h))
        for rle in rles:
            mask = self.load_mask(rle, w, h)
            mask_zeros += mask
        return (mask_zeros > 0).astype(np.uint8) * 255

    def __getitem__(self, index):
        if self.mode in ['train']:
            name = self.names[index]
            img_name = name.split('/')[-1].replace('.dcm', '')
            filePath = './input/dicom-images-train/' + name
            img = self.load_img(filePath)
            w, h = img.shape
            rles = self.masks_rle[img_name]
            mask = self.load_labels(rles, h, w)
            img, mask = self.transform(img, mask)

            return img, mask
        elif self.mode in ['valid']:
            name = self.names[index]
            img_name = name.split('/')[-1].replace('.dcm', '')
            filePath = './input/dicom-images-train/' + name
            if not os.path.exists(filePath):
                filePath = './input/dicom-images-test/' + name
            img = self.load_img(filePath)
            w, h = img.shape
            rles = self.masks_rle[img_name]
            mask = self.load_labels(rles, h, w)
            img, mask = self.transform(img, mask)
            return img, mask
        elif self.mode in ['test']:
            name = self.names[index]
            img_name = name.split('/')[-1].replace('.dcm', '')
            filePath = './input/dicom-images-test/' + name
            img = self.load_img(filePath)
            imgs = self.transform(img)
            return imgs, img_name
        else:
            name = self.names[index]
            img_name = name.split('/')[-1].replace('.dcm', '')
            filePath = './input/stage2/stage2_test/' + name
            img = self.load_img(filePath)
            imgs = self.transform(img)
            return imgs, img_name


class Dataset_pne_external(Dataset):
    def __init__(self, names, mode='train', transform=None, transform_test=None):
        super(Dataset_pne_external, self).__init__()
        self.names = names
        self.mode = mode
        self.transform = transform
        if mode in ['train', 'valid']:
            self.masks_rle = self.get_rle()
        # 00029075_003.png
        if mode == 'train':
            print('loading ex')
            self.transform_test = transform_test
            nihIds = self.getNIHId()
            print(len(nihIds))
            self.testPath = EXDATAPATH
            self.testNames = glob.glob(self.testPath)
            print(len(self.testNames))
            self.postestNames = [testName for testName in self.testNames if testName.split('/')[-1].split('_')[0] in nihIds]
            self.negtestNames = [testName for testName in self.testNames if testName.split('/')[-1].split('_')[0] not in nihIds]
            print(len(self.postestNames))
            print(len(self.negtestNames))

    def getNIHId(self):
        data = pd.read_csv('./input/nih_id.csv')
        nih_ids = data['NIH_ID'].tolist()
        return [nih_id.split('_')[0] for nih_id in nih_ids]

    def get_rle(self):
        # 'ImageId', u' EncodedPixels'
        data = pd.read_csv('./input/stage2/stage_2_train.csv')
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

    def __len__(self):
        return len(self.names)

    def load_img(self, filePath):
        return pydicom.dcmread(filePath).pixel_array

    def load_mask(self, rle, w, h):
        return rle2mask(rle, w, h)

    def load_labels(self, rles, w, h):
        mask_zeros = np.zeros((w, h))
        for rle in rles:
            mask = self.load_mask(rle, w, h)
            mask_zeros += mask
        return (mask_zeros > 0).astype(np.uint8) * 255

    def __getitem__(self, index):
        if self.mode in ['train']:
            name = self.names[index]
            testname = random.choice(self.testNames)
            testImg = cv2.imread(testname, cv2.IMREAD_GRAYSCALE)

            img_name = name.split('/')[-1].replace('.dcm', '')
            filePath = './input/dicom-images-train/' + name
            if not os.path.exists(filePath):
                filePath = './input/dicom-images-test/' + name
            img = self.load_img(filePath)
            w, h = img.shape
            rles = self.masks_rle[img_name]
            mask = self.load_labels(rles, h, w)
            img, mask = self.transform(img, mask)
            test_img_hard, test_img_simple = self.transform_test(testImg)
            return img, mask, test_img_hard, test_img_simple
        elif self.mode in ['valid']:
            name = self.names[index]
            img_name = name.split('/')[-1].replace('.dcm', '')
            filePath = './input/dicom-images-train/' + name
            if not os.path.exists(filePath):
                filePath = './input/dicom-images-test/' + name
            img = self.load_img(filePath)
            w, h = img.shape
            rles = self.masks_rle[img_name]
            mask = self.load_labels(rles, h, w)
            img, mask = self.transform(img, mask)
            return img, mask
        else:
            name = self.names[index]
            img_name = name.split('/')[-1].replace('.dcm', '')
            filePath = './input/dicom-images-test/' + name
            img = self.load_img(filePath)
            imgs = self.transform(img)
            return imgs, img_name

