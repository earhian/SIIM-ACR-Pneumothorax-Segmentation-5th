# SIIM-ACR-Pneumothorax-Segmentation-5th
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/leaderboard

#### Dependencies
- python==3.6
- torch==1.0+
- torchvision==0.3+


## Solution  
https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107603#latest-620540

### prepare data  
## competition dataset  
./input/dicom-images-train  
./input/dicom-images-test  

## NIH data
download NIH dataset             
https://www.kaggle.com/nih-chest-xrays/data             
common.py: EXDATAPATH = './input/NIH/images/*.png'




### Train  
python train_semi.py --modelname="seresnext50" --fold_index=0 --batch_size=16 --lr=3e-4

### Test  

stage1: python test_json.py
stage2: python test_json_stage2.py

### ensemble
stage1: python ensemble_forsub1.py
stage2: python ensemble_forsub1_stage2.py
