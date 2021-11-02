# High-Resolution-Chest-X-ray-Bone-Suppression
Code of  [High-Resolution Chest X-ray Bone Suppression Using Unpaired CT Structural Priors](http://miracle.ict.ac.cn/?page_id=814)
## Requirement:
```
Python==3.6
Torch==1.1.0
Scipy ==1.1.0   if your Scipy>=1.3.0, you will meet “ImportError: cannot import name 'imresize'”
```
## Get started
There three steps of our work: 1. cycleganxray2gradient, 2. Pytorch-UNet-master_trainbonemask and 3. histogram_match.

Please download our whole project code [HERE](https://pan.baidu.com/s/1OISU_Th6ATA0a8lkFIhDaw) code: 6orv (including weight, test imges) instead of the code in this github.

Pleas feel free to contact me if you have any problem: han.li[at]miracle.ict.ac.cn / 137412918[at]qq.com 

### Step1: cycleganxray2gradient
```
 cd 1cycleganxray2gradient 
 python test.py --dataroot ./datasets --name maps_cyclegan4 --model cycle_gan --epoch 110
```
Input data location: ‘./datasets/testA’

Output data location: ’ ./results/maps_cyclegan4/test_110/images’

### Step2: Pytorch-UNet-master_trainbonemask
```
cd ..
cd 2Pytorch-UNet-master_trainbonemask
python newpredict.py -c ./checkpoints/CP50.pth
```
Input data location: Output data location in step1

Output bone data location: ’ ./bone ’

Output lung mask location: ’ ./lung ’

### Step3：histogram_match.
```
cd ..
cd 3histogram_match
python bonemask2result.py
```
input data location: ’ ./dataset'’

final result location : ‘./result’
