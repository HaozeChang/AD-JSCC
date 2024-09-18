# An NOT OFFICIAL torch implementation of ADJSCC
# This is an NOT official torch implementation of ADJSCC:
## [paper:Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules](https://ieeexplore.ieee.org/document/9438648)
## Usage
### 1.create env:
```python
#!/usr/bin/env python3
conda create -n env_name python=3.8
conda activate env_name  
```
replace env_name with your design!

### 2.essential package:
```python
#!/usr/bin/env python3
pip install pytorch-lightning==1.7.7
pip install torch==1.9.0+cu111
pip install torchmetrics==0.11.4
pip install torchvision==0.10.0+cu111

```
these works fine for my:
Python  3.8(ubuntu18.04)
Cuda  11.1

### 3.train and test:
```python
#!/usr/bin/env python3
./train.sh
```

```python
#!/usr/bin/env python3
python train_jscc.py --ckpt_addr '' --batch_size 64 --device 1 --max_epoches 500 --check_val_every_n_epoch 50 --save_ckpt_every_n_epochs 100 --fast_dev_run False --channel 'Rayl'
```
change the sh file as you want!


## Dataset:
cifar10 from 
```python
#!/usr/bin/env python3
torchvision.datasets.CIFAR10
```

COCO dataset can be used to replace Imagenet

## Citation: 
J. Xu, B. Ai, W. Chen, A. Yang, P. Sun and M. Rodrigues, "Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 4, pp. 2315-2328, April 2022, doi: 10.1109/TCSVT.2021.3082521.



