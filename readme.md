# An NOT OFFICIAL torch implementation of ADJSCC
# This is an NOT official torch implementation of ADJSCC:
## [paper:Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules](https://ieeexplore.ieee.org/document/9438648)
The implementation thanks to Jorge! Full respect!
[GDN pytorch](https://github.com/jorge-pessoa/pytorch-gdn)
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

## Caution!
When trained with multiple GPU, an error might occur to gdn.py for putting tensors on different devices. 

In most multiple GPU training situations, the
```python
#!/usr/bin/env python3
torch.device('cuda')
```
will put tensor on **'cuda:0'**,thus might cause proble,

### potential methods:
1.dynamically put gdn on the input tensor x's device like:
```python
#!/usr/bin/env python3
class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncodingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.PReLU()
        self.gdn = None

    def forward(self, x):
        # if GDN not intilized，dynamicallye create GDN layer based on x's device
        if self.gdn is None:
            self.gdn = GDN(ch=x.size(1)).to(x.device)

        x = self.conv(x)
        x = self.gdn(x)
        x = self.activation(x)
        return x

```

2.change GDN layer:
```python
#!/usr/bin/env python3
import torch
from torch import nn
from torch.autograd import Function

class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        # 确保 b 在 inputs 的设备上
        b = torch.ones(inputs.size(), device=inputs.device) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """
    def __init__(self,
                ch,
                inverse=False,
                beta_min=1e-6,
                gamma_init=.1,
                reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = nn.Parameter(torch.tensor([reparam_offset]))

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** 0.5
        self.gamma_bound = self.reparam_offset

        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        device = inputs.device  

        beta = LowerBound.apply(self.beta, self.beta_bound).to(device)
        beta = beta ** 2 - self.pedestal.to(device)

        gamma = LowerBound.apply(self.gamma, self.gamma_bound).to(device)
        gamma = gamma ** 2 - self.pedestal.to(device)
        gamma = gamma.view(inputs.size(1), inputs.size(1), 1, 1).to(device)

        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)
        
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        return outputs

```

## Citation: 
J. Xu, B. Ai, W. Chen, A. Yang, P. Sun and M. Rodrigues, "Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 4, pp. 2315-2328, April 2022, doi: 10.1109/TCSVT.2021.3082521.



