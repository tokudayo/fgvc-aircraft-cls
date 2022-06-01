import os
from copy import deepcopy

import torch
import torchvision
from torch import nn


def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.append(m.weight)
            params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.LayerNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
        elif isinstance(m, torchvision.models.convnext.CNBlock):
            params_no_decay.append(m.layer_scale)
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)

    return params_decay, params_no_decay


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._last_val = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._last_val = val
        self._sum += val * n
        self._count += n
    
    @property
    def last_val(self):
        return self._last_val

    @property
    def avg(self):
        return self._sum / self._count


class L2SP_ConvNext(nn.Module):
    def __init__(self, ref_model, alpha: float = 5e-5, beta: float = 2e-5):
        """
        L2SP regularization tailored for ConvNeXt.
        """
        super().__init__()
        ref = {x[0]:x[1].detach().clone() for x in ref_model.named_parameters()}
        for v in ref.values():
            v.requires_grad = False
        # Remove layer norm, biases and layer scale from decay
        keys = list(ref.keys())
        noreg = set()
        for k in keys:
            if len(k.split('.')) == 6:
                if k.split('.')[-2] == '2':
                    ref.pop(k)
                    noreg.add(k)
            elif 'bias' in k:
                ref.pop(k)
                noreg.add(k)
            elif 'layer_scale' in k:
                ref.pop(k)
                noreg.add(k)
        ref.pop('features.0.1.weight')
        noreg.add('features.0.1.weight')
        for i in range(2, 7, 2):
            ref.pop(f'features.{i}.0.weight')
            noreg.add(f'features.{i}.0.weight')
        
        self.ref = ref
        self.noreg = noreg
        self.alpha = alpha
        self.beta = beta

    def __call__(self, model):
        for k, v in model.named_parameters():
            if k in self.ref:
                v.grad.add_(v - self.ref[k], alpha=self.alpha)
            elif ('layer_scale' not in k) and ('bias' not in k) and (k not in self.noreg):
                v.grad.add_(v, alpha=self.beta)


class L2SP_InceptionV4(nn.Module):
    def __init__(self, ref_model, alpha: float = 1e-4, beta: float = 1e-5):
        """
        L2SP regularization tailored for ConvNeXt.
        """
        super().__init__()
        ref = {x[0]:x[1].detach().clone() for x in ref_model.named_parameters()}
        for v in ref.values():
            v.requires_grad = False
        # Remove layer norm, biases and layer scale from decay
        keys = list(ref.keys())
        for name in keys:
            if 'bn' in name or 'bias' in name:
                ref.pop(name)
        
        self.ref = ref
        self.alpha = alpha
        self.beta = beta

    def __call__(self, model):
        for k, v in model.named_parameters():
            if k in self.ref:
                v.grad.add_(v - self.ref[k], alpha=self.alpha)
            elif ('bias' not in k) and ('bn' not in k):
                v.grad.add_(v, alpha=self.beta)


class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def dataset_mean_and_std(dataset):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for input, _ in dataset:
        input = input.view(3, -1)
        mean += input.mean(1)
        std += input.std(1)

    mean /= len(dataset)
    std /= len(dataset)
    return mean, std


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        pass
