# utils.py
import torch
import math

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CosineScheduler(object):
    def __init__(self, optimizer, param_name, t_max, value_min=0, warmup_t=0, const_t=0):
        self.optimizer = optimizer
        self.param_name = param_name
        self.t_max = t_max
        self.value_min = value_min
        self.warmup_t = warmup_t
        self.const_t = const_t
        self.t = 0
        self.value_max = [group[param_name] for group in optimizer.param_groups][0]

    def step(self, epoch):
        self.t = epoch
        value = self._get_value()
        for param_group in self.optimizer.param_groups:
            param_group[self.param_name] = value

    def _get_value(self):
        if self.t < self.warmup_t:
            return self.value_max * self.t / self.warmup_t
        elif self.t < self.warmup_t + self.const_t:
            return self.value_max
        else:
            t = self.t - self.warmup_t - self.const_t
            t_max = self.t_max - self.warmup_t - self.const_t
            return self.value_min + 0.5 * (self.value_max - self.value_min) * (1 + math.cos(math.pi * t / t_max))

    def state_dict(self):
        return {'t': self.t}

    def load_state_dict(self, state_dict):
        self.t = state_dict['t']

def pad_img(img, patch_size):
    _, _, H, W = img.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img
