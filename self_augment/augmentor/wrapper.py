import numpy as np
from copy import deepcopy
from .affine_wrapper import RandomAffineWrapper
from .tps_wrapper import TPSWrapper


class AugmentWrapper:
    def __init__(self):
        self.random_affine = RandomAffineWrapper()
        self.tps = TPSWrapper()

        self.warps = {
            'tps': self.tps,
            'affine': self.random_affine,
        }

    def gen_augment_param(self, key_params, key_name):
        for aug_name, aug in self.warps.items():
            _key_param = key_params[aug_name]
            _key_param['key_name'] = key_name
            aug.gen(**_key_param)
        return

    def augment(self, key_name, image, p):
        output = deepcopy(image)

        if p < 0.65:
            warp = {'tps': self.tps}
        else:
            warp = {'affine': self.random_affine}

        for aug_name, aug in warp.items():
            output = aug.augment(output, key_name)

        return output
