import random

__all__ = ['DataTransform', 'CompositeTransform']


class DataTransform(object):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs: dict) -> dict:
        if self.p < 1 and random.uniform(0, 1) > self.p:
            return inputs
        return self.apply(inputs)

    def apply(self, inputs: dict) -> dict:
        raise NotImplementedError


class CompositeTransform(DataTransform):
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, inputs):
        for transform in self.transforms:
            inputs = transform(inputs)

        return inputs
