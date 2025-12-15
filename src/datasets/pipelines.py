from torchvision import transforms
from src.core.registry import PIPELINES

@PIPELINES.register_module()
class Resize:
    def __init__(self, size):
        self.size = size
        self.transform = transforms.Resize(size)
        
    def __call__(self, img):
        return self.transform(img)

@PIPELINES.register_module()
class CenterCrop:
    def __init__(self, size):
        self.size = size
        self.transform = transforms.CenterCrop(size)
        
    def __call__(self, img):
        return self.transform(img)

@PIPELINES.register_module()
class ToTensor:
    def __init__(self):
        self.transform = transforms.ToTensor()
        
    def __call__(self, img):
        return self.transform(img)

@PIPELINES.register_module()
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.transform = transforms.Normalize(mean=mean, std=std)
        
    def __call__(self, img):
        return self.transform(img)

@PIPELINES.register_module()
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
        self.transform = transforms.RandomHorizontalFlip(p=p)
    
    def __call__(self, img):
        return self.transform(img)

@PIPELINES.register_module()
class Compose:
    def __init__(self, transforms_cfg):
        """
        Args:
            transforms_cfg (list): List of config dicts for transforms.
        """
        self.transforms = []
        for t_cfg in transforms_cfg:
             self.transforms.append(PIPELINES.build(t_cfg))
        self.compose = transforms.Compose(self.transforms)

    def __call__(self, img):
        return self.compose(img)
