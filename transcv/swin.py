# AUTOGENERATED! DO NOT EDIT! File to edit: 01_swin.ipynb (unless otherwise specified).

__all__ = ['SwinT']

# Cell
#hide
import timm
from nbdev.showdoc import *
from fastai.vision.all import *

# Cell
class SwinT :
    """Class for setting up a Swin Transformer model. The architecture is specified by `arch`, and the
    number of classes is specified by `num_classes`. Returns a pretrained model, by default, or an
    initialised model if `pretrained` is set to `False`.
    """

    def __init__ (self, arch, num_classes, pretrained = True) :
        self.arch = arch
        self.pretrained = pretrained
        self.num_classes = num_classes

    def get_model (self) :
        """Method for getting the Swin Transformer model.
        """
        model_timm = timm.create_model(self.arch, pretrained = self.pretrained, num_classes = self.num_classes)
        model = nn.Sequential(model_timm)
        return model