# AUTOGENERATED! DO NOT EDIT! File to edit: 00_visrectrans.ipynb (unless otherwise specified).

__all__ = ['EmbedBlock', 'Header', 'custom_ViT', 'trunc_normal_', 'VisRecTrans']

# Cell
#hide
from nbdev.showdoc import *
from fastai.vision.all import *
import timm
import math
import warnings

# Cell
#hide
# Heavily inspired by "https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py"

class EmbedBlock (Module) :
  def __init__ (self, num_patches, embed_dim) :
    self.cls_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embeds = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

  def forward (self, x) :
    B = x.shape[0]
    cls_tokens = self.cls_tokens.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim = 1)
    x = x + self.pos_embeds
    return x

# Cell
#hide
# Heavily inspired by "https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py"

class Header (Module) :
  def __init__ (self, ni, num_classes) :
    self.head = nn.Linear(ni, num_classes)

  def forward (self, x) :
    x = x[:, 0]                  # Extracting the clsass token, which is used for the classification task.
    x = self.head(x)
    return x

# Cell
#hide
def custom_ViT (timm_model_name, num_patches, embed_dim, ni, num_classes, pretrained = True) :
  model = timm.create_model(timm_model_name, pretrained)
  module_layers = list(model.children())
  return nn.Sequential(
      module_layers[0],
      EmbedBlock(num_patches, embed_dim),
      nn.Sequential(*module_layers[1:-1]),
      Header(ni, num_classes)
  )

# Cell
#hide
# Heavily inspired by "https://github.com/rwightman/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py"

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(layer, param, mean=0., std=1., a=-2., b=2.):
    # type : (Tensor, float, float, float, float) -> Tensor
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    tensor = layer.get_parameter(param)
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# Cell
class VisRecTrans :
    """Class for setting up a vision transformer for visual recognition.
    Returns a pretrained custom ViT model for the given `model_name` and `num_classes`, by default, or, with randomly initialized parameters, if `pretrained`
    is set to False.
    """

    models_list = ['vit_large_patch16_224', 'vit_large_patch16_224_in21k', 'vit_huge_patch14_224_in21k', 'vit_small_patch16_224', 'vit_small_patch16_224_in21k']
    # Two tasks : (1) Generalize the assignments of num_path (2) (3) ()
    def __init__(self, model_name, num_classes, pretrained = True) :
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        if self.model_name == 'vit_small_patch16_224' :
            self.num_patches = 196
            self.embed_dim = 384
            self.ni = 384

        elif self.model_name == 'vit_small_patch16_224_in21k' :
            self.num_patches = 196
            self.embed_dim = 384
            self.ni = 384

        elif self.model_name == 'vit_large_patch16_224' :
            self.num_patches = 196
            self.embed_dim = 1024
            self.ni = 1024

        elif self.model_name == 'vit_large_patch16_224_in21k' :
            self.num_patches = 196
            self.embed_dim = 1024
            self.ni = 1024

        elif self.model_name == 'vit_huge_patch14_224_in21k' :
            self.num_patches = 256
            self.embed_dim = 1280
            self.ni = 1280

    def create_model (self) :
        """Method for creating the model.
        """
        return custom_ViT(self.model_name, self.num_patches, self.embed_dim, self.ni, self.num_classes, self.pretrained)

    def initialize (self, model) :
        """Mthod for initializing the given `model`. This method uses truncated normal distribution for
        initializing the position embedding as well as the class token, and, the head of the model is
        initialized using He initialization.
        """
        trunc_normal_(model[1], 'cls_tokens')
        trunc_normal_(model[1], 'pos_embeds')
        apply_init(model[3], nn.init.kaiming_normal_)