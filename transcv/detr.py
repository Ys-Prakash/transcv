# AUTOGENERATED! DO NOT EDIT! File to edit: 01_detr.ipynb (unless otherwise specified).

__all__ = ['DETR']

# Cell
#hide
from nbdev.showdoc import *
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from fastai.vision.all import *

# Cell
class DETR :
    """Class for setting up a detection transformer for object detection.

    Attributes :
        img_path : Path to the image
    """

    def __init__ (self, img_path) :
        self.img_path = img_path
        self.test_img = PILImage.create(self.img_path)

    def get_detr (self) :
        """Method for getting the detr model.
        """
        feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
        model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
        return feature_extractor, model

    def infer (self, feature_extractor, model):
        """Mthod for getting the bounding boxes.

        Attributes :
            feature_extractor : The feature extractor, which is returned by the `get_dter` method
            model : The detr model, returned by the `get_detr` method
        """
        input = feature_extractor(self.test_img, return_tensors = 'pt')
        output = model(**input)
        return output

    def show_image (self) :
        """Method to display the image
        """
        self.test_img.show()