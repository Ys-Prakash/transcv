# AUTOGENERATED! DO NOT EDIT! File to edit: 01_detr.ipynb (unless otherwise specified).

__all__ = ['DETR']

# Cell
#hide
from nbdev.showdoc import *
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from fastai.vision.all import *

# Cell
class DETR :
    """Class for setting up a Detection Transformer (DETR).
    """

    def __init__ (self) :
        self.feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
        self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

    def infer (self, img_paths):
        """Method for getting the bounding boxes and the predicted class logits. This method takes in a list of paths of images (`img_paths`) and
        returns a list of outputs.
        """
        if isinstance(img_paths, list) :
            outputs = []
            for path in img_paths :
                img = PILImage.create(path)
                input = self.feature_extractor(img, return_tensors = 'pt')
                output = self.model(**input)
                outputs.append(output)
                return outputs
        else :
            img = PILImage.create(img_paths)
            input = self.feature_extractor(img, return_tensors = 'pt')
            output = self.model(**input)
            return output

    def show_image (self, img_path) :
        """Method to display the image, for the given image path (`img_path`).
        """
        img = PILImage.create(img_path)
        img.show()