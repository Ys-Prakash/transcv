# transcv
> A library for creating custom vision transformers for computer vision


## Install

`pip install transcv`

Along with transcv, we also need fastai and nbdev. So, it is recommended to use :

`pip install fastai nbdev transcv -q --upgrade`

## How to use

### An example for using the VisRecTrans class for getting a custom ViT model :

```python
from transcv.visrectrans import VisRecTrans
```

```python
vis_rec_ob = VisRecTrans('vit_small_patch16_224', 10, False)
model = vis_rec_ob.create_model()
vis_rec_ob.initialize(model)
```

Now, the `model` can be used with the [Learner](https://docs.fast.ai/learner.html#Learner) class, of [fastai](https://docs.fast.ai), and can be fine-tuned on any dataset. For the details of the visual recognition part, please see `VisRecTrans`.

### An example for using the DETR class for object detection :

```python
from transcv.detr import DETR
```

```python
path = untar_data(URLs.PASCAL_2007)
```

```python
#hide_output
detr_ob = DETR()
files = get_image_files(path/'test')
output = detr_ob.infer(files[0])
```

```python
output.pred_boxes
```




    tensor([[[0.4998, 0.7705, 1.0000, 0.4544],
             [0.5043, 0.5217, 0.7801, 0.4690],
             [0.3807, 0.8267, 0.7597, 0.3410],
             [0.5103, 0.4983, 0.7999, 0.4244],
             [0.5093, 0.4782, 0.7939, 0.3937],
             [0.3961, 0.7677, 0.7890, 0.4582],
             [0.5246, 0.8235, 0.5314, 0.3467],
             [0.2041, 0.9911, 0.4032, 0.0178],
             [0.1740, 0.0186, 0.3439, 0.0373],
             [0.4996, 0.0453, 1.0000, 0.0910],
             [0.4998, 0.8150, 1.0000, 0.3707],
             [0.4998, 0.8286, 1.0000, 0.3427],
             [0.5000, 0.7444, 1.0000, 0.5037],
             [0.8419, 0.8722, 0.3138, 0.2496],
             [0.5210, 0.9930, 0.4287, 0.0142],
             [0.4999, 0.6016, 0.9998, 0.7827],
             [0.0439, 0.0317, 0.0879, 0.0639],
             [0.7768, 0.4815, 0.2954, 0.4026],
             [0.8052, 0.8714, 0.3888, 0.2518],
             [0.4998, 0.5979, 0.9999, 0.7937],
             [0.4998, 0.7983, 1.0000, 0.4052],
             [0.4999, 0.7469, 1.0000, 0.5034],
             [0.4998, 0.5003, 1.0000, 1.0000],
             [0.8520, 0.8738, 0.2940, 0.2474],
             [0.5773, 0.6394, 0.6527, 0.7037],
             [0.5218, 0.8176, 0.5295, 0.3592],
             [0.5000, 0.6934, 1.0000, 0.6041],
             [0.3998, 0.6806, 0.7965, 0.6301],
             [0.3794, 0.4746, 0.5195, 0.3769],
             [0.0156, 0.4003, 0.0319, 0.7608],
             [0.4998, 0.7542, 1.0000, 0.4888],
             [0.3918, 0.8151, 0.7788, 0.3618],
             [0.3112, 0.9934, 0.0887, 0.0130],
             [0.8570, 0.8105, 0.2809, 0.3686],
             [0.5000, 0.6819, 1.0000, 0.6284],
             [0.4546, 0.6457, 0.6633, 0.6903],
             [0.8498, 0.7406, 0.3007, 0.5083],
             [0.3962, 0.6223, 0.7886, 0.7386],
             [0.3442, 0.9925, 0.1389, 0.0149],
             [0.7520, 0.4876, 0.3147, 0.3965],
             [0.8243, 0.6926, 0.3515, 0.6104],
             [0.4999, 0.5252, 1.0000, 0.9439],
             [0.5121, 0.4769, 0.8002, 0.3832],
             [0.3940, 0.7672, 0.7853, 0.4625],
             [0.5000, 0.7093, 0.9999, 0.5740],
             [0.4765, 0.5770, 0.8888, 0.8446],
             [0.6198, 0.6417, 0.7562, 0.7060],
             [0.5226, 0.8227, 0.5242, 0.3511],
             [0.4999, 0.6741, 1.0000, 0.6480],
             [0.3348, 0.9906, 0.1113, 0.0187],
             [0.4855, 0.6185, 0.9420, 0.7526],
             [0.5053, 0.5330, 0.7910, 0.4980],
             [0.3772, 0.8669, 0.7514, 0.2598],
             [0.3653, 0.4746, 0.5010, 0.3934],
             [0.1746, 0.8756, 0.3465, 0.2435],
             [0.3967, 0.7395, 0.7900, 0.5125],
             [0.4998, 0.8212, 1.0000, 0.3595],
             [0.5247, 0.7652, 0.5449, 0.4635],
             [0.5120, 0.6103, 0.7933, 0.7677],
             [0.4999, 0.6341, 0.9999, 0.7210],
             [0.2922, 0.4744, 0.3562, 0.3774],
             [0.4998, 0.6294, 0.9999, 0.7332],
             [0.0234, 0.0415, 0.0469, 0.0835],
             [0.4999, 0.6015, 1.0000, 0.7938],
             [0.1635, 0.0347, 0.3350, 0.0694],
             [0.3812, 0.8496, 0.7615, 0.2946],
             [0.0130, 0.0446, 0.0261, 0.0894],
             [0.5000, 0.7145, 1.0000, 0.5630],
             [0.4868, 0.6310, 0.8003, 0.7174],
             [0.4999, 0.7019, 1.0000, 0.5882],
             [0.4999, 0.6661, 1.0000, 0.6626],
             [0.5109, 0.6371, 0.8017, 0.7140],
             [0.4998, 0.8484, 0.9999, 0.2942],
             [0.6447, 0.8675, 0.7109, 0.2595],
             [0.4998, 0.8647, 1.0000, 0.2632],
             [0.5706, 0.6398, 0.6632, 0.7016],
             [0.1405, 0.7332, 0.2836, 0.5162],
             [0.1574, 0.8649, 0.3111, 0.2619],
             [0.3545, 0.4766, 0.4844, 0.3859],
             [0.0271, 0.0428, 0.0545, 0.0863],
             [0.8507, 0.8732, 0.2963, 0.2478],
             [0.3985, 0.7934, 0.7944, 0.4019],
             [0.4998, 0.7244, 1.0000, 0.5434],
             [0.4574, 0.4849, 0.8999, 0.4180],
             [0.1710, 0.7705, 0.3398, 0.4535],
             [0.7915, 0.8725, 0.4162, 0.2509],
             [0.8555, 0.8383, 0.2876, 0.3180],
             [0.5122, 0.5909, 0.8280, 0.8085],
             [0.4998, 0.8275, 1.0000, 0.3406],
             [0.5201, 0.8177, 0.5340, 0.3551],
             [0.5216, 0.7953, 0.5380, 0.4036],
             [0.4999, 0.6047, 1.0000, 0.7804],
             [0.3957, 0.7803, 0.7892, 0.4359],
             [0.5497, 0.6499, 0.6169, 0.6854],
             [0.5231, 0.7744, 0.5388, 0.4480],
             [0.3924, 0.8166, 0.7849, 0.3596],
             [0.1247, 0.6477, 0.2503, 0.3493],
             [0.5079, 0.4778, 0.8009, 0.3844],
             [0.4994, 0.5652, 0.9769, 0.8688],
             [0.5093, 0.4876, 0.7867, 0.4046]]], grad_fn=<SigmoidBackward>)



For more details of the object detection part, please see `DETR`.
