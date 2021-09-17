# transcv
> A library for creating custom vision transformers for computer vision problems


## Install

`pip install transcv`

## How to use

### An example for using the VisRecTrans API for getting a custom ViT model :

```python
from transcv.visrectrans import VisRecTrans
```

```python
vis_rec_ob = VisRecTrans('vit_small_patch16_224', 10, False)
model = vis_rec_ob.create_model()
vis_rec_ob.initialize(model)
```

Now, the `model` can be used with the `Learner` API, of fastai, and can be fine-tuned on any dataset. For the details of the visual recognition part, please see `VisRecTrans`.

### An example for using the VisRecTrans API for object detection :

```python
from transcv.detr import DETR
```

```python
path = untar_data(URLs.PASCAL_2007)
```

```python
#hide_output
detr_ob = DETR(path/'test'/'001407.jpg')
feature_extractr, model = detr_ob.get_detr()
output = detr_ob.infer(feature_extractr, model)
```

```python
output.pred_boxes
```




    tensor([[[0.0312, 0.6896, 0.0470, 0.1512],
             [0.4686, 0.6384, 0.0217, 0.0654],
             [0.1392, 0.8852, 0.1581, 0.2278],
             [0.5187, 0.6178, 0.0255, 0.0928],
             [0.1942, 0.6072, 0.1401, 0.0415],
             [0.2812, 0.8212, 0.1268, 0.1037],
             [0.2804, 0.8271, 0.1278, 0.0968],
             [0.1645, 0.7937, 0.1100, 0.0560],
             [0.1438, 0.5111, 0.0223, 0.0478],
             [0.1409, 0.5077, 0.0240, 0.0535],
             [0.5292, 0.9635, 0.1701, 0.0734],
             [0.1172, 0.7622, 0.1599, 0.1978],
             [0.9257, 0.8673, 0.1486, 0.2624],
             [0.9246, 0.8062, 0.1501, 0.1402],
             [0.2432, 0.9316, 0.2007, 0.1341],
             [0.0699, 0.6324, 0.0974, 0.0624],
             [0.0638, 0.5387, 0.1272, 0.0998],
             [0.8021, 0.4340, 0.1109, 0.2251],
             [0.8007, 0.5989, 0.1001, 0.0419],
             [0.0915, 0.6315, 0.1830, 0.0500],
             [0.9265, 0.8870, 0.1467, 0.2251],
             [0.9276, 0.8046, 0.1452, 0.1409],
             [0.0087, 0.5914, 0.0175, 0.0497],
             [0.9273, 0.9331, 0.1451, 0.1316],
             [0.5300, 0.6831, 0.2422, 0.0531],
             [0.4649, 0.9633, 0.4186, 0.0749],
             [0.0487, 0.6206, 0.0942, 0.0505],
             [0.1992, 0.8837, 0.2411, 0.2290],
             [0.2018, 0.6019, 0.1359, 0.0341],
             [0.0120, 0.6667, 0.0240, 0.1860],
             [0.0154, 0.6810, 0.0308, 0.1597],
             [0.2397, 0.9269, 0.2032, 0.1415],
             [0.2037, 0.9455, 0.1599, 0.1087],
             [0.0460, 0.6860, 0.0546, 0.1224],
             [0.9286, 0.8167, 0.1433, 0.1581],
             [0.2802, 0.8077, 0.1308, 0.1103],
             [0.9255, 0.8000, 0.1482, 0.1344],
             [0.1372, 0.7176, 0.1265, 0.1153],
             [0.2203, 0.8303, 0.2241, 0.1243],
             [0.7895, 0.5877, 0.0862, 0.0426],
             [0.8175, 0.5755, 0.0767, 0.0385],
             [0.8707, 0.5081, 0.2576, 0.4161],
             [0.5147, 0.7057, 0.9788, 0.2616],
             [0.2389, 0.8863, 0.2142, 0.2258],
             [0.0685, 0.6882, 0.0992, 0.1819],
             [0.2010, 0.8751, 0.2879, 0.2476],
             [0.8003, 0.5916, 0.1132, 0.0614],
             [0.5277, 0.9632, 0.1662, 0.0737],
             [0.0663, 0.6872, 0.0957, 0.1756],
             [0.1754, 0.7230, 0.0656, 0.0915],
             [0.1507, 0.5143, 0.0214, 0.0450],
             [0.8060, 0.4833, 0.1286, 0.3272],
             [0.2390, 0.9365, 0.2186, 0.1267],
             [0.1262, 0.7490, 0.1461, 0.1977],
             [0.1546, 0.8062, 0.1233, 0.0752],
             [0.1309, 0.7816, 0.1414, 0.2387],
             [0.0704, 0.6889, 0.0949, 0.1664],
             [0.4696, 0.6609, 0.0247, 0.1055],
             [0.4309, 0.9210, 0.5776, 0.1583],
             [0.5187, 0.5372, 0.9614, 0.9180],
             [0.1795, 0.6197, 0.1577, 0.0745],
             [0.1322, 0.8314, 0.1503, 0.3318],
             [0.3978, 0.5464, 0.1722, 0.0743],
             [0.1193, 0.6406, 0.0417, 0.0401],
             [0.8729, 0.4517, 0.2538, 0.2856],
             [0.1613, 0.8661, 0.1173, 0.1819],
             [0.0086, 0.5891, 0.0171, 0.0376],
             [0.1281, 0.8297, 0.1554, 0.3335],
             [0.1438, 0.6557, 0.0782, 0.0440],
             [0.0109, 0.6569, 0.0217, 0.1143],
             [0.9851, 0.5244, 0.0298, 0.0967],
             [0.3636, 0.9294, 0.4896, 0.1423],
             [0.1291, 0.7488, 0.1502, 0.0830],
             [0.6227, 0.6896, 0.0494, 0.0315],
             [0.5279, 0.9640, 0.1649, 0.0720],
             [0.4997, 0.8700, 1.0000, 0.2559],
             [0.1308, 0.7633, 0.1402, 0.2084],
             [0.0599, 0.7646, 0.0473, 0.1865],
             [0.1920, 0.6073, 0.1455, 0.0394],
             [0.0263, 0.6209, 0.0525, 0.0325],
             [0.9253, 0.8684, 0.1495, 0.2601],
             [0.2798, 0.8211, 0.1344, 0.1043],
             [0.9923, 0.8932, 0.0154, 0.2121],
             [0.2181, 0.6101, 0.2218, 0.0769],
             [0.1358, 0.7033, 0.1247, 0.0951],
             [0.9288, 0.9366, 0.1427, 0.1257],
             [0.0477, 0.7370, 0.0407, 0.1320],
             [0.5050, 0.6328, 0.7798, 0.1395],
             [0.2261, 0.9319, 0.2362, 0.1363],
             [0.2852, 0.8553, 0.1349, 0.1586],
             [0.5353, 0.8845, 0.9227, 0.2293],
             [0.8114, 0.4833, 0.1457, 0.3277],
             [0.5983, 0.9492, 0.7848, 0.1025],
             [0.4899, 0.6325, 0.0563, 0.0943],
             [0.4576, 0.9727, 0.4862, 0.0548],
             [0.3157, 0.9329, 0.4226, 0.1344],
             [0.0779, 0.6372, 0.0834, 0.0733],
             [0.8622, 0.4802, 0.2745, 0.3325],
             [0.5740, 0.5695, 0.8270, 0.8567],
             [0.5199, 0.6161, 0.0286, 0.1010]]], grad_fn=<SigmoidBackward>)



For more details of the object detection part, please see `DETR`
