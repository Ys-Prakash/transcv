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
embed_callback = vis_rec_ob.get_callback()
```

Now, the `model`, along with the `embed_callback`, can be used with the [Learner](https://docs.fast.ai/learner.html#Learner) class, of [fastai](https://docs.fast.ai), and can be fine-tuned on any dataset. For the details of the visual recognition part, please see `VisRecTrans`.

### An example for using the SwinT class for visual recognition :

```python
from transcv.swin import SwinT
```

```python
swint_ob = SwinT('swin_base_patch4_window7_224', pretrained = False, num_classes = 10)
swin_model = swint_ob.get_model()
assert isinstance(swin_model, nn.Sequential)
```

Now, the `swin_model` can be used with the [Learner](https://docs.fast.ai/learner.html#Learner) class, of [fastai](https://docs.fast.ai), and can be fine-tuned on any dataset. For more details of the Swin transformer model, please see `SwinT`.
