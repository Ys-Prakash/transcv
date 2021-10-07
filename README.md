# transcv
> transcv creates vision transformers for visual recognition which can be pre-trained using self-supervised learning 


## Install

`pip install transcv`

Along with transcv, we also need fastai and nbdev. So, it is recommended to use :

`pip install fastai nbdev transcv -q --upgrade`

## How to use

### An example for using the VisRecTrans class for getting a custom ViT model :

```
from transcv.visrectrans import VisRecTrans
```

```
vis_rec_ob = VisRecTrans('vit_small_patch16_224', 10, False)
model = vis_rec_ob.create_model()
vis_rec_ob.initialize(model)
embed_callback = vis_rec_ob.get_callback()
```

Now, the `model`, along with the `embed_callback`, can be used with the [Learner](https://docs.fast.ai/learner.html#Learner) class, of [fastai](https://docs.fast.ai), and can be fine-tuned on any image classification dataset. For the details of the visual recognition part, please see `VisRecTrans`.

### An example for using the SwinT class for building a Swin transformer model :

```
from transcv.swin import SwinT
```

```
swint_ob = SwinT('swin_base_patch4_window7_224', pretrained = False, num_classes = 10)
swin_model = swint_ob.get_model()
assert isinstance(swin_model, nn.Sequential)
```

Now, the `swin_model` can be used with the [Learner](https://docs.fast.ai/learner.html#Learner) class, of [fastai](https://docs.fast.ai), and can be fine-tuned on any dataset for visual recognition task. For the details of the Swin transformer model, please see `SwinT`.
