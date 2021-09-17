# VisRecTrans
> A library for creating custom vision transformers for computer vision problems


## Install

`pip install transcv`

## How to use

Some examples :

```python
from transcv.visrectrans import VisRecTrans
```

```python
vis_rec_ob = VisRecTrans('vit_small_patch16_224', 10, False)
model = vis_rec_ob.create_model()
vis_rec_ob.initialize(model)
```

Now, the `model` can be used with the `Learner` API, of fastai, and can be fine-tuned on any dataset.
