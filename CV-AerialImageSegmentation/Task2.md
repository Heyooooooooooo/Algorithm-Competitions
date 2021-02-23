# Task2：数据扩增方法（3天）

- 学习主题：语义分割任务中数据扩增方法
- 学习内容：掌握语义分割任务中数据扩增方法的细节和使用
- 学习成果：数据扩增方法的实践

## 学习内容

具体学习内容 @[Datawhale](https://github.com/datawhalechina) 已发布在[这里](https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/Task2%EF%BC%9A%E6%95%B0%E6%8D%AE%E6%89%A9%E5%A2%9E%E6%96%B9%E6%B3%95.md)。

## zongjie

### albumentations的例子

```python
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import numpy as np

def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)

image = np.ones((300, 300, 3), dtype=np.uint8)
mask = np.ones((300, 300), dtype=np.uint8)
whatever_data = "my name"
augmentation = strong_aug(p=0.9)
data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
augmented = augmentation(**data)
image, mask, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], augmented["additional"]
```
如`Oneof`,它会选择里面的其中之一增样操作，权重是按照，每个增样操作的概率。

```python
OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2)
```
如里面3个他们的权重为3:1:3,于是$p=0.2$的概率(事件A)会从里面抽一个增样，抽中OpticalDistortion(p=0.3)（事件B），的条件概率$p(B|A)=\frac{3}{7}$


