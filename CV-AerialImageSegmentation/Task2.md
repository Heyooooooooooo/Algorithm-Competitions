# Task2：数据扩增方法（3天）

- 学习主题：语义分割任务中数据扩增方法
- 学习内容：掌握语义分割任务中数据扩增方法的细节和使用
- 学习成果：数据扩增方法的实践

## 学习内容

具体学习内容 @[Datawhale](https://github.com/datawhalechina) 已发布在[这里](https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/Task2%EF%BC%9A%E6%95%B0%E6%8D%AE%E6%89%A9%E5%A2%9E%E6%96%B9%E6%B3%95.md)。

## 总结

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

### 为什么增强操作多了，反而效果更差了?

如下面这种变换，看上去做了很多操作，但是出现了一些问题。
```python
trfm = A.Compose([
     #改变HSV
     A.ColorJitter(p=0.5),
     A.HueSaturationValue(p=0.5),
     #resize
     A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    #旋转与缩放
     A.HorizontalFlip(p=0.5),
     A.VerticalFlip(p=0.5),
     A.RandomRotate90(),
     A.ShiftScaleRotate(scale_limit=(-0.3,-0.05),p=0.5),#随机旋转(-45~45),随机缩小0.3~0.05
     #透视变换
     A.IAAPerspective(scale=(0.03,0.05),p=0.5),#透视变化，随机放大0.03~0.05
     #加噪声
     A.ISONoise(p=0.5),
     A.GaussNoise(p=0.5),
     # 平滑处理
     A.GaussianBlur(blur_limit=3,p=0.5),#核大小给3，设置成更大，根本看不清了
 ])
```
概率为p=0.5的操作就有9个，如果要返回不进行操作的原图的概率为$\frac{1}{2}^9$，很多时候，得到的都不是原图。参加训练的原图也寥寥无几。因此在训练的时候，很多图片都是增强过度了，跟原图的相似度相差很大，从而导致验证集损失无法收敛。

于是，可以运用A.Compose([op],p=0.5)在尾部加上概率，保证原图的数量。
或者通过减少增强操作，来保证原图数量，比如baseline中的增强操作步骤还是比较少的，因此能有不错的效果。
或者可以选用A.OneOf([op],p=0.5)来组合操作，也可以在Compose中套Compose，在有效保证鲁棒性的同时，也避免了过度偏离原始数据（原图）。

## 致谢

感谢 @[Datawhale](https://github.com/datawhalechina) 制作的课程与分享的学习资料！

## 参考

1. [Datawhale学习教程](https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/Task2%EF%BC%9A%E6%95%B0%E6%8D%AE%E6%89%A9%E5%A2%9E%E6%96%B9%E6%B3%95.md)
2. [学习爱好者：菊头蝙蝠的分享](https://blog.csdn.net/qq_21539375/article/details/113943166)

