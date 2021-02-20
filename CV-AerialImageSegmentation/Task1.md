# Task1：赛题理解

- 学习主题：理解赛题内容解题流程
- 学习内容：赛题理解、数据读取、比赛baseline构建
- 学习成果：比赛baseline提交

## 学习内容

具体学习内容 @[Datawhale](https://github.com/datawhalechina) 已发布在[这里](https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/Task1%EF%BC%9A%E8%B5%9B%E9%A2%98%E7%90%86%E8%A7%A3.md)。

## 课后作业

### 1. 理解RLE编码过程，并完成赛题数据读取并可视化

以下代码来源于@[Datawhale](https://github.com/datawhalechina) 的 [Baseline](https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/baseline.ipynb) 教程。

1. 将图片编码为rle格式

	```python
	import numpy as np
	import pandas as pd
	import cv2
	
	# 将图片编码为rle格式
	def rle_encode(im):
	    '''
	    im: numpy array, 1 - mask, 0 - background
	    Returns run length as string formated
	    '''
	    pixels = im.flatten(order = 'F')
	    pixels = np.concatenate([[0], pixels, [0]])
	    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
	    runs[1::2] -= runs[::2]
	    return ' '.join(str(x) for x in runs)
	```

2. 将rle格式进行解码为图片

	```python
	def rle_decode(mask_rle, shape=(512, 512)):
	    '''
	    mask_rle: run-length as string formated (start length)
	    shape: (height,width) of array to return 
	    Returns numpy array, 1 - mask, 0 - background
	
	    '''
	    s = mask_rle.split()
	    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
	    starts -= 1
	    ends = starts + lengths
	    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
	    for lo, hi in zip(starts, ends):
	        img[lo:hi] = 1
	    return img.reshape(shape, order='F')
	```

3. 定义数据集

	```python
	class TianChiDataset(D.Dataset):
	    def __init__(self, paths, rles, transform, test_mode=False):
	        self.paths = paths
	        self.rles = rles
	        self.transform = transform
	        self.test_mode = test_mode
	        
	        self.len = len(paths)
	        self.as_tensor = T.Compose([
	            T.ToPILImage(),
	            T.Resize(IMAGE_SIZE),
	            T.ToTensor(),
	            T.Normalize([0.625, 0.448, 0.688],
	                        [0.131, 0.177, 0.101]),
	        ])
	        
	    # get data operation
	    def __getitem__(self, index):
	        #img = cv2.imread(self.paths[index])
	        img = np.array(Image.open(self.paths[index]))
	        
	        if not self.test_mode:
	            mask = rle_decode(self.rles[index])
	            augments = self.transform(image=img, mask=mask)
	            return self.as_tensor(augments['image']), augments['mask'][None]#（3，256，256），（1，256，256）
	        else:
	            return self.as_tensor(img), ''        
	    
	    def __len__(self):
	        """
	        Total number of samples in the dataset
	        """
	        return self.len
	```

4. 可视化

	```python
	image, mask = dataset[0]
	plt.figure(figsize=(16,8))
	plt.subplot(121)
	plt.imshow(mask[0], cmap='gray')
	plt.subplot(122)
	plt.imshow(image[0])
	plt.show()# 补上
	```

	结果如下：

	

### 2. 统计所有图片整图中没有任何建筑物像素占所有训练集图片的比例

```python
import pandas as pd

train_mask = pd.read_csv("数据集/train_mask.csv",sep="\t",names=["name","mask"])
train_mask["mask"]=train_mask["mask"].fillna("")
l = len(train_mask)
sum=0
for i in range(l):
    if train_mask["mask"].iloc[i]=="":
        sum+=1
print(sum/l)
```

结果如下：

```python
# 得到输出的结果(没有任何建筑物像素占所有训练集图片的比例)
>>>0.17346666666666666
# 30000张图中有5204张图是没有任何建筑的
```

### 3. 统计所有图片中建筑物像素占所有像素的比例

```python
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


train_mask = pd.read_csv("数据集/train_mask.csv",sep="\t",names=["name","mask"])
train_mask["mask"]=train_mask["mask"].fillna("")
l = len(train_mask)

ratio_ls = []
for i in tqdm(range(l)):
    if train_mask["mask"].iloc[i]!="":
        ls = list(map(int,train_mask["mask"].iloc[i].split(" ")))
        number = sum(ls[1::2])
        pic_path = "数据集/"+"train/"+train_mask["name"].iloc[i]
        img = np.array(Image.open(pic_path))
        ratio = number/(img.shape[0]*img.shape[1])
    else:
        ratio = 0

    ratio_ls.append(ratio)
pd.Series(ratio_ls).to_csv("ratio_ls")
```

将每张图片的建筑物像素占所有像素的比例存入到列表中，并保存下来。

```python
ratio = pd.read_csv("ratio_ls")
print("所有图片中建筑像素平均占比:",np.mean(ratio.iloc[:,1]))
ratio_ = ratio.iloc[:,1][(ratio.iloc[:,1])!=0]
print("有建筑图片中建筑像素平均占比:",np.mean(ratio_))
ratio = np.array(ratio)[:,1]
print("建筑像素占比最大值",np.max(ratio))
print("有建筑图片中,建筑像素占比最小值",np.min(ratio_))
```

结果如下：
!image[](./pic/可视化.jpg)

```
所有图片中建筑像素平均占比: 0.15708140207926433
有建筑图片中建筑像素平均占比: 0.19004847807621914
建筑像素占比最大值 0.9992218017578124
有建筑图片中,建筑像素占比最小值 3.814697265625e-06
```

### 4. 统计所有图片中建筑物区域平均区域大小

区域大小，也就是白色像素点数量和。

```python
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

train_mask = pd.read_csv("数据集/train_mask.csv",sep="\t",names=["name","mask"])
train_mask["mask"]=train_mask["mask"].fillna("")
l = len(train_mask)

sum_ls = []
for i in tqdm(range(l)):
    if train_mask["mask"].iloc[i]!="":
        ls = list(map(int,train_mask["mask"].iloc[i].split(" ")))
        number = sum(ls[1::2])
        # pic_path = "数据集/"+"train/"+train_mask["name"].iloc[i]
        # img = np.array(Image.open(pic_path))
        # ratio = number/(img.shape[0]*img.shape[1])
    else:
        number = 0
    sum_ls.append(number)
pd.Series(sum_ls).to_csv("point_sum_ls")
```

```python
ls = pd.read_csv("point_sum_ls")
print(np.mean(ls.iloc[:,1]))
ls_ = ls[(ls.iloc[:,1])!=0]
print(np.mean(ls_.iloc[:,1]))
```

结果如下：

```python
所有图中建筑物区域平均区域大小 41177.94706666667
含有建筑物的图中建筑物区域平均区域大小 49820.06823681239
```

## 致谢

感谢 @[Datawhale](https://github.com/datawhalechina) 制作的课程与分享的学习资料！

## 参考

1. [Baseline](https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/baseline.ipynb)
2. [Datawhale学习教程](https://github.com/datawhalechina/team-learning-cv/blob/master/AerialImageSegmentation/baseline.ipynb)
3. [学习爱好者：菊头蝙蝠的分享](https://blog.csdn.net/qq_21539375/article/details/113825571)
