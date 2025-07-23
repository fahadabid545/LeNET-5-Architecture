# LeNet-5 Architecture 

A simple and powerful Convolutional Neural Network (CNN) architecture developed by **Yann LeCun et al.** in 1998, primarily used for **handwritten digit recognition**. This model is considered one of the foundational works in deep learning for computer vision.

---

## 🏗️ Architecture Overview

LeNet-5 consists of **7 layers** (excluding input), including convolutional, subsampling (pooling), and fully connected layers.

### 🔍 Layer-wise Breakdown

| Layer        | Type               | Parameters                                  | Output Size     |
|--------------|--------------------|----------------------------------------------|-----------------|
| Input        | Image              | 32 × 32 Grayscale                            | 32×32×1         |
| C1           | Convolution        | (5×5×1)×6 + 6 = **156**                      | 28×28×6         |
| S2           | Average Pooling    | No trainable parameters                      | 14×14×6         |
| C3           | Convolution        | (5×5×6)×16 + 16 = **2,416**                  | 10×10×16        |
| S4           | Average Pooling    | No trainable parameters                      | 5×5×16          |
| C5           | Convolution        | (5×5×16)×120 + 120 = **48,120**              | 1×1×120         |
| F6           | Fully Connected    | 120×84 + 84 = **10,164**                     | 84              |
| Output       | Fully Connected    | 84×10 + 10 = **850**                         | 10              |


---

## ⚙️ Technologies & Libraries

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-323754.svg?style=flat&logo=matplotlib&logoColor=white)

--- 

## 🔧 Implementation Outline (Keras)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

model = Sequential([
    Conv2D(6, kernel_size=5, activation='tanh', input_shape=(32, 32, 1)),
    AveragePooling2D(pool_size=2),
    Conv2D(16, kernel_size=5, activation='tanh'),
    AveragePooling2D(pool_size=2),
    Conv2D(120, kernel_size=5, activation='tanh'),
    Flatten(),
    Dense(84, activation='tanh'),
    Dense(10, activation='softmax')
])





