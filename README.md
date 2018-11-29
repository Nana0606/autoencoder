# Autoencoder
主要包括自编码器及其变形的理论+实践。

因时间原因，代码中epoch设置的较小，实际状况下，肯定要更大。

## 主要内容
暂时代码包括普通自编码器（Autoencoder.py）、栈式自编码器（StackAutoencoder）、稀疏自编码器（SparseAutoencoder.py）和去噪自编码器（DenoisingAutoencoder.py）的简单实现，代码每一步都有注释。

关于收缩自编码器、变分自编码器、CNN自编码器等后更。

- 基于框架：Keras2.0.4
- 数据集：Mnist

具体设置等请参见代码或者博客

## 代码运行结果：

### 1、普通自编码器：

- 简单自动编码器架构图
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/ae_structure.png" width="60%" alt="自动编码器架构图"/></div>

- Encoder层输出结果可视化
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/ae_encoder_result.png" width="40%" alt="自动编码器Encoder层输出结果可视化"/></div>

- Autoencoder生成图片和原图片对比
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/ae_generate_comparison.png" width="90%" alt="自动编码器生成图片和原图片对比"/></div>

### 2、栈式自编码器：

- 栈式自动编码器架构图
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/stackAe_structure.png" width="60%" alt="栈式自动编码器架构图"/></div>

- Encoder层输出结果可视化
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/stackAe_encoder_result.png" width="40%" alt="栈式自动编码器Encoder层输出结果可视化"/></div>

- Stack Autoencoder生成图片和原图片对比
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/stackAe_generate_comparison.png" width="90%" alt="栈式自动编码器生成图片和原图片对比"/></div>

### 3、稀疏自编码器：

- 稀疏自动编码器架构图
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/sparseAe_structure.png" width="60%" alt="稀疏自动编码器架构图"/></div>

- Encoder层输出结果可视化
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/sparseAe_encoder_result.png" width="40%" alt="栈式自动编码器Encoder层输出结果可视化"/></div>

- Sparse Autoencoder生成图片和原图片对比
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/sparseAe_generate_comparison.png" width="90%" alt="栈式自动编码器生成图片和原图片对比"/></div>

### 4、去噪自编码器：

- 去噪自动编码器架构图
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/denoisingAe_structure.png" width="60%" alt="栈式自动编码器架构图"/></div>

- Encoder层输出结果可视化
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/denoisingAe_encoder_result.png" width="40%" alt="栈式自动编码器Encoder层输出结果可视化"/></div>

- Denoising Autoencoder原图片和添加噪声后图片对比
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/noising_data.png" width="90%" alt="栈式自动编码器原图片和添加噪声后图片对比"/></div>

- Denoising Autoencoder生成图片和原图片对比
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/denoisingAe_generate_comparison.png" width="90%" alt="栈式自动编码器生成图片和原图片对比"/></div>

### 5、卷积自编码器：

- 卷积自动编码器架构图
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/convAe_structure.png" width="60%" alt="卷积自动编码器架构图"/></div>

- Convolutional Autoencoder生成图片和原图片对比
<div align=center><img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/convAe_generate_comparison.png" width="90%" alt="卷积自动编码器生成图片和原图片对比"/></div>

- Convolutional Autoencoder训练accuracy和loss变化图
<div align=center>
  <img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/convAe_accuracy.png" width="45%" alt="卷积自动编码器accuracy变化"/>
  <img src="https://github.com/Nana0606/Autoencoder/blob/master/imgs/convAe_loss.png" width="45%" alt="卷积自动编码器loss变化"/>
</div>

## PDF整理
PDF来源于本人的理解+整理，部分图片来源于网上，已有标注，PDF对应博客详见：https://blog.csdn.net/quiet_girl/article/details/84401029 。