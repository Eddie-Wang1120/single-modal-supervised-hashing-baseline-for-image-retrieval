SSHBIR Toolkit
==============

## What is SSHBIR?
SSHBIR is a baseline frame toolkit including 10 different single-model supervised hashing methods.The toolkit can draw a mAP curves and Time curves for each methods in diverse bit number, which can offer help to all learners for hashing image retrieval when you want to get to know the detail codes and differences for single-model supervised hashing methods.I also made a summary for recent single-model supervised hashing methods, you can check it in my other project [Single-Modal-Supervised-Hashing-Methods-Papers-Collection](https://github.com/Eddie-Wang1120/Single-Modal-Supervised-Hashing-Methods-Papers-Collection).

## Why build SSHBIR?
When I begin to learn about hashing methods for image retrieval, I was greatly helped by the [HABIR Toolkit](https://github.com/willard-yuan/hashing-baseline-for-image-retrieval), which includes lots of unsupervised hashing methods. However, after my serach, I found out that basically there is rarely a ready-made frame for all the supervised hashing methods. Therefore, the SSHBIR toolkit was built. Hopes this toolkit can give a hand and welcome for all the suggestion.

## How to use SSHBIR?
**Database**<br>  

The SSHBIR uses the processed CIFAR10-Gist512 as database, which had already split into trainset and testset, and I put the link of both the original one and the processed one below for better understanding.<br>

* [[original Cifar-10]](https://www.cs.toronto.edu/~kriz/cifar.html)
* [[processed Cifar-10 code:3smq **used in this toolkit**]](https://pan.baidu.com/s/1fZQihwP4TDgrZThndH__Pw)

When you already download the processed CIFAR-10.mat, put it in the SSHBIR Toolkit directory like the picture example.<br>

![image](https://github.com/Eddie-Wang1120/single-modal-supervised-hashing-baseline-for-image-retrieval/blob/main/img/pos.png)

**Run demo**<br>  
`main_demo.m` : The main script to evaluate the performance.If changing the hashing methods and the bit number in this demo, you can get corresponding mAP curves and Time curves.<br>

## Hashing methods in this toolkit
* **COSDISH**: Wang-Cheng Kang, Wu-Jun Li, Zhi-Hua Zhou."Column Sampling Based Discrete Supervised Hashing" (2016)Proceedings of the AAAI Conference on Artificial Intelligence
* **FSDH**: Jie Gui, Tongliang Liu, Zhenan Sun, Dacheng Tao, Tieniu Tan."Fast Supervised Discrete Hashing" (2017)IEEE Trans on Pattern Analysis and Machine Intelligence
* **FSSH**: Xin Luo, Liqiang Nie, Xiangnan He, Ye Wu, Zhen-Duo Chen, Xin-Shun Xu."Fast Scalable Supervised Hashing" (2018)International Conference on Research on Development in Information Retrieval
* **KSH**: Wei Liu, Jun Wang, Rongrong Ji, Yu-Gang Jiang, Shih-Fu Chang."Supervised hashing with kernels"(2012)IEEE Conference on Computer Vision and Pattern Recognition
* **LFH**: Peichao Zhang, Wei Zhang, Wujun Li,Minyi Guo."Supervised hashing with latent factor models" (2014)International Conference on Research on Development in Information Retrieval
* **POSH**: Zheng Zhang, Xiaofeng Zhu, Guangming Lu, Yudong Zhang."Probability Ordinal-Preserving Semantic Hashing for Large-Scale Image Retrieval" (2021)ACM Transactions on Knowledge Discovery from Data
* **RSLH**: Xingbo Liu, Xiushan Nie, Qi Dai, Yupan Huang, Li Lian, Yilong Yin."Reinforced Short-Length Hashing" (2020)IEEE Transactions on Circuits and Systems for Video Technology
* **SCDH**: Yong Chen, Zhibao Tian, Hui Zhang, Jun Wang, Dell Zhang."Strongly Constrained Discrete Hashing" (2020)IEEE Transactions on Image Processing
* **SDH**: Fumin Shen, Chunhua Shen, Wei Liu, Heng Tao Shen."Supervised Discrete Hashing" (2015)Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
* **SSLH**: Xingbo Liu, Xiushan Nie, Quan Zhou, Xiaoming Xi, Lei Zhu, Yilong Yin."Supervised Short-Length Hashing" (2019)International Joint Conference on Artificial Intelligence
* **More single-model supervised hashing methods you can check it in my other project [[Single-Modal-Supervised-Hashing-Methods-Papers-Collection]](https://github.com/Eddie-Wang1120/Single-Modal-Supervised-Hashing-Methods-Papers-Collection)**

## Environment
* Matlab

## Version Update
* 1.0 5/11/2020 basic version
