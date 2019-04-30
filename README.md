# Musical Genre Classification

We discuss the application of convolutional neural networks for the task of music genre classification. 
We focus in the small data-set case and how to build CNN architecture. We start with data augmentation strategy in music domain, and compare well-known architecture in the 1D, 2D, sample CNN with law data and augmented-data.
Moreover, we suggest best performance CNN architecture in small-data music-genre classification. Then, we compare normalization method and optimizers. we will be discussed to see how to obtain the model that fits better in the music genre classification. 
Finally, we evaluate its performance in the GTZAN dataset, used in a lot of works, in order to compare the performance of our approach with the state of the art.

### Dataset
We use the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset which has been the most widely used in the music genre classification task. 
The dataset contains 30-second audio files including 10 different genres including reggae, classical, country, jazz, metal, pop, disco, hiphop, rock and blues. 
For this homework, we are going to use a subset of GTZAN with only 8 genres. You can download the subset from [this link](https://drive.google.com/file/d/1rHw-1NR_Taoz6kTfJ4MPR5YTxyoCed1W/view?usp=sharing).

Once you downloaded the dataset, unzip and move the dataset to your home folder. After you have done this, you should have the following content in the dataset folder.  

### Data augmentation
Data augmentation is the process by which we create new synthetic training samples by adding small perturbations on our initial training set. 
The objective is to make our model invariant to those perturbations and enhance its ability to generalize. In order to this to work adding the perturbations must conserve the same label as the original training sample.
- Add Noise
- Shift
- Speed Change
- Pitch Shift
- Pitch and Speed
- Multiply Value
- Percussive

<img src="/img/augmentation.png">
<img src="/img/mel.png">

### Result
The model with the best validation accuracy is the 4Layer CNN with 77%. The test accuracy of this model is 83.39%. 
Sample_rate 22050 used in feature engineering, fft size 1024, win size 1024, hop size 512, num mels 128, feature length 1024. 
We also recorded 26 epochs based on early stop criteria. Stochastic gradient descent was used, and learning rate 0.01, momentum 0.9, weight decay 1e-6, using nesterov showed the best performance.

<img src="/img/cnn.png">

Model | Train Acc | Valid Acc  | Train Acc(Augmented) | Valid Acc(Augmented) | Test Acc
:---:|:---:|:---:|:---:|:---:|:---:
5L-1D CNN | 0.97 | 0.55 | 0.99 | 0.70
AlexNet | 0.98 | 0.63 | 0.99 | 0.72 
VGG11 | 0.99 | 0.68 | 0.99 | 0.76
VGG13 | 0.97 | 0.68 | 0.99 | 0.74 
VGG16 | 0.99 | 0.69 | 0.99 | 0.75 
VGG19 | 0.98 | 0.67 | 0.99 | 0.74 
GooLeNet | 0.75 | 0.57 | 0.99 | 0.65 
ResNet34 | 0.99 | 0.63 | 0.99 | 0.70 
ResNet50 | 0.99 | 0.61 | 0.99 | 0.69 
DenseNet | 0.98 | 0.66 | 0.99 | 0.76
Sample CNN Basic Block | 0.13 | 0.13 | 0.15 | 0.13 
4L-2D CNN | 0.93 | 0.62 | 0.95 | 0.77 | 83.39
4L-2D CNN + GRU | 0.92 | 0.64 | 0.99 | 0.76 | 81.55

### Experiments

* [x] [Custom 1D CNN]() (5Layer 1D-CNN)
* [x] [Alexnet]() (Alexnet)
* [x] [Very Deep Convolutional Networks for Large-Scale Image Recognition]()(Vgg16)
  + https://arxiv.org/abs/1409.1556
* [x] [Goin deeper with convolutions]() (Inception)
  + https://arxiv.org/abs/1409.4842
* [x] [Deep Residual Learning for Image Recognition]()(ResNet)
  + https://arxiv.org/abs/1512.03385
* [x] [Sample-level CNN Architectures for Music Auto-tagging Using Raw Waveforms]() (SampleCNN)
  + https://arxiv.org/abs/1710.10451
* [x] [Custom 2D CNN]() (4Layer 2D-CNN)
* [x] [Custom 2D CNN + GRU]() (4Layer 2D-CNN + GRU)

### Requirements 
Before you run baseline code you will need PyTorch. Please install PyTorch from [this link](https://pytorch.org/get-started/locally/).
We will use PyTorch 1.0 because it is the first official version.

* Python 3.7 (recommended)
* Numpy
* Librosa
* PyTorch 1.0

### Learning code
First, you augmentation data
```
$ audio_augmentation.py
```
Second, Feature extraction using mel-spectogram
```
$ feature_extraction.py
```   
Check hparams.py and change a parameters, and take a train_test
```
$ train_test.py
```   