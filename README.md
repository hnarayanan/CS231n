# Understanding Convolutional Neural Networks

This repository is an archive of the course [CS231n: Convolutional
Neural Networks for Visual Recognition](http://cs231n.stanford.edu)
(Winter 2016). If you’re even vaguely interested in this topic, you
should probably take this class. It is outstanding.

To use this repository, [make a fork of
it](https://help.github.com/articles/fork-a-repo/) and then
tick off the items in the following syllabus as you complete
them. (You can tick off items by replacing `[ ]` with `[x]` in
`README.md`.)

Happy learning!

[Harish Narayanan](https://harishnarayanan.org/), 2017

## Course Syllabus

- [ ] **Lecture 1:** Intro to computer vision, historical context
   - [ ] [Video](https://youtu.be/NfnWJUyUJYU)
   - [ ] [Slides](slides/lecture1.pdf)
- [ ] **Lecture 2:** Image classification and the data-driven
      approach; k-nearest neighbors; Linear classification I
   - [ ] [Video](https://youtu.be/8inugqHkfvE)
   - [ ] [Slides](slides/lecture2.pdf)
   - [ ] [Python/NumPy tutorial](notes/python-numpy-tutorial.pdf)
   - [ ] [Image classification notes](notes/image-classification.pdf)
   - [ ] [Linear classification notes](notes/linear-classification.pdf)
- [ ] **Lecture 3:** Linear classification II; Higher-level
      representations, image features; Optimization, stochastic
      gradient descent
   - [ ] [Video](https://youtu.be/qlLChbHhbg4)
   - [ ] [Slides](slides/lecture3.pdf)
   - [ ] [Linear classification notes](notes/linear-classification.pdf)
   - [ ] [Optimization notes](notes/optimization.pdf)
- [ ] **Lecture 4:** Backpropagation; Introduction to neural networks
   - [ ] [Video](https://youtu.be/i94OvYb6noo)
   - [ ] [Slides](slides/lecture4.pdf)
   - [ ] [Backprop notes](notes/backprop.pdf)
   - [ ] Related references
      - [ ] [Efficient Backprop](papers/efficient-backprop.pdf) -- 3/44
      - [ ] [Automatic differentiation survey](papers/automatic-differentiation.pdf)
      - [ ] [Calculus on Computational Graphs](papers/backprop-calculus.pdf)
      - [ ] [Backpropagation Algorithm](papers/backprop-algorithm.pdf)
      - [ ] [Learning: Neural Nets, Back Propagation](https://youtu.be/q0pm3BrIUFo)
- [ ] **[Assignment 1](assignments/assignment1/assignment1.pdf)**
   - [ ] k-Nearest Neighbor classifier
   - [ ] Training a Support Vector Machine
   - [ ] Implement a Softmax classifier
   - [ ] Two-Layer Neural Network
   - [ ] Higher Level Representations: Image Features
   - [ ] Cool Bonus: Do something extra!
- [ ] **Lecture 5:** Training Neural Networks Part 1; Activation
      functions, weight initialization, gradient flow, batch
      normalization; Babysitting the learning process, hyperparameter
      optimization
   - [ ] [Video](https://youtu.be/gYpoJMlgyXA)
   - [ ] [Slides](slides/lecture5.pdf)
   - [ ] [Neural Nets notes 1](notes/neural-nets-1.pdf)
   - [ ] [Neural Nets notes 2](notes/neural-nets-2.pdf)
   - [ ] [Neural Nets notes 3](notes/neural-nets-3.pdf)
   - [ ] Related references
      - [ ] [Tips/Tricks 1](papers/sgd-tricks.pdf)
      - [ ] [Tips/Tricks 2](papers/efficient-backprop.pdf)
      - [ ] [Tips/Tricks 3](papers/practical-sgd.pdf)
      - [ ] [Deep learning review article](papers/deep-review.pdf)
- [ ] **Lecture 6:** Training Neural Networks Part 2: parameter
      updates, ensembles, dropout; Convolutional Neural Networks:
      intro
   - [ ] [Video](https://youtu.be/hd_KFJ5ktUc)
   - [ ] [Slides](slides/lecture6.pdf)
   - [ ] [Neural Nets notes 3](notes/neural-nets-3.pdf)
- [ ] **Lecture 7:** Convolutional Neural Networks: architectures,
      convolution / pooling layers; Case study of ImageNet challenge
      winning ConvNets
   - [ ] [Video](https://youtu.be/LxfUGhug-iQ)
   - [ ] [Slides](slides/lecture7.pdf)
   - [ ] [ConvNet notes](notes/conv-nets.pdf)
- [ ] **Lecture 8:** ConvNets for spatial localization; Object
      detection
   - [ ] [Video](https://youtu.be/GxZrEKZfW2o)
   - [ ] [Slides](slides/lecture8.pdf)
- [ ] **Lecture 9:** Understanding and visualizing Convolutional
      Neural Networks; Backprop into image: Visualizations, deep
      dream, artistic style transfer; Adversarial fooling examples
   - [ ] [Video](https://youtu.be/ta5fdaqDT3M)
   - [ ] [Slides](slides/lecture9.pdf)
- [ ] **[Assignment 2](assignments/assignment2/assignment2.pdf)**
   - [ ] Fully-connected Neural Network
   - [ ] Batch Normalization
   - [ ] Dropout
   - [ ] ConvNet on CIFAR-10
   - [ ] Do something extra!
- [ ] **Lecture 10:** Recurrent Neural Networks (RNN), Long Short Term
       Memory (LSTM); RNN language models; Image captioning
   - [ ] [Video](https://youtu.be/yCC09vCHzF8)
   - [ ] [Slides](slides/lecture10.pdf)
   - [ ] Related references
      - [ ] [Recurrent neural networks](papers/rnn.html)
      - [ ] [Min Char RNN](https://gist.github.com/karpathy/d4dee566867f8291f086)
      - [ ] [Char RNN](https://github.com/karpathy/char-rnn)
      - [ ] [NeuralTalk2](https://github.com/karpathy/neuraltalk2)
- [ ] **Lecture 11:** Training ConvNets in practice; Data
      augmentation, transfer learning; Distributed training, CPU/GPU
      bottlenecks; Efficient convolutions
   - [ ] [Video](https://youtu.be/pA4BsUK3oP4)
   - [ ] [Slides](slides/lecture11.pdf)
- [ ] **Lecture 12:** Overview of Caffe/Torch/Theano/TensorFlow
   - [ ] [Video](https://youtu.be/Vf_-OkqbwPo)
   - [ ] [Slides](slides/lecture12.pdf)
- [ ] **[Assignment 3](assignments/assignment3/assignment3.pdf)**
   - [ ] Image Captioning with Vanilla RNNs
   - [ ] Image Captioning with LSTMs
   - [ ] Image Gradients: Saliency maps and Fooling Images
   - [ ] Image Generation: Classes, Inversion, DeepDream
   - [ ] Do something extra!
- [ ] **Lecture 13:** Segmentation; Soft attention models; Spatial
      transformer networks
   - [ ] [Video](https://youtu.be/ByjaPdWXKJ4)
   - [ ] [Slides](slides/lecture13.pdf)
- [ ] **Lecture 14:** ConvNets for videos; Unsupervised learning
   - [ ] [Video](https://youtu.be/ekyBklxwQMU)
   - [ ] [Slides](slides/lecture14.pdf)
- [ ] **Invited Lecture:** A sampling of deep learning at Google
   - [ ] [Video](https://youtu.be/T7YkPWpwFD4)
- [ ] **Lecture 15:** Conclusions
   - [ ] [Slides](slides/lecture15.pdf)
