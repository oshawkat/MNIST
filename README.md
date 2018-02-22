# Hand-Written Digit Recognition

This project looks at different neural network techniques for recognizing hand-written digits using the MNIST dataset.  Approaches include:

* Softmax classifier: ~91% accuracy
* Deep Convolutional Neural Net (CNN): ~99.2% accuracy

### Prerequisites

This module has been tested on Python 2.7 and requires Numpy and Tensorflow.  You can use PiP to install Numpy:

```
pip install numpy
```
or view alternative installation instructions on the [Numpy site](https://www.scipy.org/scipylib/download.html).

Tensorflow can be installed following instructions from the [Tensorflow site](https://www.tensorflow.org/install/).


### Use

Note that training can be computationaly intensive so program execution can take a long time depending on your system.

Each approach is independent and can be executed by running the individual python script eg:

```
python mnist_deepcnn.py
```

The first time you run one of these learners, please allow time for the MNIST dataset to be downloaded to your machine.  

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Tensorflow tutorials and guides - [Tensorflow](https://www.tensorflow.org/tutorials/)
* README template courtesy of Bille Thompson - [PurpleBooth](https://github.com/PurpleBooth)

