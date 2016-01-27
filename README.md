# deep_calibration
Experiments to test the performance of calibration techniques on Artificial
Neural Networks

### Dependencies

This code needs the next dependencies

* [Keras] - Deep Learning library for Theano and TensorFlow
    * [Theano] - A CPU and GPU Math Expression Compiler
    * [TensorFlow] - Software library for numerical computation using data flow
      graphs

### Download and run the code

To run the code just paste this few lines in a terminal

```sh
git clone https://github.com/perellonieto/deep_calibration.git
cd deep_calibration/scripts/
./get_mnist.sh
python create_dataset.py
cd ..
python train_lenet.py
```

### Todos

 - Add comments

## License

MIT

[//]: # (References)
   [Keras]: <https://github.com/fchollet/keras>
   [Theano]: <https://github.com/Theano/Theano>
   [TensorFlow]: <https://github.com/tensorflow/tensorflow>
