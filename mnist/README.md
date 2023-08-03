Train and test a MLP on the MNIST image database.
It includes the following:
* raw MNIST dataset as compressed binary files.
* mnist_dataset.py - contains a custom pytorch Dataset for loading the raw compressed MNIST dataset and turning
  it into torch.Tensor instances.  This was done as an excersize, I could have used torchvision.datasets.MNIST instead.
* mnist_nn_trainer.py - use the raw MNIST dataset to train a MLP, with a single hidden layer (784-512-10).
  It outputs the error rate achived and the weights as a json file.