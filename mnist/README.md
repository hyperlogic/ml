Train and test a MLP on the MNIST image database.
It includes the following:
* raw MNIST dataset as compressed binary files.
* **mnist_dataset.py** - contains a custom pytorch Dataset for loading the raw compressed MNIST dataset and turning
  it into torch.Tensor instances.  This was done as an exercise, I could have used torchvision.datasets.MNIST instead.
* **mnist_net.py** - contains the neural network module, a 3 layer MLP with 789-512-10 nodes.
* **mnist_nn_trainer.py** - use the raw MNIST dataset to train the mnist_net model.
  It outputs the error rate achieved. It also outputs the weights & biases in various formats.
  * **params.json** - model weights and biases in json format
  * **params.bin** - binary file, as well as the c struct that can be used to read in this binary file.
  * **model_state_dict.dat** - torch.save() the model.state_dict that can be loaded back into a model later.  See mnist_
* **mnist_nn_inference.py** - uses torch.load() to load the state_dict saved from minst_nn_trainer.  It manually
  evaluates a single input.  This was used to help debug issues with mnist_eval.c
* **mnist_eval.c** - c program that will load in the params.bin created in mnist_nn_trainer.py and use it to evaluate
  the model on a single input.