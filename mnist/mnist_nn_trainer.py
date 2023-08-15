#
# Uses pytorch to train a MLP, with one hidden layer (784-512-10)
# on the MNIST dataset.
#
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import mnist_dataset
import mnist_net

DEBUG_COUNT = 500
MAX_EPOCHS = 50
BATCH_SIZE = 32
VAL_DATASET_SIZE = 10000


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"cuda.is_available() = {torch.cuda.is_available()}")
    print(f"device = {device}")

    model = mnist_net.Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # print model
    print("model =")
    print(model)
    print("parameters =")
    for name, param in model.named_parameters():
        print(f"    {name}, size = {param.size()}")

    # load the training dataset
    full_train_dataset = mnist_dataset.MNIST_Dataset(
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
    )

    # split the dataset into two parts, one for training and one for validation
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [len(full_train_dataset) - VAL_DATASET_SIZE, VAL_DATASET_SIZE],
    )
    torch.manual_seed(torch.initial_seed())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    #
    # train
    #

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    max_epochs_without_improvement = 10
    train_start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        # train the model
        train_loss = 0.0
        train_count = 0

        for image, target in train_loader:
            # transfer tensors to gpu
            image, target = image.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, target)

            train_loss += loss.item()
            train_count += 1

            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / train_count

        # validate the model
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for image, target in val_loader:
                # transfer tensors to gpu
                image, target = image.to(device), target.to(device)

                output = model(image)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_count += 1

        avg_val_loss = val_loss / val_count
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch+1}: Training Loss = {avg_train_loss}, Validation Loss = {avg_val_loss}"
        )

        if epochs_without_improvement >= max_epochs_without_improvement:
            print("Early stopping triggered. Stopping training.")
            break

    train_end_time = time.time()
    print(f"Training took {train_end_time - train_start_time} sec")

    # load the test dataset
    testset = mnist_dataset.MNIST_Dataset(
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model.eval()
    with torch.no_grad():
        fail_count = 0
        test_count = 0
        for image, target in testloader:
            # transfer to gpu
            image = image.to(device)

            output = model(image)

            output_vecs = [x for x in output]
            target_vecs = [x for x in target]

            for o, t in zip(output_vecs, target_vecs):
                expected = t.argmax()
                actual = o.argmax()

                if expected != actual:
                    fail_count += 1
                test_count += 1

        print(f"Testing error rate = {100 * fail_count / test_count}%")

    #
    # dump parameters to a json file
    #

    params = []
    for name, param in model.named_parameters():
        params.append({"name": name, "param": param.view(-1).tolist()})

    with open("params.json", "w") as out:
        out.write(json.dumps(params, indent=4))

    #
    # output state_dict
    #

    torch.save(model.state_dict(), "model_state_dict.dat")

    #
    # dump parameters to a binary file
    #

    with open("params.bin", "wb") as out:
        with torch.no_grad():
            print("// structure of params.bin")
            print("struct {")
            for name, param in model.named_parameters():
                flat = param.cpu().flatten()
                print(f"    float {name.replace('.', '_')}[{len(flat)}];")
                out.write(flat.numpy().tobytes())
            print("} Params;")


if __name__ == "__main__":
    main()
