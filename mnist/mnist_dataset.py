import gzip
import sys
import struct
import torch
import torch.nn.functional as F


# load .gz file and inflate it
def load_gz_file(filename):
    with gzip.open(filename, "rb") as f:
        return f.read()


def load_minst_images_file(filename):
    raw_image_data = load_gz_file(filename)
    magic, num_images, rows, cols = struct.unpack(">IIII", raw_image_data[:16])
    if magic != 2051:
        raise ValueError(
            f"Magic number mismatch for {filename}, expected 2051, got {magic}"
        )

    # segment data into images
    IMAGE_DATA_OFFSET = 16
    images = []
    for i in range(num_images):
        start = IMAGE_DATA_OFFSET + i * rows * cols
        stop = start + rows * cols
        slice = raw_image_data[start:stop]
        images.append([float(byte) for byte in slice])

    return images


def load_minst_labels_file(filename):
    raw_label_data = load_gz_file(filename)
    magic, num_labels = struct.unpack(">II", raw_label_data[:8])
    if magic != 2049:
        raise ValueError(
            f"Magic number mismatch for {filename}, expected 2051, got {magic}"
        )

    # segment labels
    LABEL_DATA_OFFSET = 8
    labels = []
    for i in range(num_labels):
        start = LABEL_DATA_OFFSET + i
        stop = start + 1
        byte = struct.unpack(">B", (raw_label_data[start:stop]))[0]
        labels.append(int(byte))

    return labels


def one_hot(klass, num_classes=10):
    return [1.0 if klass == i else 0.0 for i in range(num_classes)]


# DataSet
class MNIST_Dataset(torch.utils.data.Dataset):
    def __init__(self, images_filename, labels_filename):
        self.images = load_minst_images_file(images_filename)
        self.labels = load_minst_labels_file(labels_filename)

        if len(self.images) != len(self.labels):
            raise ValueError(
                f"Images and Label count mismatch num_images = {len(self.images)}, num_labels = {len(self.labels)}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.Tensor(self.images[idx]), torch.Tensor(one_hot(self.labels[idx]))
