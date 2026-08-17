import tensorflow_datasets as tfds
import numpy as np

def to_numpy(dataset):
    images = []
    labels = []

    for image, label in dataset:
        images.append(image.numpy())
        labels.append(label.numpy())

    return np.array(images), np.array(labels)

train_dataset, test_dataset = tfds.load("mnist", split = ["train", "test"], as_supervised = True)

train_images, train_labels = to_numpy(train_dataset)
test_images, test_labels = to_numpy(test_dataset)

train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_labels = train_labels.astype(np.float32)
test_labels = test_labels.astype(np.float32)

train_images.tofile("train_images.bin")
train_labels.tofile("train_labels.bin")
test_images.tofile("test_images.bin")
test_labels.tofile("test_labels.bin")

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
