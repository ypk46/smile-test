# pylint: disable=no-member
import os
import cv2
import click
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths, resize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array
from model.lenet import LeNet


@click.command()
@click.option(
    "-d",
    "--dataset",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to input dataset of faces.",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(exists=False),
    help="Path to output model.",
)
def train(dataset: str, output: str):
    """
    Train a smile detection model.
    """
    images = []
    labels = []

    for image_path in sorted(list(paths.list_images(dataset))):
        # Load image, pre-process it and store it in data list
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = resize(img, width=28)
        img = img_to_array(img)
        images.append(img)

        # Extract class label from path
        label = image_path.split(os.path.sep)[-2]
        labels.append(label)

    # Scale raw pixel intensities to range of 0 and 1
    images = np.array(images, dtype="float") / 255.0

    # Convert labels from integers to vectors
    labels = np.array(labels)
    encoder = LabelEncoder().fit(labels)
    labels = to_categorical(encoder.transform(labels), 2)

    # Handling class imbalance by updating the class weights
    class_totals = labels.sum(axis=0)
    class_weight = {}

    for i, _ in enumerate(class_totals):
        class_weight[i] = class_totals.max() / class_totals[i]

    # Split test/train dataset
    (train_x, test_x, train_y, test_y) = train_test_split(
        images,
        labels,
        test_size=0.20,
        stratify=labels,
        random_state=42,
    )

    # Train the model
    model = LeNet.build(28, 28, 1, 2)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    result = model.fit(
        train_x,
        train_y,
        validation_data=(test_x, test_y),
        class_weight=class_weight,
        batch_size=64,
        epochs=15,
        verbose=1,
    )

    # Evaluate trained model
    predictions = model.predict(test_x, batch_size=64)
    print(
        classification_report(
            test_y.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=encoder.classes_,
        )
    )

    # Save trained model
    model.save(output)

    # plot the training + testing loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 15), result.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 15), result.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 15), result.history["accuracy"], label="acc")
    plt.plot(np.arange(0, 15), result.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train()  # pylint: disable=no-value-for-parameter
