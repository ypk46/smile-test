from keras import backend
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense


class LeNet:
    """LeNet implementation using Keras."""

    @staticmethod
    def build(width: int, height: int, depth: int, classes: int):
        """Build LeNet convolutional neural network."""
        # Initialize the model
        model = Sequential()
        input_shape = (height, width, depth)

        # If using channel first, update the shape
        if backend.image_data_format() == "channel_first":
            input_shape = (depth, height, width)

        # Build the LeNet CNN
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
