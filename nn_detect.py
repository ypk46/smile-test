# pylint: disable=no-member
import cv2
import click
import imutils
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array


@click.command()
@click.option(
    "-c",
    "--cascade",
    required=True,
    type=click.Path(exists=True),
    help="Path to where the face cascade resides",
)
@click.option(
    "-m",
    "--model",
    required=True,
    type=click.Path(exists=True),
    help="Path to the smile detector model.",
)
@click.option(
    "-v",
    "--video",
    help="Path to optional video file.",
)
def detect(cascade: str, model: str, video: str):
    """Detect smiles from video file or real time video feed."""
    smile_counter = 0

    # Load haar cascade detector and smile detection model
    detector = cv2.CascadeClassifier(cascade)
    model = load_model(model)

    # Check if video file was not supplied
    if not video:
        camera = cv2.VideoCapture(1)
    else:
        camera = cv2.VideoCapture(video)

    while True:
        # Grap current frame
        (grabbed, frame) = camera.read()

        if video and not grabbed:
            break

        # Resize frame, convert it to grayscale and clone original frame
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clone = frame.copy()

        # Detect faces
        detections = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Loop over the faces bounding boxes
        for (face_x, face_y, face_w, face_h) in detections:
            roi = gray[face_y : face_y + face_h, face_x : face_x + face_w]
            roi = cv2.resize(roi, (28, 28))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict if face is smiling or not
            (not_smiling, smiling) = model.predict(roi)[0]
            label = "Smiling" if smiling > not_smiling else "Not Smiling"

            if label == "Smiling":
                smile_counter += 1

            cv2.putText(
                clone,
                label,
                (face_x, face_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2,
            )
            cv2.rectangle(
                clone,
                (face_x, face_y),
                (face_x + face_w, face_y + face_h),
                (0, 0, 255),
                2,
            )

        cv2.imshow("Face", clone)

        # If the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    click.echo(f"We detected smiles in {smile_counter} frames.")


if __name__ == "__main__":
    detect()  # pylint: disable=no-value-for-parameter
