# pylint: disable=no-member
import cv2
import dlib
import click
from imutils import resize

RATIO_THRESHOLD = 0.30


@click.command()
@click.option(
    "-p",
    "--predictor",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to landmark predictor file.",
)
@click.option(
    "-f",
    "--file",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to video file to load.",
)
def detect(predictor: str, file: str):
    """
    Detect smiles from video feed using landmarks calculations.
    """
    # Initialize face detector and facial landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor)

    # Check if video file was not supplied
    if not file:
        camera = cv2.VideoCapture(0)
    else:
        camera = cv2.VideoCapture(file)

    while True:
        # Grap current frame
        (grabbed, frame) = camera.read()

        if not grabbed:
            break

        frame = resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray, 0)

        # Loop over the face detections
        for face in faces:
            # Get facial landmarks
            shape = predictor(gray, face)

            # Calculate libs/jaw ratio
            lips_width = abs(shape.parts()[49].x - shape.parts()[55].x)
            jaw_width = abs(shape.parts()[3].x - shape.parts()[15].x)
            ratio = lips_width / jaw_width

            if ratio > RATIO_THRESHOLD:
                result = "Smiling"
            else:
                result = "No Smiling"

            # Draw result text
            cv2.putText(
                frame,
                result,
                (50, 50),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                str(ratio),
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            # Draw landmarks
            # points = [(p.x, p.y) for p in shape.parts()]
            # for point in points:
            #     cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)

        cv2.imshow("Face", frame)

        # If the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect()  # pylint: disable=no-value-for-parameter
