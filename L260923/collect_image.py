import cv2
import os

DEV = "/dev/video1"


def video(capture: cv2.VideoCapture):
    index = 0
    codec = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    capture.set(cv2.CAP_PROP_FOURCC, codec)
    capture.set(cv2.CAP_PROP_FPS, 60)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WINDOW_NAME = "Frame"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)

    file_dir = os.path.dirname(os.path.abspath(__file__))

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame = cv2.flip(frame, 0)

        if cv2.waitKey(1) & 0xFF == 32:  # Space
            save_dir = os.path.join(file_dir, "images")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_path = os.path.join(save_dir, f"image_{index:04d}.jpg")

            cv2.imwrite(file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"Saved image to {file_path}")
            index += 1

        cv2.putText(
            frame,
            f"#{index:04d}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "You can press [Space] to collect images",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == 27:  # Esc
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture = cv2.VideoCapture(DEV)
    video(capture)
