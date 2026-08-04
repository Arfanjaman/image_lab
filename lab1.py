import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_libraries():
    print("Testing libraries...")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    img[25:75, 25:75, 1] = 255
    img[40:60, 40:60, 2] = 255

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.shape != (100, 100):
        raise RuntimeError("OpenCV image conversion failed")

    mean_val = np.mean(img)
    if not (0 <= mean_val <= 255):
        raise RuntimeError("NumPy image operation failed")

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Library Test")
    fig.savefig("library_test.png")
    plt.close(fig)

    if not os.path.exists("library_test.png"):
        raise RuntimeError("Matplotlib image save failed")

    print("All library tests passed.")
    print("Saved image: library_test.png")


if __name__ == "__main__":
    test_libraries()