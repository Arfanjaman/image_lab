import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os


def get_gaus_kernel(sigma, size):
    """
    Generate a normalized Gaussian kernel.

    Parameters:
        sigma: Standard deviation of the Gaussian distribution
        size: Kernel size, such as 3, 5, or 7

    Returns:
        A size x size Gaussian kernel
    """

    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")

    if size <= 0 or size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number.")

    kernel = np.zeros((size, size), dtype=np.float64)

    center = size // 2
    total = 0.0

    for i in range(size):
        for j in range(size):

            x = i - center
            y = j - center

            exponent = -((x * x) + (y * y)) / (2 * sigma * sigma)

            kernel[i, j] = (
                1 / (2 * math.pi * sigma * sigma)
            ) * math.exp(exponent)

            total += kernel[i, j]

    # Normalize the kernel so that all values add up to 1
    for i in range(size):
        for j in range(size):
            kernel[i, j] = kernel[i, j] / total

    return kernel


def convolution(image, kernel):
    """
    Perform manual 2D convolution on a grayscale image.

    Parameters:
        image: Grayscale input image
        kernel: Convolution kernel

    Returns:
        Convolved output image
    """

    if image is None:
        raise ValueError("Input image is empty.")

    if len(image.shape) != 2:
        raise ValueError("This function currently supports grayscale images only.")

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Manually create a zero-padded image
    padded_height = image_height + 2 * pad_height
    padded_width = image_width + 2 * pad_width

    padded_image = np.zeros(
        (padded_height, padded_width),
        dtype=np.float64
    )

    # Copy the original image into the middle of the padded image
    for i in range(image_height):
        for j in range(image_width):
            padded_image[i + pad_height, j + pad_width] = image[i, j]

    output = np.empty_like(image, dtype=np.float64)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):

            pixel_sum = 0.0

            for m in range(kernel_height):
                for n in range(kernel_width):

                    # Do not flip the kernel
                    kernel_value = kernel[m, n]
                    # Flip the kernel for mathematical convolution
                    #kernel_value = kernel[
                    #    kernel_height - 1 - m,
                    #    kernel_width - 1 - n
                    #]

                    image_value = padded_image[i + m, j + n]

                    pixel_sum += image_value * kernel_value

            output[i, j] = pixel_sum

    # Ensure pixel values remain between 0 and 255
    output = np.clip(output, 0, 255)

    return output.astype(np.uint8)


# ---------------- Main program ----------------

input_path = r"Inputs\homework_a1_b1.png"
output_folder = r"Outputs"

os.makedirs(output_folder, exist_ok=True)

image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Could not read image: {input_path}")

sigma = 1.0
kernel_size = 5

gaussian_kernel = get_gaus_kernel(sigma, kernel_size)

blurred_image = convolution(image, gaussian_kernel)

print("Gaussian kernel:")
print(gaussian_kernel)

print("\nSum of kernel values:")
print(np.sum(gaussian_kernel))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(blurred_image, cmap="gray", vmin=0, vmax=255)
plt.title("Gaussian Blurred Image")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "gaussian_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

cv2.imwrite(
    os.path.join(output_folder, "gaussian_blurred.png"),
    blurred_image
)