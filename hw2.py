import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os


def get_gaus_kernel(sigma, size):
    """
    Generate a normalized 2D Gaussian kernel.

    Parameters:
        sigma: Standard deviation for both x and y axes
        size: Kernel size, such as 3, 5, or 7

    Returns:
        A size x size Gaussian kernel
    """

    if sigma <= 0:
        raise ValueError("Sigma must be greater than 0.")

    if size <= 0 or size % 2 == 0:
        raise ValueError(
            "Kernel size must be a positive odd number."
        )

    kernel = np.zeros(
        (size, size),
        dtype=np.float64
    )

    center = size // 2
    total = 0.0

    for i in range(size):
        for j in range(size):

            x = i - center
            y = j - center

            exponent = -(
                (x * x) + (y * y)
            ) / (2 * sigma * sigma)

            kernel[i, j] = (
                1 / (2 * math.pi * sigma * sigma)
            ) * math.exp(exponent)

            total += kernel[i, j]

    # Normalize so that kernel sum becomes 1
    for i in range(size):
        for j in range(size):
            kernel[i, j] /= total

    return kernel


def convolution(image, kernel):
    """
    Perform manual convolution on a grayscale or color image.

    The kernel is not flipped, according to the assignment.
    """

    if image is None:
        raise ValueError("Input image is empty.")

    kernel_height, kernel_width = kernel.shape

    if kernel_height % 2 == 0 or kernel_width % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Determine whether the image is grayscale or color
    if len(image.shape) == 2:
        image_height, image_width = image.shape
        channels = 1

    elif len(image.shape) == 3:
        image_height, image_width, channels = image.shape

    else:
        raise ValueError("Unsupported image format.")

    padded_height = image_height + 2 * pad_height
    padded_width = image_width + 2 * pad_width

    # -------------------------------------------------
    # Grayscale image
    # -------------------------------------------------

    if channels == 1:

        padded_image = np.zeros(
            (padded_height, padded_width),
            dtype=np.float64
        )

        # Manually copy original image into padded image
        for i in range(image_height):
            for j in range(image_width):
                padded_image[
                    i + pad_height,
                    j + pad_width
                ] = image[i, j]

        output = np.zeros(
            (image_height, image_width),
            dtype=np.float64
        )

        for i in range(image_height):
            for j in range(image_width):

                pixel_sum = 0.0

                for m in range(kernel_height):
                    for n in range(kernel_width):

                        image_value = padded_image[
                            i + m,
                            j + n
                        ]

                        # Do not flip the kernel
                        kernel_value = kernel[m, n]
                        #IF We have to  Flip the kernel for mathematical convolution
                        #kernel_value = kernel[
                        #    kernel_height - 1 - m,
                        #    kernel_width - 1 - n
                        #]
                        pixel_sum += (
                            image_value * kernel_value
                        )

                output[i, j] = pixel_sum

    # -------------------------------------------------
    # Color image
    # -------------------------------------------------

    else:

        padded_image = np.zeros(
            (
                padded_height,
                padded_width,
                channels
            ),
            dtype=np.float64
        )

        # Manually copy every channel
        for i in range(image_height):
            for j in range(image_width):
                for c in range(channels):

                    padded_image[
                        i + pad_height,
                        j + pad_width,
                        c
                    ] = image[i, j, c]

        output = np.zeros(
            (
                image_height,
                image_width,
                channels
            ),
            dtype=np.float64
        )

        for i in range(image_height):
            for j in range(image_width):

                # Separate sum for B, G and R
                pixel_sum = np.zeros(
                    channels,
                    dtype=np.float64
                )

                for m in range(kernel_height):
                    for n in range(kernel_width):

                        kernel_value = kernel[m, n]

                        for c in range(channels):

                            image_value = padded_image[
                                i + m,
                                j + n,
                                c
                            ]

                            pixel_sum[c] += (
                                image_value * kernel_value
                            )

                for c in range(channels):
                    output[i, j, c] = pixel_sum[c]

    output = np.clip(output, 0, 255)

    return output.astype(np.uint8)


# =====================================================
# Main program
# =====================================================


input_path = r"Inputs\homework_a1_b1.png"
output_folder = r"Outputs"

os.makedirs(
    output_folder,
    exist_ok=True
)

sigma = 3.0
kernel_size = 7

gaussian_kernel = get_gaus_kernel(
    sigma,
    kernel_size
)

print("Gaussian kernel:")
print(gaussian_kernel)

print("\nSum of kernel values:")
print(np.sum(gaussian_kernel))


def process_image(input_path, read_mode, image_type):
    """
    Read, convolve, display and save an image.

    Parameters:
        input_path: Path of the input image

        read_mode:
            cv2.IMREAD_GRAYSCALE or cv2.IMREAD_COLOR

        image_type:
            A name used for plot titles and output filenames
    """

    image = cv2.imread(
        input_path,
        read_mode
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {input_path}"
        )

    blurred_image = convolution(
        image,
        gaussian_kernel
    )

    print(f"\n{image_type} image shape:")
    print(image.shape)

    # --------------------------------------------------------
    # Prepare the images for Matplotlib
    # --------------------------------------------------------

    if len(image.shape) == 2:
        # Grayscale image has shape:
        # height × width

        display_original = image
        display_blurred = blurred_image
        display_cmap = "gray"

    else:
        # Color image has shape:
        # height × width × 3

        # OpenCV uses BGR but Matplotlib expects RGB
        display_original = image[:, :, ::-1]
        display_blurred = blurred_image[:, :, ::-1]
        display_cmap = None

    # --------------------------------------------------------
    # Original, kernel and convolved image in one plot
    # --------------------------------------------------------

    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)

    if display_cmap is None:
        plt.imshow(display_original)
    else:
        plt.imshow(
            display_original,
            cmap=display_cmap,
            vmin=0,
            vmax=255
        )

    plt.title(f"Original {image_type} Image")
    plt.axis("off")

    # Gaussian kernel
    plt.subplot(1, 3, 2)

    plt.imshow(
        gaussian_kernel,
        cmap="gray",
        interpolation="nearest"
    )

    plt.title("Gaussian Kernel")

    # Display individual kernel values
    for i in range(kernel_size):
        for j in range(kernel_size):
            plt.text(
                j,
                i,
                f"{gaussian_kernel[i, j]:.3f}",
                ha="center",
                va="center"
            )

    plt.xticks(range(kernel_size))
    plt.yticks(range(kernel_size))

    # Convolved image
    plt.subplot(1, 3, 3)

    if display_cmap is None:
        plt.imshow(display_blurred)
    else:
        plt.imshow(
            display_blurred,
            cmap=display_cmap,
            vmin=0,
            vmax=255
        )

    plt.title(f"Convolved {image_type} Image")
    plt.axis("off")

    plt.tight_layout()

    # --------------------------------------------------------
    # Save the complete comparison plot
    # --------------------------------------------------------

    comparison_path = os.path.join(
        output_folder,
        f"gaussian_comparison_{image_type.lower()}.png"
    )

    plt.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # --------------------------------------------------------
    # Save the convolved image
    # --------------------------------------------------------

    blurred_path = os.path.join(
        output_folder,
        f"gaussian_blurred_{image_type.lower()}.png"
    )

    cv2.imwrite(
        blurred_path,
        blurred_image
    )

    print(f"{image_type} comparison saved to:")
    print(comparison_path)

    print(f"{image_type} blurred image saved to:")
    print(blurred_path)


# ============================================================
# Process the image as grayscale
# ============================================================

process_image(
    input_path=input_path,
    read_mode=cv2.IMREAD_GRAYSCALE,
    image_type="Grayscale"
)


# ============================================================
# Process the image as color
# ============================================================

process_image(
    input_path=input_path,
    read_mode=cv2.IMREAD_COLOR,
    image_type="Color"
)