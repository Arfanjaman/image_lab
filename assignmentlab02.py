import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#function to calculate histogram for a single chnanel
def calculate_histogram(channel):
    if channel is None:
        raise ValueError("Input channel is empty.")

    if len(channel.shape) != 2:
        raise ValueError("The input must be a single 2D channel.")

    image_height, image_width = channel.shape

    histogram = np.zeros(
        256,
        dtype=np.int64
    )

    # Count every intensity value manually
    for i in range(image_height):
        for j in range(image_width):

            intensity = int(channel[i, j])
            histogram[intensity] += 1

    return histogram


def calculate_pdf(histogram, total_pixels):
 

    if total_pixels <= 0:
        raise ValueError("Total number of pixels must be positive.")

    pdf = np.zeros(
        256,
        dtype=np.float64
    )

    for intensity in range(256):
        pdf[intensity] = (
            histogram[intensity] / total_pixels
        )

    return pdf


def calculate_cdf(pdf):
   

    cdf = np.zeros(
        256,
        dtype=np.float64
    )

    cumulative_probability = 0.0

    for intensity in range(256):

        cumulative_probability += pdf[intensity]
        cdf[intensity] = cumulative_probability

    return cdf


def create_equalization_map(cdf):
    """
    Create the histogram-equalization lookup table manually.

    Formula:
        new_intensity =
            round(
                ((CDF - CDF_min) / (1 - CDF_min)) * 255
            )
    """

    equalization_map = np.zeros(
        256,
        dtype=np.uint8
    )

    # Find the first non-zero CDF value manually
    cdf_min = 0.0

    for intensity in range(256):
        if cdf[intensity] > 0:
            cdf_min = cdf[intensity]
            break

    denominator = 1.0 - cdf_min

    # A constant-valued image cannot be stretched normally
    if denominator <= 0:
        for intensity in range(256):
            equalization_map[intensity] = intensity

        return equalization_map

    for intensity in range(256):

        mapped_value = round(
            (
                (cdf[intensity] - cdf_min)
                / denominator
            ) * 255
        )

        # Keep the mapped value inside the valid 8-bit range
        if mapped_value < 0:
            mapped_value = 0

        elif mapped_value > 255:
            mapped_value = 255

        equalization_map[intensity] = mapped_value

    return equalization_map


def apply_equalization(channel, equalization_map):
    """
    Replace every pixel intensity manually using the lookup table.
    """

    image_height, image_width = channel.shape

    equalized_channel = np.zeros(
        (image_height, image_width),
        dtype=np.uint8
    )

    for i in range(image_height):
        for j in range(image_width):

            old_intensity = int(channel[i, j])

            equalized_channel[i, j] = (
                equalization_map[old_intensity]
            )

    return equalized_channel


def equalize_rgb_channels(image):

    if image is None:
        raise ValueError("Input image is empty.")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(
            "The input must be a three-channel color image."
        )

    image_height, image_width, channels = image.shape
    total_pixels = image_height * image_width

    equalized_image = np.zeros(
        (
            image_height,
            image_width,
            channels
        ),
        dtype=np.uint8
    )

    original_histograms = []
    original_pdfs = []
    original_cdfs = []

    equalized_histograms = []
    equalized_pdfs = []
    equalized_cdfs = []

    # Process Blue, Green and Red separately
    for channel_index in range(channels):

        # Manually copy one channel
        channel = np.zeros(
            (image_height, image_width),
            dtype=np.uint8
        )

        for i in range(image_height):
            for j in range(image_width):
                channel[i, j] = image[
                    i,
                    j,
                    channel_index
                ]

        # Original statistics
        histogram = calculate_histogram(channel)

        pdf = calculate_pdf(
            histogram,
            total_pixels
        )

        cdf = calculate_cdf(pdf)

        # Create mapping and equalize the channel
        equalization_map = create_equalization_map(cdf)

        equalized_channel = apply_equalization(
            channel,
            equalization_map
        )

        # Manually place the equalized channel into the output image
        for i in range(image_height):
            for j in range(image_width):

                equalized_image[
                    i,
                    j,
                    channel_index
                ] = equalized_channel[i, j]

        # Equalized statistics
        equalized_histogram = calculate_histogram(
            equalized_channel
        )

        equalized_pdf = calculate_pdf(
            equalized_histogram,
            total_pixels
        )

        equalized_cdf = calculate_cdf(
            equalized_pdf
        )

        original_histograms.append(histogram)
        original_pdfs.append(pdf)
        original_cdfs.append(cdf)

        equalized_histograms.append(
            equalized_histogram
        )

        equalized_pdfs.append(
            equalized_pdf
        )

        equalized_cdfs.append(
            equalized_cdf
        )

    return (
        equalized_image,
        original_histograms,
        original_pdfs,
        original_cdfs,
        equalized_histograms,
        equalized_pdfs,
        equalized_cdfs
    )


def create_result_figure(
    original_image,
    equalized_image,
    original_pdfs,
    original_cdfs,
    equalized_pdfs,
    equalized_cdfs,
    output_path
):

    intensity_values = np.arange(256)

    # Stored list order is B, G, R
    blue_index = 0
    green_index = 1
    red_index = 2

    # OpenCV uses BGR, while Matplotlib expects RGB
    display_original = original_image[:, :, ::-1]
    display_equalized = equalized_image[:, :, ::-1]

    plt.figure(
        figsize=(18, 10)
    )

    # =====================================================
    # Row 1, Column 1: Original RGB image
    # =====================================================

    plt.subplot(2, 3, 1)

    plt.imshow(display_original)
    plt.title("Original RGB Image")
    plt.axis("off")

    # =====================================================
    # Row 1, Column 2: Original RGB PDFs
    # =====================================================

    plt.subplot(2, 3, 2)

    plt.plot(
        intensity_values,
        original_pdfs[red_index],
        color="red",
        label="Red"
    )

    plt.plot(
        intensity_values,
        original_pdfs[green_index],
        color="green",
        label="Green"
    )

    plt.plot(
        intensity_values,
        original_pdfs[blue_index],
        color="blue",
        label="Blue"
    )

    plt.title("Original RGB PDFs")
    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.xlim(0, 255)
    plt.legend()
    plt.grid(alpha=0.2)

    # =====================================================
    # Row 1, Column 3: Original RGB CDFs
    # =====================================================

    plt.subplot(2, 3, 3)

    plt.plot(
        intensity_values,
        original_cdfs[red_index],
        color="red",
        label="Red"
    )

    plt.plot(
        intensity_values,
        original_cdfs[green_index],
        color="green",
        label="Green"
    )

    plt.plot(
        intensity_values,
        original_cdfs[blue_index],
        color="blue",
        label="Blue"
    )

    plt.title("Original RGB CDFs")
    plt.xlabel("Intensity")
    plt.ylabel("Cumulative Probability")
    plt.xlim(0, 255)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(alpha=0.2)

    # =====================================================
    # Row 2, Column 1: Equalized RGB image
    # =====================================================

    plt.subplot(2, 3, 4)

    plt.imshow(display_equalized)
    plt.title("Per-Channel Equalized Image")
    plt.axis("off")

    # =====================================================
    # Row 2, Column 2: Equalized RGB PDFs
    # =====================================================

    plt.subplot(2, 3, 5)

    plt.plot(
        intensity_values,
        equalized_pdfs[red_index],
        color="red",
        label="Red"
    )

    plt.plot(
        intensity_values,
        equalized_pdfs[green_index],
        color="green",
        label="Green"
    )

    plt.plot(
        intensity_values,
        equalized_pdfs[blue_index],
        color="blue",
        label="Blue"
    )

    plt.title("Equalized RGB PDFs")
    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.xlim(0, 255)
    plt.legend()
    plt.grid(alpha=0.2)

    # =====================================================
    # Row 2, Column 3: Equalized RGB CDFs
    # =====================================================

    plt.subplot(2, 3, 6)

    plt.plot(
        intensity_values,
        equalized_cdfs[red_index],
        color="red",
        label="Red"
    )

    plt.plot(
        intensity_values,
        equalized_cdfs[green_index],
        color="green",
        label="Green"
    )

    plt.plot(
        intensity_values,
        equalized_cdfs[blue_index],
        color="blue",
        label="Blue"
    )

    plt.title("Equalized RGB CDFs")
    plt.xlabel("Intensity")
    plt.ylabel("Cumulative Probability")
    plt.xlim(0, 255)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# =====================================================
# Main program
# =====================================================


input_path = r"Inputs\boat.jpg"
output_folder = r"Outputs"

os.makedirs(
    output_folder,
    exist_ok=True
)

image = cv2.imread(
    input_path,
    cv2.IMREAD_COLOR
)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {input_path}"
    )

print("Original image shape:")
print(image.shape)

(
    equalized_image,
    original_histograms,
    original_pdfs,
    original_cdfs,
    equalized_histograms,
    equalized_pdfs,
    equalized_cdfs
) = equalize_rgb_channels(image)

# Save the equalized RGB image
equalized_image_path = os.path.join(
    output_folder,
    "per_channel_equalized_rgb.png"
)

cv2.imwrite(
    equalized_image_path,
    equalized_image
)

# Save the complete 2 x 3 result figure
comparison_path = os.path.join(
    output_folder,
    "histogram_equalization_comparison.png"
)

create_result_figure(
    original_image=image,
    equalized_image=equalized_image,
    original_pdfs=original_pdfs,
    original_cdfs=original_cdfs,
    equalized_pdfs=equalized_pdfs,
    equalized_cdfs=equalized_cdfs,
    output_path=comparison_path
)

print("\nEqualized RGB image saved to:")
print(equalized_image_path)

print("\nComplete comparison figure saved to:")
print(comparison_path)
