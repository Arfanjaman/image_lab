import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os


def manual_dft_2d(image):
    """Compute the 2D DFT manually using four nested loops."""
    if image is None or len(image.shape) != 2:
        raise ValueError("Input must be a grayscale image.")

    rows, cols = image.shape
    result = np.zeros((rows, cols), dtype=np.complex128)

    for u in range(rows):
        for v in range(cols):
            real_sum = 0.0
            imag_sum = 0.0

            for x in range(rows):
                for y in range(cols):
                    angle = -2.0 * math.pi * (
                        (u * x / rows) + (v * y / cols)
                    )
                    value = float(image[x, y])
                    real_sum += value * math.cos(angle)
                    imag_sum += value * math.sin(angle)

            result[u, v] = complex(real_sum, imag_sum)

    return result


def manual_idft_2d(frequency_image):
    """Compute the inverse 2D DFT manually using four nested loops."""
    rows, cols = frequency_image.shape
    result = np.zeros((rows, cols), dtype=np.float64)

    for x in range(rows):
        for y in range(cols):
            real_sum = 0.0

            for u in range(rows):
                for v in range(cols):
                    angle = 2.0 * math.pi * (
                        (u * x / rows) + (v * y / cols)
                    )
                    value = frequency_image[u, v]
                    real_sum += (
                        value.real * math.cos(angle)
                        - value.imag * math.sin(angle)
                    )

            result[x, y] = real_sum / (rows * cols)

    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


def manual_fft_shift(frequency_image):
    """Move low frequencies to the center manually."""
    rows, cols = frequency_image.shape
    shifted = np.zeros((rows, cols), dtype=np.complex128)
    row_shift = rows // 2
    col_shift = cols // 2

    for i in range(rows):
        for j in range(cols):
            new_i = (i + row_shift) % rows
            new_j = (j + col_shift) % cols
            shifted[new_i, new_j] = frequency_image[i, j]

    return shifted


def manual_ifft_shift(shifted_image):
    """Undo the frequency shift manually."""
    rows, cols = shifted_image.shape
    unshifted = np.zeros((rows, cols), dtype=np.complex128)
    row_shift = (rows + 1) // 2
    col_shift = (cols + 1) // 2

    for i in range(rows):
        for j in range(cols):
            new_i = (i + row_shift) % rows
            new_j = (j + col_shift) % cols
            unshifted[new_i, new_j] = shifted_image[i, j]

    return unshifted


def magnitude_spectrum(frequency_image):
    """Calculate a displayable log-magnitude spectrum manually."""
    rows, cols = frequency_image.shape
    magnitude = np.zeros((rows, cols), dtype=np.float64)
    max_value = 0.0

    for i in range(rows):
        for j in range(cols):
            real = frequency_image[i, j].real
            imag = frequency_image[i, j].imag
            value = math.log(1.0 + math.sqrt(real * real + imag * imag))
            magnitude[i, j] = value
            if value > max_value:
                max_value = value

    if max_value > 0:
        for i in range(rows):
            for j in range(cols):
                magnitude[i, j] = magnitude[i, j] / max_value * 255.0

    return magnitude.astype(np.uint8)


def create_low_pass_mask(rows, cols, radius):
    """Create an ideal circular low-pass filter with loops."""
    mask = np.zeros((rows, cols), dtype=np.float64)
    center_row = rows // 2
    center_col = cols // 2

    for i in range(rows):
        for j in range(cols):
            distance = math.sqrt(
                (i - center_row) ** 2 + (j - center_col) ** 2
            )
            if distance <= radius:
                mask[i, j] = 1.0

    return mask


def create_high_pass_mask(rows, cols, radius):
    """Create an ideal circular high-pass filter with loops."""
    mask = np.ones((rows, cols), dtype=np.float64)
    center_row = rows // 2
    center_col = cols // 2

    for i in range(rows):
        for j in range(cols):
            distance = math.sqrt(
                (i - center_row) ** 2 + (j - center_col) ** 2
            )
            if distance <= radius:
                mask[i, j] = 0.0

    return mask


def apply_mask(frequency_image, mask):
    """Multiply a complex spectrum by a real mask manually."""
    rows, cols = frequency_image.shape
    result = np.zeros((rows, cols), dtype=np.complex128)

    for i in range(rows):
        for j in range(cols):
            result[i, j] = frequency_image[i, j] * mask[i, j]

    return result


def process_frequency_domain(image, radius):
    dft_result = manual_dft_2d(image)
    shifted_dft = manual_fft_shift(dft_result)

    rows, cols = image.shape
    low_mask = create_low_pass_mask(rows, cols, radius)
    high_mask = create_high_pass_mask(rows, cols, radius)

    low_frequency = apply_mask(shifted_dft, low_mask)
    high_frequency = apply_mask(shifted_dft, high_mask)

    low_image = manual_idft_2d(manual_ifft_shift(low_frequency))
    high_image = manual_idft_2d(manual_ifft_shift(high_frequency))

    return (
        magnitude_spectrum(shifted_dft),
        low_mask,
        high_mask,
        magnitude_spectrum(low_frequency),
        magnitude_spectrum(high_frequency),
        low_image,
        high_image,
    )


# =====================================================
# Main program
# =====================================================

input_path = r"Inputs\input_image.png"
output_folder = r"Outputs"
os.makedirs(output_folder, exist_ok=True)

image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Could not read image: {input_path}")

# Manual DFT has O(M^2N^2) complexity, so use a small image.
small_size = 32
image = cv2.resize(
    image,
    (small_size, small_size),
    interpolation=cv2.INTER_AREA,
)

radius = 5
(
    original_spectrum,
    low_mask,
    high_mask,
    low_spectrum,
    high_spectrum,
    low_image,
    high_image,
) = process_frequency_domain(image, radius)

cv2.imwrite(os.path.join(output_folder, "low_pass_filtered.png"), low_image)
cv2.imwrite(os.path.join(output_folder, "high_pass_filtered.png"), high_image)

plt.figure(figsize=(16, 10))

items = [
    (image, "Original Image"),
    (original_spectrum, "Magnitude Spectrum"),
    (low_mask, "Low-Pass Mask"),
    (low_spectrum, "Low-Pass Spectrum"),
    (low_image, "Low-Pass Output"),
    (high_mask, "High-Pass Mask"),
    (high_spectrum, "High-Pass Spectrum"),
    (high_image, "High-Pass Output"),
]

for index in range(len(items)):
    plt.subplot(2, 4, index + 1)
    plt.imshow(items[index][0], cmap="gray")
    plt.title(items[index][1])
    plt.axis("off")

plt.tight_layout()
comparison_path = os.path.join(output_folder, "frequency_domain_comparison.png")
plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
plt.show()

print("Low-pass output saved to:", os.path.join(output_folder, "low_pass_filtered.png"))
print("High-pass output saved to:", os.path.join(output_folder, "high_pass_filtered.png"))
print("Comparison figure saved to:", comparison_path)
