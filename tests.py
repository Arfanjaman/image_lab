import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"chest.png", cv2.IMREAD_GRAYSCALE)

# output = 255 - image

height = image.shape[0]
width = image.shape[1]

output = np.empty_like(image)
#negative transformation
for i in range(height):
    for j in range(width):
        output[i, j] = 255 - image[i, j]

plt.figure(figsize=(4, 3))

plt.subplot(1, 4, 1)
plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title("Negative")
plt.axis("off")

#gamma transformation
for i in range(height):
    for j in range(width):
        r = image[i, j] / 255
        s = 1 * r ** 2
        output[i, j] = (255 * s).astype(np.uint8)


plt.subplot(1, 4, 3)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title("Gamma")
plt.axis("off")

for i in range(height):
    for j in range(width):
        r = image[i, j].astype(np.float32)
        s = 30 * np.log(1 + r)
        output[i, j] = s.astype(np.uint8)

plt.subplot(1, 4, 4)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title("Logarithmic")
plt.axis("off")



plt.tight_layout()

plt.show()

