import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import ImageFilter

# This function will demonstrate two side of the image

def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1)
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2)
    plt.title(title_2)
    plt.show()

def show_image(image):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.show()




image = Image.open("C:/Users/achit/OneDrive/Desktop/Pics/lenna.png")
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

# The images compromised of RGB values from 0 to 255.

# Getting the number of rows and columns of the image.
rows, cols = image.size

# Creating a normal distribution with a mean of 0 and a standard deviation of 1.
noise = np.random.normal(0,15,(rows, cols, 3)).astype(np.uint8)

# Add the noise to the image

noisy_image = image + noise

# Creates a PIl from image from an array

noisy_image = Image.fromarray(noisy_image)

# In a sense, the fromarray() is like a magic wand that understands your lego idea and turn into a real image.

# Plotting th eimage

plot_image(image,  noisy_image, title_1 = "Original Image", title_2 = "Noisy Image")


# Filtering Noise

from PIL import ImageFilter

# Creating a kernal which is a 5 by 5 where each value is 1/36

kernel = np.ones((5,5))/36

# Create a imageFilter kernel by providing the kernel size and flattened kernel

kernel_filter = ImageFilter.Kernel((5,5, kernel.flatten()))

# Filters the images using the kernel
image_filtered = noisy_image.filter(kernel_filter)

plot_image(image_filtered, noisy_image, title_1 = "Image Filtered", title_2 = "Noisy Image")

# Having a smaller kernels keeps the image sharp, but filter less noise

# Creating a 3x3 kernel

kernel = np.ones((3,3))/36

# Create a imageFilter kernel by providing the kernel size and flattened kernel
kernel_filter = ImageFilter.Kernel((3,3, kernel.flatten()))

# Filters the images using the kernel

image_filtered = noisy_image.filter(kernel_filter)

plot_image(image_filtered, noisy_image, title_1 = "Image Filtered", title_2 = "Noisy Image")

# Using Gaussian Blur

image_filtered = noisy_image.filter(ImageFilter.GaussianBlur)

# Using GaussianBlue with 4 kernel size

image_filtered = noisy_image.filter(ImageFilter.GaussianBlur(4))

plot_image(image_filtered, noisy_image, title_1 = "Gaussian Blur", title_2 = "Noisy Image")

# Sharpening the image
# Image sharpening requires smoothing the image and calculates the derivatives of the image.
kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])

kernel = ImageFilter.Kernel((3,3), kernel.flatten())

sharpened = image.filter(kernel)

plot_image(sharpened, noisy_image, title_1 = "Sharpened Image", title_2 = "Noisy Image")

# We can also sharpen the image using a predefined filter

sharpened = image.filter(ImageFilter.SHARPEN)

plot_image(sharpened, noisy_image, title_1 = "Sharpened Image", title_2 = "Noisy Image")

# Edge Detection

img_gray = Image.open("C:/Users/achit/OneDrive/Desktop/Pics/barbara.png")
plt.figure(figsize=(10,10))
plt.imshow(img_gray, cmap = "gray")
plt.show()


# Enhancing the image through the edge detection

img_gray = img_gray.filter(ImageFilter.EDGE_ENHANCE)
plt.figure(figsize=(10,10))
plt.imshow(img_gray, cmap = "gray")
plt.show()

# Filtering the image using FIND_EDGES

img_gray = img_gray.filter(ImageFilter.FIND_EDGES)
plt.figure(figsize=(10,10))
plt.imshow(img_gray, cmap = "gray")
plt.show()

# Median Filtering

image = Image.open("C:/Users/achit/OneDrive/Desktop/Pics/cameraman.jpeg")
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

# Median Filtering, filtering the background image, while increasing the segmentation between cameraman and the bacgrkound

image = image.filter(ImageFilter.MedianFilter)
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()



def median_filter(image, size):
    image = image.filter(ImageFilter.MedianFilter(size))
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.show()


median_filter_1 = median_filter(image, 1)
median_filter_3 = median_filter(image, 3)

