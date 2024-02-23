import matplotlib.pyplot as plt
import numpy as np
import cv2



def plot_image(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB))
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
    plt.title(title_2)
    plt.show()

def plt_plot(image):
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


image = cv2.imread("C:/Users/achit/OneDrive/Desktop/Pics/lenna.png")
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()


# Get the number of rows and columns of the image.

rows, cols = image.shape

# Creating a normal distribution with a mean of 0 and a standard deviation of 1.
noise = np.random.normal(0,15,(rows, cols, 3)).astype(np.uint8)

noisy_image = image + noise

plot_image(image,  noisy_image, title_1 = "Original Image", title_2 = "Noisy Image")

# Filtering Noise

kernel = np.ones((6,6))/36
image_filtered = cv2.filter2D(src = noisy_image, ddepth = -1, kernel =kernel)
# or
image_filtered = cv2.filter2D(noisy_image, -1, kernel)

plot_image(image_filtered, noisy_image, title_1 = "Image Filtered", title_2 = "Noisy Image")

# Creating a smaller kernel with 4,4

kernel = np.ones((4,4))/36
image_filtered = cv2.filter2D(src = noisy_image, ddepth = -1, kernel =kernel)
plot_image(image_filtered, noisy_image, title_1 = "Image Filtered", title_2 = "Noisy Image")


# Using Gaussian Blur
# Using noise with 5x5 kernel size
image_filtered = cv2.GaussianBlur(noisy_image, (5,5), sigmaX = 4, sigmaY = 4)
image_filtered = cv2.GaussianBlur(noisy_image, (5,5), 5, 5)
plot_image(image_filtered, noisy_image, title_1 = "Gaussian Blur", title_2 = "Noisy Image")

# Using on image with noise using 11 x 11 kernel size

image_filtered = cv2.GaussianBlur(noisy_image, (11,11), sigmaX = 10, sigmaY = 10)
image_filtered = cv2.GaussianBlur(noisy_image, (11,11), 10, 10)
plot_image(image_filtered, noisy_image, title_1 = "Gaussian Blur", title_2 = "Noisy Image")

# Image sharpening

kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])

sharpened = cv2.filter2D(image, -1, kernel)
plot_image(sharpened, noisy_image, title_1 = "Sharpened Image", title_2 = "Noisy Image")

# Edge Detection

img_gray = cv2.imread("C:/Users/achit/OneDrive/Desktop/Pics/barbara.png")
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB))
plt.show()


# Using Gaussian Blur

img_gray = cv2.GaussianBlur(img_gray, (3,3), 0.1, 0.1)
plt.figure(figsize=(10,10))
plt.imshow(img_gray, cmap = "gray")
plt.show()

# Applying sobel function

ddepht = cv2.CV_16S
grad_x = cv2.sobel(img_gray, ddepth, dx = 1, dy = 0, ksize = 3)

grad_y = cv2.Sobel(src=img_gray, ddepth=ddepth, dx=0, dy=1, ksize=3)
plt.imshow(grad_y,cmap='gray')

# Converts the values back to a number between 0 and 255
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)


grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


plt.figure(figsize=(10,10))
plt.imshow(grad, cmap = "gray")
plt.show()

# Median Filtering

image = cv2.imread("C:/Users/achit/OneDrive/Desktop/Pics/cameraman.jpeg")
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

# Filter the image using Median Blur with a kernel of size 5
filtered_image = cv2.medianBlur(image, 5)
# Make the image larger when it renders
plt.figure(figsize=(10,10))
# Renders the image
plt.imshow(filtered_image,cmap="gray")
plt.show()

ret, outs = cv2.threshold(src = image, thresh = 0, maxval = 255, type = cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

# Make the image larger when it renders
plt.figure(figsize=(10,10))

# Render the image
plt.imshow(outs, cmap='gray')