import os
import cv2
import matplotlib.pyplot as plt

def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

my_image = "C:/Users/achit/OneDrive/Desktop/Pics/lenna.png"
cwd = os.getcwd()
image_path = os.path.join(cwd, my_image)
image = cv2.imread(my_image)

type(image)

image.min()
image.max()

# Using imshow to display the image

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.show()

# The RGB channels are different, so we need to convert them to BGR

new_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

plt.figure(figsize=(10, 10))
plt.imshow(new_image)
plt.show()

# Loading image using the path if it is not in the directory

image = cv2.imread(image_path)
image.shape

# Saving the image as in jpg format

cv2.imwrite("lenna.jpg", image)

# Grayscale images, bascailly a image processing operations

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_gray.shape

# Plotting the image as an image

plt.figure(figsize=(10, 10))
plt.imshow(image_gray, cmap='gray')
plt.show()

# Saving the image as in jpg format

cv2.imwrite("lenna_gray.jpg", image_gray)


# Loading grayscale image

im_gray = cv2.imread("C:/Users/achit/OneDrive/Desktop/Pics/barbara.png", cv2.IMREAD_GRAYSCALE)

# Plotting the image as an image


plt.figure(figsize=(10, 10))
plt.imshow(im_gray, cmap='gray')
plt.show()


# Working on color channels

baboon = cv2.imread("C:/Users/achit/OneDrive/Desktop/Pics/baboon.png")

# Plotting the image as an image

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()

# Obtaining the 3 color channel

blue, green, red = baboon[:, :, 0], baboon[:, :, 1], baboon[:, :, 2]

# Concatenating the 3 color channels

im_bgr = cv2.vconcat([blue, green, red])

# Plotting the image next to the original image

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("RGB image")
plt.subplot(122)
plt.imshow(im_bgr,cmap='gray')
plt.title("Different color channels  blue (top), green (middle), red (bottom)  ")
plt.show()


# Indexing and slicing the image

rows = 256
plt.figure(figsize=(10,10))
plt.imshow(new_image[0:rows,:,:])
plt.show()


columns = 256
plt.figure(figsize=(10,10))
plt.imshow(new_image[:,0:columns,:])
plt.show()

# Slicing columns and rows

plt.figure(figsize=(10,10))
plt.imshow(new_image[0:rows,0:columns,:])
plt.show()

# Copying an array for another variable

A = new_image.copy()
plt.imshow(A)
plt.show()

# Converting the image to RED

baboon_red = baboon.copy()
baboon_red[:, :, 0] = 0
baboon_red[:, :, 1] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon_red, cv2.COLOR_BGR2RGB))
plt.show()

# Converting the image to BLUE

baboon_blue = baboon.copy()
baboon_blue[:, :, 2] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon_blue, cv2.COLOR_BGR2RGB))
plt.show()

# Converting the image to GREEN

baboon_green = baboon.copy()
baboon_green[:, :, 1] = 0
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon_green, cv2.COLOR_BGR2RGB))
plt.show()
