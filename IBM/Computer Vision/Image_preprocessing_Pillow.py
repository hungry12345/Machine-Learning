def get_concat_h(im1, im2):
    #https://note.nkmk.me/en/python-pillow-concat-images/
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


my_image = "C:/Users/achit/OneDrive/Desktop/Pics/lenna.png"

# Getting the path of the image to be concatenated
import os
cwd = os.getcwd()
image_path = os.path.join(cwd, my_image)


# Loading Image
from PIL import Image

image = Image.open(my_image)

type(image)

# Showing the image

image.show()

# Or we can use matplotlib to show the image

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()

# We could use this method to show the image

image = image.open(image_path)

# We can use size() method to get the size of the image.
# The first element of the tuple is the width and the second element is the height.

print(image.size)

# The mode will be 'RGB' if the image is in RGB mode.

print(image.mode)

# The load() method will load the image into memory.
im = image.load()

x = 0
y = 1

im[x,y]

im = image.save('lenna.png')
# Grayscale images, bascailly a image processing operations

from PIL import ImageOps

image_gray = ImageOps.grayscale(image)

# Checking the image type
print(image_gray.mode) # This will print out 'L' for grayscale images

image_gray.show()

# Quantizing the image by half, makes the intensity less intense

image_gray.quantize(256//2)

image_gray.show()

for n in range(3,8):
    plt.figure(figsize=(10,10))

    plt.imshow(get_concat_h(image_gray,  image_gray.quantize(256//2**n)))
    plt.title("256 Quantization Levels  left vs {}  Quantization Levels right".format(256//2**n))
    plt.show()


# Working with different colors

baboon = Image.open("C:/Users/achit/OneDrive/Desktop/Pics/baboon.png")

red, green, blue = baboon.split()
get_concat_h(baboon, red)
get_concat_h(baboon, green)
get_concat_h(baboon, blue)

# Using Numpy to convert the image to a numpy array

import numpy as np

array = np.array(image)
print(type(array))

print(array)

array[0,0]
array.min()
array.max()

# Plotting the array as an image

plt.figure(figsize=(10,10))
plt.imshow(array)
plt.show()

# Returning 256 columns corresponding to the first half of the image

columns = 256
plt.figure(figsize=(10,10))
plt.imshow(array[:,0:columns,:])
plt.show()

# Copying an array for another variable

A = array.copy()
plt.imshow(A)
plt.show()

B = A
A[:,:,:] = 0
plt.imshow(B)
plt.show()

# Converting the image to RGB

baboon_array = np.array(baboon)
plt.figure(figsize=(10,10))
plt.imshow(baboon_array)
plt.show()

# Converting the image to grayscale
baboon_array = np.array(baboon)
plt.figure(figsize=(10,10))
plt.imshow(baboon_array[:,:,0], cmap='gray')
plt.show()

# Creating a new array and set all but red color channels to 0,

baboon_red=baboon_array.copy()
baboon_red[:,:,1] = 0
baboon_red[:,:,2] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_red)
plt.show()

baboon_blue=baboon_array.copy()
baboon_blue[:,:,0] = 0
baboon_blue[:,:,1] = 0
plt.figure(figsize=(10,10))
plt.imshow(baboon_blue)
plt.show()


# write your code here
from PIL import Image
lenna = Image.open('lenne')
lenna_array = np.array(lenna)
lenna_blue = lenna_array.copy()
lenna_blue[:,:,0] = 0
lenna_blue[:,:,1] = 0
plt.figure(figsize=(10,10))
plt.imshow(lenna_blue)
plt.show()