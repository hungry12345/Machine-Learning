import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import ImageOps

baboon = np.array(Image.open("C:/Users/achit/OneDrive/Desktop/Pics/baboon.png"))
plt.figure(figsize=(10, 10))
plt.imshow(baboon)
plt.show()

A = baboon # Doing this would point to the same location in memory as the baboon image

# We can verify whehter the baboon image has the same memory as the original image

id(A) == id(baboon)

# However if we use copy() method, the memory of the baboon image is not the same as the original image

B = baboon.copy()
id(A) == id(B) # Will print out false because the memory of the baboon image is not the same as the original image

#  Comparing the orginal image of Baboon and copy of A

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(baboon)
plt.title("baboon")
plt.subplot(122)
plt.imshow(A)
plt.title("array A")
plt.show()

# We can see that the memory of the baboon image is not the same as the original image

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(baboon)
plt.title("baboon")
plt.subplot(122)
plt.imshow(B)
plt.title("array B")
plt.show()

image = Image.open("C:/Users/achit/OneDrive/Desktop/Pics/cat.png")
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()


# Casting into the numpy array and find its shape
# This is also the old traditional way of flipping of an image
array = np.array(image)
width, height, C = array.shape
print(width, height, C)

# Flipping the image vertically

array_flip = np.zeros((height, width, C), dtype=np.uint8)

# Let's flip the image vertically using Pillow

im_flip = ImageOps.flip(image)
plt.figure(figsize=(10,10))
plt.imshow(im_flip)
plt.show()

# Flipping the image horizontally using transpose()
im_flip = image.transpose(1)
plt.imshow(im_flip)
plt.show()

# The image module has many keyword functions to rotate an image

flip = {"FLIP_LEFT_RIGHT": Image.FLIP_LEFT_RIGHT,
        "FLIP_TOP_BOTTOM": Image.FLIP_TOP_BOTTOM,
        "ROTATE_90": Image.ROTATE_90,
        "ROTATE_180": Image.ROTATE_180,
        "ROTATE_270": Image.ROTATE_270,
        "TRANSPOSE": Image.TRANSPOSE,
        "TRANSVERSE": Image.TRANSVERSE}

# We can see the value intergers

flip["FLIP_LEFT_RIGHT"]

# Plotting the orginal image with flipped image

for key, values in flip.items():
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("orignal")
    plt.subplot(1,2,2)
    plt.imshow(image.transpose(values))
    plt.title(key)
    plt.show()


upper = 150
lower = 400

# Cropping an Image

upper = 150
lower = 400


# Cropping the upper and lower part of the image
crop_top = array[upper: lower,:,:]
plt.figure(figsize=(10,10))
plt.imshow(crop_top)
plt.show()

# Cropping and left and right part of the image

left = 150
right = 400

crop_left = array[:,left:right,:]
plt.figure(figsize=(10,10))
plt.imshow(crop_left)
plt.show()


# Cropping and top and bottom part of the image using Pillow
image = Image.open("C:/Users/achit/OneDrive/Desktop/Pics/cat.png")
crop_image = image.crop((upper, lower, right, lower))
plt.figure(figsize=(10,10))
plt.imshow(crop_image)
plt.show()


# Flipping the new image

crop_image = crop_image.transpose(Image.FLIP_LEFT_RIGHT)
plt.figure(figsize=(10,10))
plt.imshow(crop_image)
plt.show()

# Changing the specific Image Pixels

array_sq = np.copy(array)
array_sq[upper:lower, left:right, 1:2] = 0

# Comparing the results with the new image

plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
plt.imshow(array)
plt.title("orignal")
plt.subplot(1,2,2)
plt.imshow(array_sq)
plt.title("Altered Image")
plt.show()

# Using ImageDraw module from PIL library

from PIL import ImageDraw

image_draw = image.copy()
image_fn = ImageDraw.Draw(im = image_draw)

# Whatever we change in image_fn, will change in image_draw

shape = [left, upper, right, lower]
image_fn.rectangle(xy = shape, fill = 'red')

# plotting the new image

plt.figure(figsize=(10,10))
plt.imshow(image_draw)
plt.show()

# Using text on image

from PIL import ImageFont

image_fn.text(xy = (0,0), text = "Hello World", fill = (0,0,0))

plt.figure(figsize=(10,10))
plt.imshow(image_draw)
plt.show()

# Overlaying images

image_lenna = Image.open("C:/Users/achit/OneDrive/Desktop/Pics/lenna.png")
array_lenna = np.array(image_lenna)

# Reassign the pixel values

array_lenna[upper:lower, left:right,:] = array[upper:lower, left:right,:]
plt.imshow(array_lenna)
plt.show()


# We can also use the paste() method allows you to overlay one image over another

image_lenna.paste(crop_image, box = (left,upper))

plt.imshow(image_lenna)
plt.show()

# We can also use method copy() applies to some PIL objects

image = Image.open("C:/Users/achit/OneDrive/Desktop/Pics/cat.png")
new_image = image
copy_image = new_image.copy()

# Checking the memory of the new image

id(image) == id(new_image)
id(image) == id(copy_image)



image_fn= ImageDraw.Draw(im=image)
image_fn.text(xy=(0,0),text="box",fill=(0,0,0))
image_fn.rectangle(xy=shape,fill="red")


plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(new_image)
plt.subplot(122)
plt.imshow(copy_image)
plt.show()