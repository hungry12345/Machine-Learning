import matplotlib.pyplot as plt
import cv2
import numpy as np

baboon = cv2.imread("C:/Users/achit/OneDrive/Desktop/Pics/baboon.png")

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.show()

A = baboon # Doing this would point to the same location in memory as the orginal baboon image

id(A) == id(baboon) # Will print out true


B = baboon.copy()

id(A) == id(B) # Will print out false because the memory of the baboon image is not the same as the original image

# If we didnt apply the copy() method to the baboon image, and lets say baboon is the same as A, then the memory of the baboon image is the same

baboon[:,:,] = 0

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("baboon")
plt.subplot(122)
plt.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
plt.title("array A")
plt.show()


# Because we use the copy() method, the memory of the baboon image is the same as the original image

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(cv2.cvtColor(baboon, cv2.COLOR_BGR2RGB))
plt.title("baboon")
plt.subplot(122)
plt.imshow(cv2.cvtColor(B, cv2.COLOR_BGR2RGB))
plt.title("array B")
plt.show()

# Flipping Images

image = cv2.imread("C:/Users/achit/OneDrive/Desktop/Pics/cat.png")
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

width, height, C = image.shape
print(width, height, C)

# Flipping the image vertically using array

array_flip = np.zeros((width, height, C), dtype = np.uint8)

for i,row in enumerate(image):
        array_flip[width-1-i,:,:]=row


plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(array_flip, cv2.COLOR_BGR2RGB))
plt.show()

# OpenCV has several ways to flip an image, we can use the flip() function

for flipcode in [0,1,-1]:
    im_flip =  cv2.flip(image,flipcode )
    plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
    plt.title("flipcode: "+str(flipcode))
    plt.show()


# We can also use the rotate function to rotate an image

im_flip = cv2.rotate(image,0)
plt.imshow(cv2.cvtColor(im_flip,cv2.COLOR_BGR2RGB))
plt.show()

flip = {"ROTATE_90_CLOCKWISE":cv2.ROTATE_90_CLOCKWISE,"ROTATE_90_COUNTERCLOCKWISE":cv2.ROTATE_90_COUNTERCLOCKWISE,"ROTATE_180":cv2.ROTATE_180}

flip["ROTATE_90_CLOCKWISE"]

upper = 150
lower = 400
crop_top = image[upper: lower,:,:]
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(crop_top, cv2.COLOR_BGR2RGB))
plt.show()

left = 150
right = 400
crop_horizontal = crop_top[: ,left:right,:]
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(crop_horizontal, cv2.COLOR_BGR2RGB))
plt.show()

# Creating a square on the image using cv2.rectangle()

array_sq = np.copy(image)
array_sq[upper:lower,left:right,:] = 0



plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.title("orignal")
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(array_sq,cv2.COLOR_BGR2RGB))
plt.title("Altered Image")
plt.show()

# Creating a rectangle on the image using cv2.rectangle()
start_point, end_point = (left, upper),(right, lower)
image_draw = np.copy(image)
cv2.rectangle(image_draw, pt1=start_point, pt2=end_point, color=(0, 255, 0), thickness=3)
plt.figure(figsize=(5,5))
plt.imshow(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))
plt.show()

# Putting text on the image using cv2.putText()

image_draw=cv2.putText(img=image,text='Stuff',org=(10,500),color=(255,255,255),fontFace=4,fontScale=5,thickness=2)
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(image_draw,cv2.COLOR_BGR2RGB))
plt.show()