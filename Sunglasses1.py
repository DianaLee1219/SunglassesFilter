# Import libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Load the Face Image
faceImage = cv2.imread("musk.jpg")

plt.imshow(faceImage[:,:,::-1]);plt.title("Face")

# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread("sunglass1.png",-1)
glassPNG = np.float32(glassPNG)/255

# Import libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'


# Load the Face Image
faceImage = cv2.imread("musk.jpg")

plt.imshow(faceImage[:,:,::-1]);plt.title("Face")

# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread("sunglass1.png",-1)
glassPNG = np.float32(glassPNG)/255

# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG, None, fx=0.3, fy=0.3)
glassHeight, glassWidth, nChannels = glassPNG.shape

# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Convert mask to binary (0 or 1)
glassMask1 = np.where(glassMask1 > 0, 1, 0).astype(np.float32)  # Convert nonzero values to 1

# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');

# Make a copy
faceWithGlassesNaive = faceImage.copy()
faceImage = np.float32(faceImage)/255

# Top left corner of the glasses
topLeftRow = 80
topLeftCol = 130

bottomRightRow = topLeftRow + glassHeight
bottomRightCol = topLeftCol + glassWidth

# Resize sunglasses to match the exact target size
glassBGR = cv2.resize(glassBGR, (glassWidth, glassHeight))
glassMask1 = cv2.resize(glassMask1, (glassWidth, glassHeight))

# Replace the eye region with the sunglass image
faceWithGlassesNaive[topLeftRow:bottomRightRow,topLeftCol:bottomRightCol] = glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])
# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI= faceWithGlassesArithmetic[topLeftRow:bottomRightRow,topLeftCol:bottomRightCol]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI,(1 -  glassMask ))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR, glassMask)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Combine the Sunglass in the Eye Region to get the augmented image
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)
alpha = 0.7  # Transparency factor (0 = fully transparent, 1 = fully opaque)
eyeRoiFinal = cv2.addWeighted(eyeROI, 1 - alpha, eyeRoiFinal, alpha, 0)

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")
# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[topLeftRow:bottomRightRow,topLeftCol:bottomRightCol]= eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");
