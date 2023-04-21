from turtle import color
import numpy as np # for multidimensional array 
import matplotlib.pyplot as plt # Matplotlib is a plotting library.
from minisom import MiniSom 
# MiniSom is a Numpy based implementation of the Self Organizing Maps.
imageData = plt.imread("image.jpg") # We have to load in our image as an array.
pixels = np.reshape(imageData, (imageData.shape[0]*imageData.shape[1], 3)) / 255.
#This is a 3D matrix. We need to convert the array into a 2D array.
# initialization
n = 4
m = 4
SOM = MiniSom(n, m, 3, sigma=1.0, learning_rate=0.2, neighborhood_function='bubble')  
# 4x4 = 16 final colors
SOM.random_weights_init(pixels)
# I copied the starting weights for later use 
initialWeights = SOM.get_weights().copy()  
#We train our model, the second is the number of iteration here we choose 1000.
SOM.train(pixels, 1000, random_order = True, verbose = True)

quantizationSOM = SOM.quantization(pixels)  # quantize each pixel of the image.
result = np.zeros(imageData.shape)
# place the quantized values into a new image
for j, k in enumerate(quantizationSOM):
    result[np.unravel_index(j, shape=(imageData.shape[0], imageData.shape[1]))] = k

# we visualize it using matplotlib.
plt.figure(figsize=(10, 5)) # Determines the size of the figure.
plt.figure(1)
# I added a title to the entire figure with the suptitle() function and made its color green.
plt.suptitle('Color Quantization', color = "Green") 
# With the subplot() function you can draw multiple plots in one figure 
plt.subplot(1, 4, 1) # the figure has 1 row, 4 columns, and this plot is the first plot.
plt.title('original image') # I used the title() function to set a title for the original image.
plt.imshow(imageData)
plt.subplot(1, 4, 2) # the figure has 1 row, 4 columns, and this plot is the second plot.
plt.title('result image') # I used the title() function to set a title for the result image.
plt.imshow(result)
plt.subplot(1, 4, 3) # the figure has 1 row, 4 columns, and this plot is the third plot.
plt.title('initial colors') # I used the title() function to set a title for the initial colors.
plt.imshow(initialWeights, interpolation='none')
plt.subplot(1, 4, 4) # the figure has 1 row, 4 columns, and this plot is the fourth plot.
plt.title('learned colors') # I used the title() function to set a title for the learned colors.
plt.imshow(SOM.get_weights(), interpolation='none')
plt.tight_layout()
plt.savefig("result.png") # to save the result.
plt.show()
