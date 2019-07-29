#Seaborn Intro!!!!
import matplotlib.pyplot as plt
import seaborn as sns
sns.lmplot(x='weight', y='hp', data=auto)
plt.show()
#sesiduals plot
sns.residplot(x='hp', y='mpg', data=auto, color='green')

#plotting higher order of regression
sns.regplot(x='weight', y='mpg', data=auto, scatter=None , color='blue', label='order 1')
sns.regplot(x='weight', y='mpg', data=auto, scatter=None , color='green', label='order 2', order = 2)
#grouping with hue
sns.lmplot(x='weight', y='hp', data=auto, hue='origin', palette='Set1')
#row grouping
sns.lmplot(x='weight', y='hp', data=auto, row='origin')

#Univariate plots
sns.stripplot(y='hp', x='cyl', data=auto,jitter=True,size =3)
sns.swarmplot(y='hp', x='cyl', data=auto, hue = 'origin')
sns.violinplot(x='cyl', y='hp', data=auto)

#mixing 2
sns.violinplot(x='cyl', y='hp', data=auto, inner=None, color='lightgray')
sns.stripplot(x='cyl', y='hp', data=auto, jitter=True, size=1.5)

#Multivariate distr
sns.jointplot(x='hp', y ='mpg',data=auto,kind='hex')
sns.pairplot(data=auto,kind='reg',hue='origin')
#heatmap after we build the cov matrix
sns.heatmap(cov_matrix)

#Visualization Time Series
plt.xticks(rotation=45)
plt.title('AAPL: 2007 to 2008')
plt.plot(view, color='blue')
#insert view
view = aapl['2007-11':'2008-04']
plt.plot(aapl)
plt.xticks(rotation=45)
plt.title('AAPL: 2001-2011')
plt.axes([0.25, 0.5, 0.35, 0.35])
plt.plot(view, color='red')
plt.xticks(rotation=45)
plt.title('2007/11-2008/04')
plt.show()

#IMG manipulation rechnigues
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')
plt.subplot(2,1,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image, cmap='gray')
# Assign pixels the flattened 1D numpy array image.flatten() 
pixels = image.flatten()

# Display a histogram of the pixels in the bottom subplot
plt.subplot(2,1,2)
plt.xlim((0,255))
plt.title('Normalized histogram')
plt.hist(pixels, bins=64, color='red', alpha=0.4, range=(0,256), normed=True)
plt.show()
#for 2 axis
plt.twinx()

#Equalizing  img histogram
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')
# Flatten the image into 1 dimension: pixels
pixels = image.flatten()
# Generate a cumulative histogram
cdf, bins, patches = plt.hist(pixels, bins=256, range=(0,256), normed=True, cumulative=True)
new_pixels = np.interp(pixels, bins[:-1], cdf*255)
# Reshape new_pixels as a 2-D array: new_image
new_image = new_pixels.reshape(image.shape)
# Display the new image with 'gray' color map
plt.subplot(2,1,1)
plt.title('Equalized image')
plt.axis('off')
plt.imshow(new_image, cmap='gray')
# Generate a histogram of the new pixels
plt.subplot(2,1,2)
pdf = plt.hist(new_pixels, bins=64, range=(0,256), normed=False,
               color='red', alpha=0.4)
plt.grid('off')
# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()
plt.xlim((0,256))
plt.grid('off')

# Add title
plt.title('PDF & CDF (equalized image)')

# Generate a cumulative histogram of the new pixels
cdf = plt.hist(new_pixels, bins=64, range=(0,256),
               cumulative=True, normed=True,
               color='blue', alpha=0.4)
plt.show()

#Histogram from colored images
# Load the image into an array: image
image = plt.imread('hs-2004-32-b-small_web.jpg')

# Display image in top subplot
plt.subplot(2,1,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image)

# Extract 2-D arrays of the RGB channels: red, blue, green
red, blue, green = image[:,:,0], image[:,:,1], image[:,:,2]

# Flatten the 2-D arrays of the RGB channels into 1-D
red_pixels = red.flatten()
blue_pixels = blue.flatten()
green_pixels = green.flatten()

# Overlay histograms of the pixels of each color in the bottom subplot
plt.subplot(2,1,2)
plt.title('Histograms from color image')
plt.xlim((0,256))
plt.hist(red_pixels, bins=64, normed=True, color='red', alpha=0.2)
plt.hist(blue_pixels, bins=64, normed=True, color='blue', alpha=0.2)
plt.hist(green_pixels, bins=64, normed=True, color='green', alpha=0.2)

# Display the plot
plt.show()

#
# Load the image into an array: image
image = plt.imread('hs-2004-32-b-small_web.jpg')

# Extract RGB channels and flatten into 1-D array
red, blue, green = image[:,:,0], image[:,:,1], image[:,:,2]
red_pixels = red.flatten()
blue_pixels = blue.flatten()
green_pixels = green.flatten()

# Generate a 2-D histogram of the red and green pixels
plt.subplot(2,2,1)
plt.grid('off') 
plt.xticks(rotation=60)
plt.xlabel('red')
plt.ylabel('green')
plt.hist2d(red_pixels, green_pixels, bins=(32,32))

# Generate a 2-D histogram of the green and blue pixels
plt.subplot(2,2,2)
plt.grid('off')
plt.xticks(rotation=60)
plt.xlabel('green')
plt.ylabel('blue')
plt.hist2d(green_pixels, blue_pixels, bins=(32, 32))

# Generate a 2-D histogram of the blue and red pixels
plt.subplot(2,2,3)
plt.grid('off')
plt.xticks(rotation=60)
plt.xlabel('blue')
plt.ylabel('red')
plt.hist2d(blue_pixels, red_pixels, bins=(32, 32))

# Display the plot
plt.show()
