#Changing Style 
plt.style.use('seaborn-notebook')
print(plt.style.available)
#Separate axis
plt.axes([0.05 ,0.05,0.425,0.9])
plt.plot(year, physical_sciences, color='blue')
plt.axes([0.525 ,0.05,0.425,0.9])
plt.plot(year, computer_science, color='red')
plt.show()

#Same with subplots
# Create a figure with 1x2 subplot and make the left subplot active
plt.subplot(1,2,1)
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')
plt.subplot(1,2,2)
plt.plot(year, computer_science, color='red', label='Physical Sciences')
plt.title('Computer Science')
plt.legend(loc='lower center')
# Use plt.tight_layout() to improve the spacing between subplots
plt.tight_layout()
plt.show()


#zOOMING
plt.xlim(1990 , 2010)
plt.ylim(0,50)
plt.axis((1990,2010,0,50))
#SAVE THE image
plt.savefig('xlim_and_ylim.png')

#annotate something
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max+5, cs_max+5), arrowprops=dict(facecolor='black'))

#style
plt.style.use('ggplot')

#Generating MESHES! for 2d visualization
# Import numpy and matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
u = np.linspace(-2,2,41)
v = np.linspace(-1,1,21)
X,Y = np.meshgrid(u,v)
Z = np.sin(3*np.sqrt(X**2 + Y**2)) 
plt.pcolor(Z)
plt.show()

plt.subplot(2,2,1)
plt.contour(X, Y, Z)
plt.subplot(2,2,2)
plt.contour(X, Y, Z, 20)
plt.subplot(2,2,3)
plt.contourf(X, Y, Z)
plt.subplot(2,2,4)
plt.contourf(X, Y, Z, 20)
plt.tight_layout()
plt.show()

plt.subplot(2,2,1)
plt.contourf(X,Y,Z,20, cmap='viridis')
plt.colorbar()
plt.title('Viridis')
plt.subplot(2,2,2)
plt.contourf(X,Y,Z,20, cmap='gray')
plt.colorbar()
plt.title('Gray')
plt.subplot(2,2,3)
plt.contourf(X,Y,Z,20, cmap='autumn')
plt.colorbar()
plt.title('Autumn')
plt.subplot(2,2,4)
plt.contourf(X,Y,Z,20, cmap='winter')
plt.colorbar()
plt.title('Winter')
plt.tight_layout()
plt.show()

#Visualizing Bivariate distribution#####
# Generate a 2-D histogram
#plt.hist2d(hp, mpg, bins=(20,20), 
 #          range=((40,235), (8, 48)))
 plt.hexbin(hp, mpg, gridsize=(15,12), 
           extent=(40,235,8, 48))
plt.colorbar()
plt.tight_layout()
plt.show()

#VIZUALIZING IMAGES!!!
img = plt.imread('480px-Astronaut-EVA.jpg')
plt.imshow(img)
# Hide the axes
plt.axis('off')
plt.show()

#Convert RGB to Greyscale!!!
intensity = img.sum(axis=2)
plt.imshow(intensity,'gray')
plt.colorbar()

#Rescaling IMAGES!!!
pmin, pmax = image.min(), image.max()
rescaled_image = 256*(image-pmin)/(pmax-pmin)

