import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from glob import glob

## Code snippets you'll need in no particular order

#right-hand sides
#1) np.reshape(avgDorm,(m,n))
#2) img.flatten() 
#3) np.linalg.svd(X,full_matrices=0)

# Identify naming pattern of files to be read into Python from glob import glob
dir = '.'
jpg_files = sorted(glob(f'{dir}/1970s_Version/*.jpg'))

# Compute number of image files
M = len(jpg_files)

# Compute size of image files
img=io.imread(jpg_files[0], as_gray=True)
m,n = img.shape

#run movie
ax = None
for i in range(0,M):
    filename = jpg_files[i]
    img=io.imread(filename, as_gray=True)
    if ax is None:
        ax = plt.imshow(img,cmap=plt.cm.gray)
    else:
        ax.set_data(img)
    plt.pause(.001)
    plt.draw()

# Initialize matrix whose columns correspond to individual images
trainingDorms = np.zeros((m*n,M))

# Loop over all image files
for i in range(0,M):

    # Acquire the filename of image i
    filename = jpg_files[i]

    # Read image i into Python as a 64 x 64 matrix
    img=io.imread(filename, as_gray=True)

    #insert image as column vector in matrix
    #-------- your code begins here--
    trainingDorms[:,i] = img.flatten()
    #-------- your code ends here----

avgDorm = np.mean(trainingDorms,axis=1)[:,np.newaxis]

#reshape column vector and display image 
#-------- your code begins here-------
img = np.reshape(avgDorm,(m,n))
#-------- your code ends here----

plt.imshow(img,cmap=plt.cm.gray)
plt.show()

# Compute eigendorms on mean-subtracted training data
X = trainingDorms - np.tile(avgDorm,(1,M))
#-------- your code begins here--
U, S, VT = np.linalg.svd(X,full_matrices=0)
#-------- your code ends here--

fig1 = plt.figure()
ax1 = fig1.add_subplot(331)
plt.axis('off')
ls = np.reshape(U[:,0],(m,n))
plt.imshow(ls,cmap=plt.cm.gray)
ax1 = fig1.add_subplot(332)
plt.axis('off')
ls = np.reshape(U[:,1],(m,n))
plt.imshow(ls,cmap=plt.cm.gray)
ax1 = fig1.add_subplot(333)
plt.axis('off')
ls = np.reshape(U[:,2],(m,n))
plt.imshow(ls,cmap=plt.cm.gray)
ax1 = fig1.add_subplot(334)
plt.axis('off')
ls = np.reshape(U[:,3],(m,n))
plt.imshow(ls,cmap=plt.cm.gray)
ax1 = fig1.add_subplot(335)
plt.axis('off')
ls = np.reshape(U[:,4],(m,n))
plt.imshow(ls,cmap=plt.cm.gray)
ax1 = fig1.add_subplot(336)
plt.axis('off')
ls = np.reshape(U[:,5],(m,n))
plt.imshow(ls,cmap=plt.cm.gray)
ax1 = fig1.add_subplot(337)
plt.axis('off')
ls = np.reshape(U[:,6],(m,n))
plt.imshow(ls,cmap=plt.cm.gray)
ax1 = fig1.add_subplot(338)
plt.axis('off')
ls = np.reshape(U[:,7],(m,n))
plt.imshow(ls,cmap=plt.cm.gray)
ax1 = fig1.add_subplot(339)
plt.axis('off')
ls = np.reshape(U[:,8],(m,n))
plt.imshow(ls,cmap=plt.cm.gray)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
plt.tight_layout()
plt.show()

# Plot right singular vector 1 and 3
plt.plot(VT[0,:])
plt.plot(VT[1,:])
plt.plot(VT[2,:])
plt.legend(['1','2','3'])
plt.show()