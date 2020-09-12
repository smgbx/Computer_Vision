#!/usr/bin/env python
# coding: utf-8

# In[80]:


# https://giusedroid.blogspot.com/2015/04/using-python-and-k-means-in-hsv-color.html
get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
from scipy.cluster.vq import vq, kmeans
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import glob
import os, os.path
import random
from sklearn.utils import shuffle


# In[20]:


target_img = 'mountains.jpg'
img = cv2.imread(target_img)
hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
hsv_image.shape


# In[57]:


# DONE
# https://stackoverflow.com/questions/26392336/importing-images-from-a-directory-python-to-list-or-dictionary
# https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder/47114735

images = []
count = 0
path = "C:\Users\Shelby\Desktop\UMKC\Academics\Fall20\ComputerVision\NWPU-RESISC45"
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    for filename in os.listdir(folder_path):
        if count < 100:
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                images.append(hsv_img) 
                count += 1
        else:
            len(images)
            count = 0
            break

len(images)


# In[58]:


images[0].shape


# In[77]:


# DONE

flattened_images = []
for image in images:
    flattened_images.append(image.reshape(-1, 3))

flattened_images[0].shape


# In[87]:


# DONE
# https://stackoverflow.com/questions/38190476/use-of-random-state-parameter-in-sklearn-utils-shuffle
# https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html

sample_pixels = []
for image in flattened_images:
    sample = shuffle(image, random_state=0)[:100] # shuffle image matrix, then return first 100 pixels
    for pixel in sample:
        sample_pixels.append(pixel)

len(sample_pixels) # = 150,000
sample_pixels[0] # = 1 x 3 array


# In[4]:


hue, sat, val = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]


# In[5]:


plt.figure(figsize=(10,8))

plt.subplot(311) # 3 rows, 1 column, next details are for the first cell
plt.subplots_adjust(hspace=.5)
plt.title("Hue")
plt.hist(np.ndarray.flatten(hue), bins = 128)

plt.subplot(312)
plt.title("Saturation")
plt.hist(np.ndarray.flatten(sat), bins = 128)

plt.subplot(313)
plt.title("Value")
plt.hist(np.ndarray.flatten(val), bins = 128)

plt.show()


# In[39]:


from PIL import Image


# In[ ]:


pixels = []
# for class in range(0, 15) of dataset:
    # for image in range(0, 100) of class:
        # convert each image to hsv
        # get 100 random pixels in format h,s,v (h x s x v)
        # reshape into n x 3
for i in range(0,15):
    for j in range(0,100):
        class_images = dataset[i][j]
    for j in range(0,100):
        

#hsv_table = kmeans(hsv_train, 64) # data (all colors of data set) -> n x 3, K


# In[38]:


#hist = cv2.calcHist([image], [channels], mask, [bins per channel], [range]
hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8,4,4], [0, 1, 0, 1, 0, 1])
plt.figure()
print hist.shape, hist.flatten().shape[0]


# In[31]:


def getColorTable(hsv_image):
    #h,w,dim = hsv_image.shape
    #cluster_data = hsv_image.reshape(h, dim).astype(float)
    cluster_data = np.float32(hsv_image.reshape(-1, 3))
    print(cluster_data.shape)
    #cluster_data = np.array(cluster_data[:,0:3], dtype=np.double)
    #centers, assignments = kmeans(np.array(cluster_data[:,0:3], dtype=np.float), K)
    #centers, assignments = kmeans(cluster_data, K)
    #print(assignents.shape())
    
    #data, dist = vq(cluster_data[:, 0:channels], centers)
    #weights = [len(data[data == i]) for i in range (0, K)]
    
    # creates a 4 column matrix in which the first element is the weight and the other three
    # represent the h, s and v values for each cluster
    #color_rank = np.column_stack((weights, centers))
    # sorts by cluster weight
    #color_rank = color_rank[np.argsort(color_rank[:,0])]
    
   # fig = plt.figure()
    #ax = Axes3D(fig)
    #ax.scatter(assignments(:,1), assignments(:,2), assignments(:,3), '*')
    #plt.show()


# In[32]:


do_cluster(hsv_image, 64)


# In[21]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(assignments(:,1), assignments(:,2), assignments(:,3), '*')

