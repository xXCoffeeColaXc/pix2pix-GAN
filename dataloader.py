from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot 

SIZE_X = 256
SIZE_Y = 256
#n_classes=4 #Number of classes for segmentation

class DataLoader():
    
    def __init__(self) -> None:
        self.src_images = self.get_src_images()
        self.tar_images = self.get_tar_images()

        print("src: {}, tar: {}", len(self.src_images), len(self.tar_images))


    def get_src_images(self):
        src_images = []
        for directory_path in glob.glob("data/masks/"):
            for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
                mask = cv2.imread(mask_path, 1)       
                mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
                src_images.append(mask)
        
        #Convert list to array for machine learning processing          
        src_images = np.array(src_images)
        return src_images

    def get_tar_images(self):
        tar_images = []
        for directory_path in glob.glob("data/images/"):
            for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
                img = cv2.imread(img_path, 1)       
                img = cv2.resize(img, (SIZE_Y, SIZE_X))
                tar_images.append(img)
       
        #Convert list to array for machine learning processing        
        tar_images = np.array(tar_images)
        return tar_images

    def preprocess_data(self):
        X1, X2 = self.src_images, self.tar_images
        # scale from [0,255] to [-1,1]
        X1 = (X1 - 127.5) / 127.5
        X2 = (X2 - 127.5) / 127.5
        return [X1, X2]
    

    def plot_sample_images(self):
        n_samples = 3
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(self.src_images[i])
        # plot target image
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(self.tar_images[i])
        pyplot.show()

