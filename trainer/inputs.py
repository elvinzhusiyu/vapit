
# Create the train and label lists
import math
import numpy as np
#import imageio
#from google.cloud import storage
import io
import tensorflow as tf
from PIL import Image

#------
def readImage(image_path):
    img = tf.io.gfile.GFile(image_path, 'rb').read()
    img = io.BytesIO(img)
    img = np.array(Image.open(img).convert('L')).astype(dtype='int32')
    height, width = img.shape
    # clipping 
    h = int(height/16)*16
    w = int(width/16)*16
    image = img[:h, :w]
    return image

#------
def load_data(data_file_path: str, label_file_path: str, rangeIndices, batch_size) -> tf.data.Dataset:
    images = []
    for i in rangeIndices:
        im = readImage(data_file_path+'/image_inline_i%04d.png' % i)
        im = np.array(im).astype(dtype='float32')/255
        (h,w) = im.shape
        im = np.reshape(im, (h,w,1))
        images.append(im)

    labels = []
    for i in rangeIndices:
        im = readImage(label_file_path+'/image_inline_i%04d.png' % i)
        im = np.array(im).astype(dtype='float32')
        (h,w) = im.shape
        im = np.reshape(im, (h,w,1))
        labels.append(im)

    #for items in images:
    #    print(items)

    seismic = np.array(images)
    label = np.array(labels)
    print("Data  Shape = ", images.shape)
    print("Label Shape = ", label.shape)
    
    #return images, labels
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(100).batch(batch_size)
    
    return dataset

