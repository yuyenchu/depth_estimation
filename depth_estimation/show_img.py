import cv2
import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

m = tf.keras.models.load_model("./models/model_20211003_171915.h5")
img = np.array(PIL.Image.open("./annotations/test/pikachu.jpeg").resize((128,128)))
# w, h, *_ = img.shape
# w, h = w//2, h//2
# img = img[w-64:w+64, h-64:h+64]
print(img.shape)
plt.imshow(img)
plt.axis('off')
plt.show()
out = (m(np.array([img/127.5-1])).numpy()[0]+1.0)*127.5
print(out)
plt.imshow(out.squeeze(),cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.show()
