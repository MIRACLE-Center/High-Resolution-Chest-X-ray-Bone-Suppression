import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
def plot_img_and_mask(img, mask):##把图和mask 直接显示出来
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    a.set_title('Input image')
    # img=img.astype(np.uint8)
    plt.imshow(img[:,:,0],cmap='gray')

    b = fig.add_subplot(1, 2, 2)
    b.set_title('Output mask')
    # mask=mask.astype(np.uint8)
    plt.imshow(mask,cmap='gray')
    plt.show()
# img=Image.open(r"C:\Users\hp\Desktop\0cdf5b5d0ce1_01.jpg")
# mask=Image.open(r"C:\Users\hp\Desktop\0cdf5b5d0ce1_01_OUT.jpg")
# plot_img_and_mask(img,mask)
