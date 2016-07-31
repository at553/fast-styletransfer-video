from generate import *
import cv2
import os
from PIL import Image

listimgs = os.listdir('frames/')


for ind in range(len(listimgs)):
    x = generate('frames/frame%i.jpg' % (ind))
    x.save('framesConverted/frame%i.jpg' % (ind))
    print(ind)