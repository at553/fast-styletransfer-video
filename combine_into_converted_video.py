import cv2, os, numpy as np
from PIL import Image

x = cv2.imread('framesConverted/frame10.jpg')
height, width, layers = x.shape

fourcc = cv2.cv.FOURCC('m', 'p', '4', 'v')

video = cv2.VideoWriter()
out = video.open('output.mov', fourcc, 24, (width, height), True)




for ind in range(len(os.listdir('framesConverted/'))):
    img = cv2.imread('framesConverted/frame%i.jpg' % ind)
    video.write((img))
    print('status: %i' % ind)

cv2.destroyAllWindows()
video.release()
