import numpy as np
import argparse
from PIL import Image
import time

import chainer
from chainer import cuda, Variable, serializers
from net import *


# parser.add_argument('--gpu', '-g', default=-1, type=int,
#                     help='GPU ID (negative value indicates CPU)')
# parser.add_argument('--model', '-m', default='models/composition.model', type=str)
# parser.add_argument('--out', '-o', default='out.jpg', type=str)
def generate(img):
    parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
    #parser.add_argument('input')

    args = parser.parse_args()

    model = FastStyleNet()
    serializers.load_npz('models/seurat.model', model)
    if -1 >= 0:
        cuda.get_device(-1).use()
        model.to_gpu()
    xp = np if -1 < 0 else cuda.cupy

    start = time.time()
    image = xp.asarray(Image.open(img).convert('RGB'), dtype=xp.float32).transpose(2, 0, 1)
    image = image.reshape((1,) + image.shape)
    x = Variable(image)

    y = model(x)
    result = cuda.to_cpu(y.data)

    result = result.transpose(0, 2, 3, 1)
    result = result.reshape((result.shape[1:]))
    result = np.uint8(result)
    print(time.time() - start, 'sec')

    # Image.fromarray(result).save(args.out)
    return Image.fromarray(result)

generate('frames/frame1255.jpg').show()