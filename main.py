from ldr import ldr
import argparse
import cv2
import glob
import os
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument('--image_folder',default='./images')
parser.add_argument('--alpha',default=2.5,type=float)

args = parser.parse_args()



for fn in glob.glob(os.path.join(args.image_folder,'*.jpg')):
	img = cv2.imread(fn,0)

    # % Pre-computing
	U = np.zeros((255,255))
	tmp_k = np.array(range(1,256))
	for layer in range(1,256):
	    U[:,layer-1] = np.minimum(tmp_k,256-layer) - np.maximum(tmp_k-layer,0)

	alpha = 2.5
	transf_fn = ldr(img,alpha,U)


	out = transf_fn[img]
	out = np.squeeze(out)
	out_he = cv2.equalizeHist(img)
	numpy_horizontal = np.hstack((img, out,out_he))

	cv2.imshow('frame',numpy_horizontal)
	cv2.waitKey(10000)
	cv2.destroyAllWindows()
