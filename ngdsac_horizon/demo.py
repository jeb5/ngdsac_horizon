import torch
import numpy as np

from torchvision import transforms

from skimage.io import imsave
from skimage.draw import line, set_color, ellipse

import argparse
import os

from ngdsac_horizon.horizon_estimator import HorizonEstimator

import cv2

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Estimate horizon lines using a trained (NG-)DSAC network.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('input', type=str,
	help='input file to process, any image or video format supported by OpenCV')

parser.add_argument('--model', '-m', default=None, 
	help='a trained network')

parser.add_argument('--capacity', '-c', type=int, default=4, 
	help='controls the model capactiy of the network, must match the model to load (multiplicative factor for number of channels)')

parser.add_argument('--imagesize', '-is', type=int, default=256, 
	help='size of input images to the network, must match the model to load')

parser.add_argument('--inlierthreshold', '-it', type=float, default=0.05, 
	help='threshold used in the soft inlier count, relative to image size')

parser.add_argument('--inlieralpha', '-ia', type=float, default=0.1, 
	help='scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)')

parser.add_argument('--inlierbeta', '-ib', type=float, default=100.0, 
	help='scaling factor within the sigmoid of the soft inlier count')

parser.add_argument('--hypotheses', '-hyps', type=int, default=16, 
	help='number of line hypotheses sampled for each image')

parser.add_argument('--uniform', '-u', action='store_true', 
	help='disable neural-guidance and sample data points uniformely, use with a DSAC model')

parser.add_argument('--scorethreshold', '-st', type=float, default=0.4, 
	help='threshold on soft inlier count for drawing the estimate (range 0-1)')

parser.add_argument('--verbose', '-v', action='store_true', 
	help='add vizualizations of neural guidance, soft inler count and hypothesis score to the output')

opt = parser.parse_args()

output_folder = 'out_' + opt.input
if not os.path.isdir(output_folder): os.makedirs(output_folder)

horizon_estimator = HorizonEstimator(
	capacity=opt.capacity,
	imagesize=opt.imagesize,
	inlierthreshold=opt.inlierthreshold,
	inlierbeta=opt.inlierbeta,
	inlieralpha=opt.inlieralpha,
	hypotheses=opt.hypotheses,
	model=opt.model,
	uniform=opt.uniform,
	device=device,
)


def process_and_draw_frame(image):
	"""
	Estimate horizon line for an image and return a visualization.

	image -- 3 dim numpy image tensor
	"""

	image = transforms.functional.to_tensor(image)  # convert to tensor

	est_parameters, score, info = horizon_estimator.process_image(image)

	image_src = info["img_src"]
	points = info["points"]
	log_probs = info["log_probs"]
	batch_inliers = info["batch_inliers"]

	def draw_line(data, lX1, lY1, lX2, lY2, clr):
		'''
		Draw a line with the given color and opacity.

		data -- image to draw to
		lX1 -- x value of line segment start point
		lY1 -- y value of line segment start point
		lX2 -- x value of line segment end point
		lY2 -- y value of line segment end point
		clr -- line color, triple of values
		'''

		rr, cc = line(lY1, lX1, lY2, lX2)
		set_color(data, (rr, cc), clr)

	def draw_models(labels, clr, data):
		'''
		Draw circles for a batch of images.
	
		labels -- line parameters, array shape (Nx2) where 
			N is the number of images in the batch
			2 is the number of line parameters (offset,  slope)
		data -- batch of images to draw to
		'''

		# number of image in batch
		n = labels.shape[0]

		for i in range (n):

			#line
			lY1 = int(labels[i, 0] * opt.imagesize)
			lY2 = int(labels[i, 1] * opt.imagesize + labels[i, 0] * opt.imagesize)
			draw_line(data[i], 0, lY1, opt.imagesize, lY2, clr)

		return data	

	def draw_wpoints(points, data, weights, clrmap):
		'''
		Draw 2D points for a batch of images.

		points -- 2D points, array shape (Nx2xM) where 
			N is the number of images in the batch
			2 is the number of point dimensions (x, y)
			M is the number of points
		data -- batch of images to draw to
		weights -- array shape (NxM), one weight per point, for visualization
		clrmap -- OpenCV color map for visualizing weights
			
		'''

		# create explicit color map
		color_map = np.arange(256).astype('u1')
		color_map = cv2.applyColorMap(color_map, clrmap)
		color_map = color_map[:,:,::-1] # BGR to RGB

		n = points.shape[0] # number of images
		m = points.shape[2] # number of points

		for i in range (0, n):

			s_idx = weights[i].sort(descending=False)[1] # draw low weight points first
			weights[i] = weights[i] / weights[i].max() # normalize weights for visualization

			for j in range(0, m):

				idx = int(s_idx[j])

				# convert weight to color
				clr_idx = float(min(1, weights[i,idx]))
				clr_idx = int(clr_idx * 255)
				clr = color_map[clr_idx, 0] / 255

				# draw point
				r = int(points[i, 0, idx] * opt.imagesize)
				c = int(points[i, 1, idx] * opt.imagesize)
				rr, cc = ellipse((r, r), c, 2)
				set_color(data[i], (rr, cc), clr)

		return data

	image_src = image_src.cpu().permute(0, 2, 3, 1).numpy()  # Torch to Numpy
	viz_probs = image_src.copy() * 0.2  # make a faint copy of the input image

	# draw estimated line
	if score > opt.scorethreshold:
		image_src = draw_models(est_parameters, clr=(0, 0, 1), data=image_src)

	viz = [image_src]

	if opt.verbose:	
		# create additional visualizations

		# draw faint estimated line 
		viz_score = viz_probs.copy()
		viz_probs = draw_models(est_parameters, clr=(0.3, 0.3, 0.3), data=viz_probs)
		viz_inliers = viz_probs.copy()

		# draw predicted points with neural guidance and soft inlier count, respectively
		viz_probs = draw_wpoints(points, viz_probs, weights=torch.exp(log_probs), clrmap=cv2.COLORMAP_PLASMA)
		viz_inliers = draw_wpoints(points, viz_inliers, weights=batch_inliers, clrmap=cv2.COLORMAP_WINTER)

		# create a explicit color map for visualize score of estimate line
		color_map = np.arange(256).astype('u1')
		color_map = cv2.applyColorMap(color_map, cv2.COLORMAP_HSV)	
		color_map = color_map[:,:,::-1]

		# map score to color
		score = int(score*100) #using only the first portion of HSV to get a nice (red, yellow, green) gradient
		clr = color_map[score, 0] / 255

		viz_score = draw_models(est_parameters, clr=clr, data=viz_score)

		viz = viz + [viz_probs, viz_inliers, viz_score]

	# undo zero padding of inputs
	padding_left, padding_right, padding_top, padding_bottom = info["padding"]
	if padding_left > 0:
		viz = [img[:,:,padding_left:,:] for img in viz]
	if padding_right > 0:
		viz = [img[:,:,:-padding_right,:] for img in viz]
	if padding_top > 0:
		viz = [img[:,padding_top:,:,:] for img in viz]
	if padding_bottom > 0:
		viz = [img[:,:-padding_bottom,:,:] for img in viz]		

	# convert to a single uchar image
	viz = np.concatenate(viz, axis=2)
	viz = viz * 255
	viz = viz.astype('u1')

	return viz[0]

# try to read input as image
image = cv2.imread(opt.input)

if image is not None:
	# success, it was an image
	viz = process_and_draw_frame(image)
	imsave(output_folder + "/result.png", viz)

else:
	#failure, try interpreting it as video
	cap = cv2.VideoCapture(opt.input)
	iteration = 0

	while(cap.isOpened()):
		ret, image = cap.read()

		if not ret:
			break

		print("Processing frame %5d." % iteration)

		viz = process_and_draw_frame(image)
		imsave(output_folder + "/frame_" + str(iteration).zfill(5) + ".png", viz)

		iteration = iteration + 1

print('Done.')
