import math
import os
import math
import sys
import cv2
from torchvision.transforms import transforms
import random
import scipy
import torch
import torch.nn as nn
from matplotlib import pyplot as plt, cm
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
from scipy.ndimage import gaussian_filter

# Import required modules
from CoordConv import AddCoordsTh
from et_module import GazePredictor


def basic_inference(image_path, args, model, device, path_to_save, n, th):
	with torch.no_grad():
		
		print("* Evaluating image %s." % image_path)

		# Read image
		original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
		original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
		original_h, original_w, _ = original_image.shape
		# Reshape image and save original ratio
		ratio_w = original_w / args["image_width"]
		ratio_h = original_h / args["image_height"]
		image = cv2.resize(original_image, (args["image_width"], args["image_height"]),
						   interpolation=cv2.INTER_AREA)
		image = image.astype(np.float32) / 255.0
		tr = transforms.Compose([transforms.ToTensor()])
		tensor_image = tr(image).cuda().unsqueeze(dim=0)

		suffix_save = image_path.split("/")[-1].split(".")[0]

		# Figure to print results
		plt.clf()
		plt.close('all')
		fig, ax = plt.subplots(5, 4, figsize=(70,50))

		# Length of the scanpaths to generate
		seq_len = 8
		for ii in range(n):

			# The input for the network
			spatial_input = []
			# The output maps from the network
			output_maps = []
			# Separated predicted x and y per timestep
			output_x = []
			output_y = []

			# CoordConv to include each element to the netx training prediction
			coord_conv = AddCoordsTh(x_dim=args["image_height"],
									 y_dim=args["image_width"],
									 with_r=False, cuda=True)

			# First point. In this case, the center of the image.
			x_train = [args["image_width"] / 2]
			y_train = [args["image_height"] / 2]

			# Spatialize the points as in the paper
			cov_point = 750
			gaussian = np.random.multivariate_normal([x_train[0], y_train[0]], [[cov_point, 0], [0, cov_point]], 4000)
			freq = np.zeros((args["image_height"], args["image_width"]))
			for g in gaussian:
				try:
					freq[int(g[1]), int(g[0])] = freq[int(g[1]), int(g[0])] + 1
				except:
					pass
			freq = gaussian_filter(freq, sigma=7)
			freq /= float(np.nanmax(freq))
			freq = (freq * 2) - 1
			# Append to the inputs
			spatial_input.append(freq)

			# Prepare network input
			spatial_input = torch.FloatTensor(spatial_input).unsqueeze(dim=1).cuda()
			spatial_input = coord_conv(spatial_input).unsqueeze(dim=0)
			train_size = spatial_input.size(1)
			for x in range(train_size):
				output_maps.append(spatial_input[0, x, 0, ...].unsqueeze(dim=0).unsqueeze(dim=0))
			# For each timestep
			for step in range(seq_len):
				# Forward model
				output_map = model(tensor_image, spatial_input)
				# Save the output
				output_maps.append(output_map)
				# Normalize map
				output_map_array = (output_map - torch.min(output_map)) / (
						torch.max(output_map) - torch.min(output_map))
				# Check probs (th is a parameter, check the paper)
				output_map_array[output_map_array <= th] = 0
				output_map_array = output_map_array.squeeze().cpu()
				indices = np.arange(output_map_array.shape[0] * output_map_array.shape[1])
				probabilities = output_map_array.flatten().numpy()
				probabilities /= probabilities.sum()
				# Get a random point based on probability weights
				random_choice = np.random.choice(indices, p=probabilities)
				random_x = (random_choice % args["image_width"]) / args["image_width"]
				random_y = (random_choice / args["image_width"]) / args["image_height"]
				output_x.append(random_x.item() * args["image_width"])
				output_y.append(random_y.item() * args["image_height"])
				# Include output to input, again spatialized
				gaussian = np.random.multivariate_normal([random_x * args["image_width"],
														  random_y * args["image_height"]], [[cov_point, 0], [0, cov_point]],
														 4000)
				freq = np.zeros((args["image_height"], args["image_width"]))
				for g in gaussian:
					try:
						freq[int(g[1]), int(g[0])] = freq[int(g[1]), int(g[0])] + 1
					except:
						pass
				freq = gaussian_filter(freq, sigma=7)
				freq /= float(np.nanmax(freq))
				freq = (freq * 2) - 1
				freq = torch.FloatTensor(freq).cuda().unsqueeze(dim=0).unsqueeze(dim=0)
				# Append to the input
				freq = coord_conv(freq)
				freq = freq.unsqueeze(dim=0)
				spatial_input = torch.cat([spatial_input, freq], dim=1)

			# Prepare to plot
			all_x = np.array(x_train + output_x)
			all_y = np.array(y_train + output_y)
			idx_x = ii // 4
			idx_y = ii % 4
			ax[idx_x, idx_y].imshow(image)
			ax[idx_x, idx_y].axis('off')
			ax[idx_x, idx_y].plot(x_train[0], y_train[0], marker='o', markersize=35., color='blue',alpha=.6)
			ax[idx_x, idx_y].plot(all_x, all_y, linewidth=14, color='blue', alpha=.5)
			for x, y in zip(all_x, all_y):
				ax[idx_x, idx_y].plot(x, y, marker='o', markersize=35., color='blue', alpha=.6)
			t = ax[idx_x, idx_y].annotate("Start", (x_train[0], y_train[0]), fontsize=55.)
			t.set_bbox(dict(facecolor='white', alpha=0.45, edgecolor='black'))
			for ann, txt in enumerate(range(seq_len)):
				t = ax[idx_x, idx_y].annotate("%d" % (ann + 1), (all_x[ann], all_y[ann]), fontsize=55.)
			t.set_bbox(dict(facecolor='white', alpha=0.20, edgecolor='black'))

		# Save
		plt.savefig("test/result_2.jpg")
		plt.clf()




