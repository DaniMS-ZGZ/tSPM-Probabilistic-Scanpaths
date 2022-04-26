from __future__ import print_function
#%matplotlib inline
import configargparse
import os
import random
import torch
import numpy as np
'''
Suppress SourceChangeWarning - we have removed comment lines and debug options from the source code.
'''
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)


# Import inference module
from inference import basic_inference
from CoordConv import AddCoordsTh
from et_module import GazePredictor


if __name__ == "__main__":

	# Parse arguments
	parser = configargparse.ArgumentParser()
	parser.add_argument("--mode", required=True, type=str, default='inference', help="[Current] Running mode: --mode inference")
	parser.add_argument("-n", required=False, type=int, default=10, help="Select the desired number of scanpaths. Default = 10")
	parser.add_argument("-th", required=False, type=float, default=0.5, help="Select the desired threshold th. Default = 0.5")
	opt = parser.parse_args()

	# Currently, there is only one admited parameter
	if opt.mode == 'inference':

		# Model currently developed to work on GPU
		device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
		print("* Working on " + str(device))
		assert str(device) == "cuda:0"
  
		# Load model
		args = {"image_height": 256, "image_width": 384}
		model = GazePredictor(args).cuda()
		# Load checkpoint
		checkpoint_path = "models/best.ckpt"
		checkpoint = torch.load(checkpoint_path)
		# Parse parameters (remove first "model." word)
		parsed_parameters = {}
		for key, value in checkpoint['state_dict'].items():
			parsed_parameters[key.replace("model.", "")] = value
		model.load_state_dict(parsed_parameters)
		model.eval()
		print("* Model correctly read.")

		# Basic inference
		image_path = "data/test_2.jpg"		# The path where your image is
		path_to_save = "test/"				# The output path to save your image
		basic_inference(image_path=image_path, args=args, model=model, device=device, path_to_save=path_to_save, n=20, th=0.5)
		print("** Done.")

