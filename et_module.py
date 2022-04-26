# Import torch library stuff
from pprint import pprint

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Basic imports
import os
import sys
import numpy as np

# Import CoordConv
from CoordConv import AddCoordsTh

import torchvision
from torchvision import datasets, models, transforms

# Import packing
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

# Import Bayesian LSTM
from blitz.modules import BayesianConv2d, BayesianLinear, BayesianLSTM


#######################
### Trying E-D LSTM ###
#######################

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = BayesianConv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias,
                              prior_sigma_1=1.0,
                              prior_sigma_2=0.1)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device="cuda:0"),
                torch.zeros(batch_size, self.hidden_dim, height, width, device="cuda:0"))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            hidden_state = hidden_state
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))


        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class GazePredictor(nn.Module):
    def __init__(self, args):
        super(GazePredictor, self).__init__()

        # Preprocess and transform image
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.coord_conv = AddCoordsTh(x_dim=args['image_height'],
                                 y_dim=args['image_width'],
                                 with_r=False, cuda=True)

        #####################
        #### Load VGG-19 ####
        #####################
        self.vgg_19 = torchvision.models.vgg19(pretrained=True)
        self.vgg_19_features = list(self.vgg_19.features.children())
        # Get first blocks, and freeze them
        self.vgg_19_first_half_feat = nn.Sequential(*self.vgg_19_features[:28])
        for p in self.vgg_19_first_half_feat.parameters():
            p.requires_grad_(False)
        # Get last 512-Conv block and finetune it
        self.vgg_19_last_half_feat = nn.Sequential(*self.vgg_19_features[28:])
        for p in self.vgg_19_last_half_feat.parameters():
            p.requires_grad_(True)
        self.vgg_feat = 512

        ######################################
        #### Load ResNet for segmentation ####
        ######################################
        self.resNet_segmentation = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.sem_feat = 21

        ##############################################
        ###### Fuse Visual and semantic features #####
        ##############################################

        self.image_features = nn.Sequential(
            nn.Conv2d(self.vgg_feat + self.sem_feat, 64, 1),
            nn.Tanh(),
            nn.Conv2d(64, 1, 1),
            nn.Tanh()
        )

        #########################################
        #### ConvLSTM for temporal inference ####
        #########################################

        self.convLSTM = ConvLSTM(input_dim=1 + 3, hidden_dim=[32, 8, 1], kernel_size=(3, 3), num_layers=3)

        # self.conv = BayesianConv2d(in_channels=16, out_channels=1, kernel_size=(1, 1))
        # self.tanh = nn.Tanh()

        ################
        #### Linear ####
        ################

        self.image_w = args['image_width'] // 4
        self.image_h = args['image_height'] // 4
        self.flatten = nn.Flatten()

    def forward(self, image, spatial_maps):

        preprocessed_image = self.preprocess(image)

        # Get image features from VGG
        with torch.no_grad():
            features = self.vgg_19_first_half_feat(preprocessed_image)
        # Features have a shape of [1, 512, H/32, W/32]
        features = self.vgg_19_last_half_feat(features)

        # Get image segmentation from resNet
        # Segmentation has a shape of [B, 21, H, W]
        segmentation = self.resNet_segmentation(preprocessed_image.repeat(2, 1, 1, 1))['out'][0, ...].unsqueeze(0)
        # Interpolated segmentation has a shape of [1, 21, H/32, W/32]
        segmentation = nn.functional.interpolate(segmentation, size=[segmentation.size(2) // 4, segmentation.size(3) // 4])
        features = nn.functional.interpolate(features, size=[segmentation.size(2), segmentation.size(3)])

        # Combine features and segmentation, and convolve them together
        final_image_features = self.image_features(torch.cat([features, segmentation], dim=1)) * 0.8    # Smooth?

        # Combine features, segmentation and spatial map (new T dimension) and coordconv
        # Expected spatial maps shape [B, T, 3, H, W] where T = time seq len, channels = heatmap, coordConvX, coordConvY
        spatial_maps = nn.functional.interpolate(spatial_maps,
                                                 size=[spatial_maps.size(2), features.size(2), features.size(3)])

        # Pass final_image_features from [B, C, H, W] to [B, T, C, H, W]
        final_image_features = final_image_features.unsqueeze(dim=1).repeat(1, spatial_maps.size(1), 1, 1, 1)
        final_input = torch.cat([final_image_features, spatial_maps], dim=2)

        # Pass everything to the LSTM
        # Expected output --> last hidden state [B, T, 256, H, W]
        # Get the very last timestep --> [B, 256, H, W]
        convLSTM_output, _ = self.convLSTM(final_input)
        convLSTM_output = convLSTM_output[-1]
        convLSTM_output = convLSTM_output[:, -1, :, :, :]

        # Expected size [1, 1, H, W]
        # final_map = self.tanh(self.conv(convLSTM_output))
        final_map = nn.functional.interpolate(convLSTM_output, size=[segmentation.size(2) * 4, segmentation.size(3) * 4])
        final_map = (final_map - torch.min(final_map)) / (torch.max(final_map) - torch.min(final_map))

        return final_map