import os
import math
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as tfs

from scipy.ndimage import center_of_mass
from PIL import Image
# from models.CNN_model import MNIST_model
# from models.CNN_model import MNIST_model

class MNIST_model(nn.Module):
    def __init__(self):
        super(MNIST_model, self).__init__()

        self.conv1 = nn.Sequential(
            # nn.Flatten(),
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #             nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #             nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(128, 64)
        self.out_2 = nn.Linear(64, 10)

    def forward(self, x):
        print(f"INPUT:{x.shape}")
        x = self.conv1(x)
        print(f"CONV1 x:{x.shape}")
        x = self.conv2(x)
        print(f"CONV2 x:{x.shape}")
        x = self.conv3(x)
        print(f"CONV3 x:{x.shape}")
        x = self.conv4(x)
        print(f"CONV4 x:{x.shape}")
        x = self.conv5(x)
        print(f"CONV5 x:{x.shape}")
        x = self.conv6(x)
        print(f"CONV6 x:{x.shape}")

        x = x.view(x.size(0), -1)
        print(f"AFTER VIEW x:{x.shape}")
        linear_1 = self.out(x)
        logits = self.out_2(linear_1)
        return F.log_softmax(logits, dim=1)


def get_best_shift(img):
	"""
	Function to measure the photo center of mass and the direction of shifting

	Parameters
	__________
	img : str
		Path to image

	Returns
	_______
	shift_x : int
		direction of shifting photo across x coord
	shift_y : int
		direction of shifting photo across y coord
	"""
	rows, cols = img.shape
	cy, cx = center_of_mass(img)
	shift_x = np.round(cols / 2.0 - cx).astype(int)
	shift_y = np.round(rows / 2.0 - cy).astype(int)

	return shift_x, shift_y


def shift(image, shift_x, shift_y):
	"""
	Function to shift image to center it for MNIST dataset.
	
	Parameters
	__________
	image : str
		the path to image in your system
	shift_x : int
		the value of the shift across x coord
	shift_y : int
		the values of the shift across y coord

	Returns
	_______
	numpy.ndarray
		output image that has the size dsize and the same type as src
	"""
	rows, cols = image.shape
	# M - matrix for warpAffine
	warp_affine_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
	shifted = cv.warpAffine(image, warp_affine_matrix, (cols, rows))
	return shifted


def rec_digit(image_path):
	"""
	Function to preprocess random photo to MNIST dataset.
	Then give that photo to model.
	
	Parameters
	__________
	image_path : str
		path to image
	
	Returns
	_______
	numpy.ndarray
		preprocess image with dimensions: 1, 28, 28 
	"""
	image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
	gray = 255 - image
	thresh, gray = cv.threshold(gray, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

	# up bound of photo
	while np.sum(gray[0]) == 0:
		gray = gray[1:]
	
	# left bound of photo
	while np.sum(gray[:, 0]) == 0:
		gray = np.delete(gray, 0, 1)
	
	# bottom bound of photo
	while np.sum(gray[-1]) == 0:
		gray = gray[:-1]
	
	# right bound of photo
	while np.sum(gray[:, -1]) == 0:
		gray = np.delete(gray, -1, 1)

	rows, cols = gray.shape
	
	# Изменяем размер, чтобы изображение помещалось в box 20x20 pixels
	if rows > cols:
		factor = 20.0 / rows
		rows = 20
		cols = int(round(factor * cols))
		gray = cv.resize(gray, (cols, rows))
	else:
		factor = 20.0 / cols
		cols = 20
		rows = int(round(factor * rows))
		gray = cv.resize(gray, (cols, rows))
	
	# Расширяем картинку до 28 пикселей, добавляя черные столбцы и ряды по краям
	cols_padding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
	rows_padding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
	gray = np.lib.pad(gray, (rows_padding, cols_padding), 'constant')

	# Сдвигаем центр масс
	shift_x, shift_y = get_best_shift(gray)
	shifted = shift(gray, shift_x, shift_y)
	gray = shifted

	image = gray / 255.0  # scale pixels from 0 to 1
	image = np.array(image).reshape(1, 28, 28)
	return image


def _imshow(img):
	"""
	Function to display photos used in mnist notebook.

	Parameters
	__________
	img : numpy.ndarray
		numpy array with pixels 
	"""
	img = img / 2 + 0.5 # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.axis('off')
	plt.show()


def test_model(photos):
	"""
	Function that test model and prints df with tests.

	Parameters
	__________
	photos : list
		list that containts paths to example photos
	
	Returns
	_______
	None
	"""
	data_tfs = tfs.Compose([
		tfs.ToTensor(),
		tfs.Normalize((0.5), (0.5))
	])
	model = MNIST_model()
	model = torch.load('./models/cnn_mnist_model.pt', map_location=torch.device('cpu'))
	# print(1)
	model.eval()
	print("ready to predict")
	model_predict = []

	for photo in photos:
		mnist_example = rec_digit(image_path=photo)
		print(mnist_example.shape, type(mnist_example))
		mnist_example_tfs = data_tfs(np.float32(mnist_example))
		print(mnist_example_tfs.shape, type(mnist_example_tfs))
		mnist_example_tfs = mnist_example_tfs.permute(1, 0, 2).to(torch.float32)
		print(mnist_example_tfs.shape, type(mnist_example_tfs))

		output_example = model(mnist_example_tfs[None, :, :, :])
		_, predicted_example = torch.max(output_example, 1)
		model_predict.append(predicted_example.item())

	ground_truth = [1, 1, 2, 2, 2, 3, 3, 3, 3,
	4, 4, 4, 4, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8,
	9, 9, 9, 9, 0, 4, 0, 5, 1, 6]
	n_test = len(ground_truth)
	df_result = pd.DataFrame({
		'Ground Truth': ground_truth,
		'Predicted label': model_predict[:n_test]})
	
	print(df_result)
	# imshow(torchvision.utils.make_grid(mnist_example_tfs[:n_test, :, :, :], nrow=n_test))
	

def some_test():
	"""
	Function to test model.

	Returns
	_______
	photos : list
		list that containts paths to example photos
	"""
	photos = []
	for filename in os.listdir('./exs'):
		f = os.path.join('./exs', filename)
		photos.append(f)
	return photos

if __name__ == '__main__':
	photos = some_test()
	test_model(photos=photos)
