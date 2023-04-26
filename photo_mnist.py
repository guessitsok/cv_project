import os
import math
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms as tfs

from scipy.ndimage import center_of_mass
from PIL import Image



def get_best_shift(img):
	"""
	Function to measure the photo center of mass
	and the direction of shifting

	Args:
		img (str): path to image

	Returns:
		int: direction of shifting photo across x coord
		int: direction of shifting photo across y coord
	"""

	rows, cols = img.shape
	cy, cx = center_of_mass(img)
	shift_x = np.round(cols / 2.0 - cx).astype(int)
	shift_y = np.round(rows / 2.0 - cy).astype(int)

	return shift_x, shift_y


def shift(image, shift_x, shift_y):
	"""
	Function to shift image to center it for MNIST dataset

	Args:
		image (str): the path to image in your system
		shift_x (int): the value of the shift across x coord
		shift_y (int): the values of the shift across y coord

	Returns:
		numpy.ndarray: output image that has the size dsize and the same type as src
	"""

	rows, cols = image.shape
	# M - matrix for warpAffine
	warp_affine_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
	shifted = cv.warpAffine(image, warp_affine_matrix, (cols, rows))
	return shifted


def rec_digit(image_path):
	"""
	Function to preprocess random photo with handwritten digit
	to MNIST dataset, then to give it to model

	Args:
		image_path (str): path to image

	Returns:
		numpy.ndarray: preprocess image with dimensions: 1, 28, 28 
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


data_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize((0.5), (0.5))
])


def _imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


def test_model(photos):
	model = torch.load('./models/MNIST_model.pt')
	model.eval()
	model_predict = []

	for photo in photos:
		mnist_example = rec_digit(image_path=photo)
		mnist_example_tfs = data_tfs(mnist_example)
		mnist_example_tfs = mnist_example_tfs.permute(1, 2, 0).to(torch.float32)
		
		output_example = model(mnist_example_tfs)
		_, predicted_example = torch.max(output_example, 1)
		model_predict.append(predicted_example.item())

	n_test = 8
	df_result = pd.DataFrame({
		'Ground Truth': [6, 7, 0, 4, 0, 5, 1, 6],
		'Predicted label': model_predict[:n_test]})
	
	print(df_result)
	# imshow(torchvision.utils.make_grid(mnist_example_tfs[:n_test, :, :, :], nrow=n_test))
	

def some_test():
	photos = []
	for filename in os.listdir('./exs'):
		f = os.path.join('./exs', filename)
		photos.append(f)
	return photos

photos = some_test()
test_model(photos=photos)
