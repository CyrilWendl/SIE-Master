import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader


#******************** BEGIN UTILITY FUNCTIONS ***************************

def extract_tiles_from_image(img, crop_size, stride, padding=0):
	if padding > 0:
		if len(img.shape) == 2:
			img_ext = np.zeros((img.shape[0]+2*padding, img.shape[1]+2*padding), dtype=np.uint8)
		else: # colored image
			img_ext = np.zeros((img.shape[0]+2*padding, img.shape[1]+2*padding, img.shape[2]), dtype=np.uint8)

		img_ext[padding:img.shape[0]+padding, padding:img.shape[1]+padding] = img
		img = img_ext

	height = img.shape[0]
	width = img.shape[1]
	tiles = []
	for y in range(0, height, stride):
		for x in range(0, width, stride):
			if y + crop_size <= height and x + crop_size <= width:
				tiles.append(img[y:y+crop_size, x:x+crop_size])
	return tiles


def extract_tiles_from_images(img_dir, crop_size, stride, padding=0):
	img_files = [f for f in os.listdir(img_dir)]
	img_files.sort()
	all_tiles = []
	for img_file in img_files:
		img_path = img_dir + img_file
		print img_path
		img_pil = Image.open(img_path)
		img = np.asarray(img_pil)
		tiles = extract_tiles_from_image(img, crop_size, stride, padding)
		all_tiles = all_tiles + tiles
	return np.asarray(all_tiles)


def extract_multiband_tiles_from_directories(list_img_dir, list_nbands, crop_size, stride, padding=0):
	total_nbands = sum(list_nbands)
	# get tiles from the first directory
	img_dir = list_img_dir[0]
	nbands = list_nbands[0]
	tiles = extract_tiles_from_images(img_dir, crop_size, stride, padding)
	ntiles = tiles.shape[0]
	output = np.zeros((ntiles, crop_size, crop_size, total_nbands), dtype=np.uint8)
	output[:,:,:,:nbands] = tiles
	acc_nbands = nbands
	tiles = None
	ndirs = len(list_img_dir)
	for i in xrange(1, ndirs):
		img_dir = list_img_dir[i]
		nbands = list_nbands[i]
		tiles = extract_tiles_from_images(img_dir, crop_size, stride, padding)
		if nbands == 1:
			output[:,:,:,acc_nbands:acc_nbands+nbands] = np.reshape(tiles, [tiles.shape[0], tiles.shape[1], tiles.shape[2],1])
		else:
			output[:,:,:,acc_nbands:acc_nbands+nbands] = tiles

		acc_nbands += nbands
		tiles = None
	return output


def flip_ud_image(img):
	if len(img.shape) == 3:
		out = np.zeros(img.shape, dtype=np.uint8)
		for i in range(img.shape[2]):
			out[:,:,i] = np.flipud(img[:,:,i])
		return out
	else:
		return np.flipud(img)


def flip_lr_image(img):
	if len(img.shape) == 3:
		out = np.zeros(img.shape, dtype=np.uint8)
		for i in range(img.shape[2]):
			out[:,:,i] = np.fliplr(img[:,:,i])
		return out
	else:
		return np.fliplr(img)


def rotate_image(img, angle):
	img_pil = Image.fromarray(img.astype(np.uint8))
	new_img_pil = img_pil.rotate(angle)
	new_img = np.asarray(new_img_pil).astype(np.uint8)
	return new_img


def modify_image_and_gt(img, gt):
	operation = random.randrange(0, 3)
	if operation == 1:
		return flip_ud_image(img), flip_ud_image(gt)
	elif operation == 2:
		return flip_lr_image(img), flip_lr_image(gt)
	else:
		return img, gt


#******************** END UTILITY FUNCTIONS ***************************


class CustomDataset(Dataset):
	def __init__(self, image_tiles, gt_tiles, use_data_augmentation=False):
		self.image_tiles = image_tiles
		self.gt_tiles = gt_tiles

		self.len = len(self.image_tiles)
		self.use_data_augmentation = use_data_augmentation

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		img_tile = self.image_tiles[index]
		gt_tile = self.gt_tiles[index]
		
		if use_data_augmentation:
			image_tile, gt_tile = modify_image_and_gt(img_tile, gt_tile)

		image_tile = torch.from_numpy(image_tile)
		gt_tile = torch.from_numpy(gt_tile)
		return image_tile, gt_tile


class Hypercolumns(nn.Module):
	def __init__(self, num_classes):
		super(Hypercolumns, self).__init__()
		self.conv0 = nn.Conv2d()

	def forward(self, x):
		return y
			
	

def train_model():




def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_path", type=str, help="Dataset path")
	parser.add_argument("num_iterations", type=int, help="Number of optimization iterations")
	parser.add_argument("learning_rate", type=float, help="Learning rate")
	parser.add_argument("use_data_augm", type=int, help="Use data augmentation")
	#parser.add_argument("checkpoints_dir", type=str, help="Checkpoints path")
	parser.add_argument("gpu_id", type=int, help="GPU id")
	parser.add_argument("use_bgclass", type=int, help="Use background class")
	args = parser.parse_args()

	dataset_path = args.dataset_path
	num_iterations = args.num_iterations 
	learning_rate = args.learning_rate 
	use_data_augm = args.use_data_augm
	#checkpoints_dir = args.checkpoints_dir
	gpu_id = args.gpu_id
	use_bg_as_new_class = args.use_bg_as_new_class
	use_bgclass = args.use_bgclass

	input_size = 128

	print "GPU ID {}".format(str(gpu_id))
	print "use_bgclass {}".format(str(use_bgclass))

	size_orig_tile = input_size 
	training_batch_size = 16 
	training_tile_stride = input_size / 2 

	# Read images
	img_dir_train = "{}/orig/train01/".format(dataset_path)
	if use_bgclass:
		gt_dir_train = "{}/gt_erode/train01/".format(dataset_path)
	else:
		gt_dir_train = "{}/gt/train01/".format(dataset_path)
	
	dsm_dir_train = "{}/dsm/train01/".format(dataset_path)
	ir_dir_train = "{}/ir/train01/".format(dataset_path)
	img_dir_test = "{}/orig/test01/".format(dataset_path)
	gt_dir_test = "{}/gt/test01/".format(dataset_path)
	dsm_dir_test = "{}/dsm/test01/".format(dataset_path)
	ir_dir_test = "{}/ir/test01/".format(dataset_path)

	# extract tiles
	img_tiles_train = extract_multiband_tiles_from_directories([img_dir_train, ir_dir_train, dsm_dir_train], [3, 1, 1], size_orig_tile, training_tile_stride)
	gt_tiles_train = extract_tiles_from_images(gt_dir_train, size_orig_tile, training_tile_stride)

	config_vars, session = train_cnn_model(img_tiles_train, gt_tiles_train, num_iterations, training_batch_size, learning_rate, input_size, use_data_augm, use_bgclass, mask_downsample_factor, checkpoints_dir, gpu_id)
	session.close()
	session = None
	config_vars = None
	
	print "================ img_tiles_train.shape {}".format(img_tiles_train.shape)
	# we are here
	config_vars, session = train_cnn_model(img_tiles_train, gt_tiles_train, num_iterations, training_batch_size, learning_rate, input_size, use_data_augm, use_bgclass, mask_downsample_factor, checkpoints_dir, gpu_id)

	start_time  = time.time()
	if gpu_id is not None:
		with tf.device("/gpu:{}".format(str(gpu_id))):
			classify_multiband_images_from_directories([img_dir_test, ir_dir_test, dsm_dir_test], [3, 1, 1], size_orig_tile, size_orig_tile, dataset_path, config_vars, gt_dir = gt_dir_test)
			#classify_multiband_images_from_directories([img_dir_test, ir_dir_test, dsm_dir_test], [3, 1, 1], size_orig_tile, size_output_tile, dataset_path, config_vars, gt_dir = gt_dir_test)			
			#classify_multiband_images_from_directories([img_dir_test, dsm_dir_test], [3, 1], size_orig_tile, size_orig_tile, dataset_path, config_vars, gt_dir = gt_dir_test)
	else:
		classify_multiband_images_from_directories([img_dir_test, ir_dir_test, dsm_dir_test], [3, 1, 1], size_orig_tile, size_orig_tile, dataset_path, config_vars, gt_dir = gt_dir_test)
		#classify_multiband_images_from_directories([img_dir_test, ir_dir_test, dsm_dir_test], [3, 1, 1], size_orig_tile, size_output_tile, dataset_path, config_vars, gt_dir = gt_dir_test)
		#classify_multiband_images_from_directories([img_dir_test, dsm_dir_test], [3, 1], size_orig_tile, size_orig_tile, dataset_path, config_vars, gt_dir = gt_dir_test)
	end_time  = time.time()
	eval_proc_time  = end_time - start_time
	print "Evaluation time {}".format(eval_proc_time)
	
	session.close()
	session = None
	config_vars = None

	

if __name__ == '__main__':
	main()
