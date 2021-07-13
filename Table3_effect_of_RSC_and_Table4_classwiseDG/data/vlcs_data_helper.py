from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import StandardDataset
from data.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info, JigsawTestDatasetMultiple
from data.concat_dataset import ConcatDataset
from data.JigsawLoader import JigsawNewDataset, JigsawTestNewDataset, customDataset, testDataset

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = [mnist, mnist, svhn, usps]
available_datasets = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets
#office_paths = {dataset: "/home/enoon/data/images/office/%s" % dataset for dataset in office_datasets}
#pacs_paths = {dataset: "/home/enoon/data/images/PACS/kfold/%s" % dataset for dataset in pacs_datasets}
#vlcs_paths = {dataset: "/home/enoon/data/images/VLCS/%s/test" % dataset for dataset in pacs_datasets}
#paths = {**office_paths, **pacs_paths, **vlcs_paths}

dataset_std = {mnist: (0.30280363, 0.30280363, 0.30280363),
			   mnist_m: (0.2384788, 0.22375608, 0.24496263),
			   svhn: (0.1951134, 0.19804622, 0.19481073),
			   synth: (0.29410212, 0.2939651, 0.29404707),
			   usps: (0.25887518, 0.25887518, 0.25887518),
			   }

dataset_mean = {mnist: (0.13909429, 0.13909429, 0.13909429),
				mnist_m: (0.45920207, 0.46326601, 0.41085603),
				svhn: (0.43744073, 0.4437959, 0.4733686),
				synth: (0.46332872, 0.46316052, 0.46327512),
				usps: (0.17025368, 0.17025368, 0.17025368),
				}


class Subset(torch.utils.data.Dataset):
	def __init__(self, dataset, limit):
		indices = torch.randperm(len(dataset))[:limit]
		self.dataset = dataset
		self.indices = indices

	def __getitem__(self, idx):
		return self.dataset[self.indices[idx]]

	def __len__(self):
		return len(self.indices)


def get_train_dataloader(args, patches):
	root = args.root_dir
	img_transformer, tile_transformer = get_train_transformers(args)
	train_dataset = customDataset(root + args.source[0], root + args.source[1], root + args.source[2], img_transformer)
	loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
	return loader

def get_val_dataloader(args, patches=False):
	img_tr = get_val_transformer(args)
	root = args.root_dir
	dataset = testDataset(root + args.target, img_tr)
	loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
	return loader

def get_train_transformers(args):
	img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
	#img_tr = [transforms.Resize((args.image_size, args.image_size))]
	#img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
	if args.random_horiz_flip > 0.0:
		img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
	if args.jitter > 0.0:
		img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
	img_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
	img_tr.append(transforms.ToTensor())
	img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

	tile_tr = []
	if args.tile_random_grayscale:
		tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
	tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

	return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
	img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
			  transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
	return transforms.Compose(img_tr)


# def get_target_jigsaw_loader(args):
#     img_transformer, tile_transformer = get_train_transformers(args)
#     name_train, _, labels_train, _ = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % args.target), 0)
#     dataset = JigsawDataset(name_train, labels_train, patches=False, img_transformer=img_transformer,
#                             tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
#     return loader
