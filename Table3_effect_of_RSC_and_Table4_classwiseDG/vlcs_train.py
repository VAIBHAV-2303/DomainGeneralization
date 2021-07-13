import argparse

import pickle
import torch
#from IPython.core.debugger import set_trace
from torch import nn
#from torch.nn import functional as F
from data import vlcs_data_helper
## from IPython.core.debugger import set_trace
from data.vlcs_data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np
from models.resnet import resnet18, resnet50
from models.inceptionresnetv2 import MyInceptionResnet

train_loss = []
train_acc = []
test_loss = []
test_acc = []

def get_args():
	parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
	parser.add_argument("--target", choices=available_datasets, help="Target")
	parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
	parser.add_argument("--image_size", type=int, default=222, help="Image size")
	# data aug stuff
	parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
	parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
	parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
	parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
	parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
	#
	parser.add_argument("--limit_source", default=None, type=int,
						help="If set, it will limit the number of training samples")
	parser.add_argument("--limit_target", default=None, type=int,
						help="If set, it will limit the number of testing samples")
	parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
	parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
	parser.add_argument("--n_classes", "-c", type=int, default=5, help="Number of classes")
	parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="resnet18")
	parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
	parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
	parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
	parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
	parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
	parser.add_argument("--classify_only_sane", default=False, type=bool, help="If true, the network will only try to classify the non scrambled images")
	parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
	parser.add_argument("--suffix", default="", help="Suffix for the logger")
	parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")
	parser.add_argument("--rsc", type=bool, default=False, help="If true will run with rsc")
	parser.add_argument("--optimizer", type=str, default="sgd", help="optimizer for training")
	parser.add_argument("--root_dir", type=str, default="./data/images", help="Root Directory for VLCS Dataset")

	return parser.parse_args()

class Trainer:
	def __init__(self, args, device):
		self.args = args
		self.device = device
		self.rsc = args.rsc
		if args.network == 'resnet18':
			model = resnet18(pretrained=True, classes=args.n_classes)
		elif args.network == 'resnet50':
			model = resnet50(pretrained=True, classes=args.n_classes)
		elif args.network == "inception":
			model = MyInceptionResnet(args.n_classes)
		else:
			model = resnet18(pretrained=True, classes=args.n_classes)
		self.model = model.to(device)
		# print(self.model)
		self.source_loader = vlcs_data_helper.get_train_dataloader(args, patches=model.is_patch_based())
		self.target_loader = vlcs_data_helper.get_val_dataloader(args, patches=model.is_patch_based())

		self.val_loader = self.target_loader

		self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
		self.len_dataloader = len(self.source_loader)
		print("Dataset size: train %d, val %d, test %d" % (
		len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
		self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all, args.optimizer,
																 nesterov=args.nesterov)
		self.n_classes = args.n_classes
		if args.target in args.source:
			self.target_id = args.source.index(args.target)
			print("Target in source: %d" % self.target_id)
			print(args.source)
		else:
			self.target_id = None

	def _do_epoch(self, epoch=None):
		criterion = nn.CrossEntropyLoss()
		epoch_loss = 0
		epoch_acc = 0
		total = 0
		self.model.train()
		for it, batch in enumerate(self.source_loader):
			data, class_l = batch["img"].to(self.device), batch["label"].to(self.device)
			self.optimizer.zero_grad()

			data_flip = torch.flip(data, (3,)).detach().clone()
			data = torch.cat((data, data_flip))
			class_l = torch.cat((class_l, class_l))

			class_logit = self.model(data, class_l, self.rsc, epoch)
			class_loss = criterion(class_logit, class_l)
			_, cls_pred = class_logit.max(dim=1)
			loss = class_loss

			loss.backward()
			self.optimizer.step()

			epoch_loss += loss.item()
			epoch_acc += torch.sum(cls_pred == class_l.data).item()
			total += class_l.shape[0]

			self.logger.log(it, len(self.source_loader),
							{"class": class_loss.item()},
							{"class": torch.sum(cls_pred == class_l.data).item(), }, data.shape[0])
			del loss, class_loss, class_logit

		train_loss.append(epoch_loss)
		train_acc.append(100 * epoch_acc / total)
		self.model.eval()
		with torch.no_grad():
			for phase, loader in self.test_loaders.items():
				total = len(loader.dataset)

				class_correct, loss = self.do_test(loader)

				class_acc = float(class_correct) / total

				if phase == "test":
					test_acc.append(100*class_acc)
					test_loss.append(loss)

				self.logger.log_test(phase, {"class": class_acc})
				self.results[phase][self.current_epoch] = class_acc

	def do_test(self, loader):
		criterion = nn.CrossEntropyLoss()
		class_correct = 0
		epoch_loss = 0
		for it, batch in enumerate(loader):
			data, class_l = batch["img"].to(self.device), batch["label"].to(self.device)

			class_logit = self.model(data, class_l, False)
			class_loss = criterion(class_logit, class_l)
			epoch_loss += class_loss.item()
			_, cls_pred = class_logit.max(dim=1)

			class_correct += torch.sum(cls_pred == class_l.data)

		return class_correct, epoch_loss


	def do_training(self):
		self.logger = Logger(self.args, update_frequency=30)
		self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
		for self.current_epoch in range(self.args.epochs):
			self.scheduler.step()
			self.logger.new_epoch(self.scheduler.get_lr())
			self._do_epoch(self.current_epoch)
		val_res = self.results["val"]
		test_res = self.results["test"]
		idx_best = val_res.argmax()
		print("Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
		val_res.max(), test_res[idx_best], test_res.max(), idx_best))
		self.logger.save_best(test_res[idx_best], test_res.max())
		return self.logger, self.model


def main():
	args = get_args()
	# args.source = ['art_painting', 'cartoon', 'sketch']
	# args.target = 'photo'
	# args.source = ['art_painting', 'cartoon', 'photo']
	# args.target = 'sketch'
	# args.source = ['art_painting', 'photo', 'sketch']
	# args.target = 'cartoon'
	# args.source = ['photo', 'cartoon', 'sketch']
	# args.target = 'art_painting'
	# --------------------------------------------
	if args.source is None:
		args.source = ["CALTECH", "LABELME", "PASCAL"]
	if args.target is None:
		args.target = "SUN"

	print("NUM CLASSES", args.n_classes) 
	print("Source Domain:", args.source)
	print("RSC", args.rsc)
	print("Target domain: {}".format(args.target))
	# print("Output Dir", args.output)
	torch.manual_seed(0)
	torch.cuda.manual_seed(0)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	trainer = Trainer(args, device)
	trainer.do_training()

	# pickle.dump(train_loss, open(args.output+"/train_loss.pkl", "wb"))
	# pickle.dump(train_acc, open(args.output+"/train_acc.pkl", "wb"))
	# pickle.dump(test_loss, open(args.output+"/test_loss.pkl", "wb"))
	# pickle.dump(test_acc, open(args.output+"/test_acc.pkl", "wb"))

	# with open(f"./{args.output}/stats.txt", "a") as f:
		# f.write(str(max(train_acc))+" "+str(max(test_acc))+"\n")

if __name__ == "__main__":
	torch.backends.cudnn.benchmark = True
	main()
