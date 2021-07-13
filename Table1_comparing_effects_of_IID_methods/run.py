import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PACSDataset import PACSDataset
from model import MyResnet, MyVGG, MyInceptionResnet, MyAlexnet

# Parameters
epochs = 30
batchSize = 32
learningRate = 0.001

# Loading data
transform = transforms.Compose([
	# transforms.Resize((224, 224)),
	transforms.RandomResizedCrop((224, 224), (0.8, 1)),
	transforms.RandomHorizontalFlip(0.5),
	transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
	transforms.RandomGrayscale(0.1),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
trainset = PACSDataset(root_dir="./data", transform=transform, dataCats=["sketch", "cartoon", "photo"])
print("Train Size:", len(trainset))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=2)

# Test set
transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
testset = PACSDataset(root_dir="./data", transform=transform, dataCats=["art_painting"])
print("Test Size:", len(testset))
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=2)
print("Loaded data")

# Loading model
# model = MyInceptionResnet().cuda()
# model = MyResnet().cuda()
# model = MyVGG().cuda()
model = MyAlexnet().cuda()
print("Loaded models")

# Defining optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.0005)
# optimizer = optim.SGD(model.parameters(), lr=learningRate, weight_decay=.0005, momentum=.9, nesterov=False)
print("Configured criterion")

# Training 
trainLosses = []
testLosses = []
testAccuracy = []
trainAccuracy = []
for epoch in range(epochs):

	trainLoss = 0
	correct = 0
	model.train()
	for i, data in enumerate(trainloader, 0):		

		# zero the parameter gradients
		optimizer.zero_grad()

		# Getting data
		inputs, labels = data[0].cuda(), data[1].cuda()

		outputs = model(inputs)
		_, predicted = torch.max(outputs.data, 1)
		correct += (predicted == labels).sum().item()
		loss = criterion(outputs, labels)

		# Gradient descent
		loss.backward()
		optimizer.step()

		# Logging
		trainLoss += loss.item()
		# print("Epoch:", epoch+1, "Batch:", i+1, "Loss:", loss.item())
	
	acc = 100*correct/len(trainset)
	trainAccuracy.append(acc)
	trainLosses.append(trainLoss)

	# Accuracy on test set
	correct = 0
	testLoss = 0
	model.eval()
	with torch.no_grad():
		for data in testloader:
			images, labels = data[0].cuda(), data[1].cuda()
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			correct += (predicted == labels).sum().item()
			testLoss += criterion(outputs, labels).item()
	testLosses.append(testLoss)
	acc = 100*correct/len(testset)
	testAccuracy.append(acc)
	print("===========================")
	print("Epoch:", epoch+1)
	print("Training Accuracy:", trainAccuracy[-1])
	print("Testing Accuracy:", testAccuracy[-1])
	print("Training Loss:", trainLosses[-1])
	print("Testing Loss:", testLosses[-1])
	print("===========================")

print("Finished Training")

# Plotting
# plt.subplot(1, 2, 1)
# plt.plot(np.arange(1, len(trainLosses)+1), trainLosses, label="train")
# plt.plot(np.arange(1, len(testLosses)+1), testLosses, label="test")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(np.arange(1, len(trainAccuracy)+1), trainAccuracy, label="train")
# plt.plot(np.arange(1, len(testAccuracy)+1), testAccuracy, label="test")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()

# plt.savefig('plot.png')


# Save models
# import pickle
# with open("model.pkl", 'wb') as f:
# 	pickle.dump(model, f)