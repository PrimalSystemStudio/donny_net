import torch
from torchvision import datasets
import matplotlib.pyplot as plt

#Download dataset
mnist = datasets.MNIST('./data', download=True)

threes = mnist.data[(mnist.targets == 3)]/255.0
fours = mnist.data[(mnist.targets == 4)]/255.0
sevens = mnist.data[(mnist.targets == 7)]/255.0

print(len(threes))
print(len(sevens))

def show_image(img):
  plt.imshow(img)
  plt.xticks([])
  plt.yticks([])
  plt.show()

print(threes.shape, sevens.shape)

combined_data = torch.cat([threes, sevens])
print(combined_data.shape)
flat_imgs = combined_data.view((-1, 28*28))
print(flat_imgs.shape)
target = torch.tensor([1]*len(threes)+[2]*len(sevens))
target.shape