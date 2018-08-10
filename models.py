import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
	def __init__(self, output_size):
		super(Policy, self).__init__()

		self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
		self.fc = nn.Linear(2048, 256)
		self.head = nn.Linear(256, output_size)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		out = F.relu((self.conv1(x)))
		out = F.relu(self.conv2(out))
		out = F.relu(self.fc(out.view(out.size(0), -1)))
		out = self.softmax(self.head(out))
		return out


class Value(nn.Module):
	def __init__(self):
		super(Value, self).__init__()

		self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
		self.fc = nn.Linear(2048, 256)
		self.head = nn.Linear(256, 1)

	def forward(self, x):
		out = F.relu((self.conv1(x)))
		out = F.relu(self.conv2(out))
		out = F.relu(self.fc(out.view(out.size(0), -1)))
		out = self.head(out)
		return out
