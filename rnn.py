import torch
import torch.nn as nn
import timm

class RBclassifier(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.conv_model = timm.create_model('resnet18', features_only=True, in_chans=1)
		with torch.inference_mode():
			x = torch.rand(1, 1, 28, 28)
			out = self.conv_model(x)
			conv_out_shape = out[-1].shape
			conv_out_size = conv_out_shape[1] * conv_out_shape[2] * conv_out_shape[3]

		self.rnn_layer = nn.LSTM(input_size=conv_out_size, hidden_size=hidden_size)
		self.fc = nn.Linear(hidden_size, 1)


	def forward(self, x):
		shape = x.shape
		conv_out = self.conv_model(x.view(-1, 28, 28))
		out, (h, c) = self.rnn_layer(conv_out.view(shape[0], shape[1], -1))
		y = self.fc(h)
		return y
