import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from modules import *

# The entire extraction system
class SE(nn.Module):
	def __init__(self, N = 256, L = 40, B = 256, H = 512, P = 3, X = 8, R = 4, C = 2):
		super(SE, self).__init__()
		self.N, self.L, self.B, self.H, self.P, self.X, self.R, self.C = N, L, B, H, P, X, R, C

		self.encoder = Encoder(L, N)
		self.separator = TemporalConvNet(N, B, H, P, X, R, C)
		self.decoder = Decoder(N, L)
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_normal_(p)
		# It contains a pretrain visual frontend
		self.visual_frontend = visualFrontend()
		self.visual_frontend.load_state_dict(torch.load('pretrain_model/frontend.pt', map_location="cuda"),strict=False)
		# Load the pretrain backend
		selfState = self.state_dict()
		loadedState = torch.load('pretrain_model/backend.pt')		
		for name, param in loadedState.items():
			if name not in selfState:
				print("%s is not in the model."%origName)
				continue
			selfState[name].copy_(param)

	def forward(self, mixture, visual):
		# Get the visual feature
		B, T, W, H = visual.shape
		#print(visual.shape)
		visual = visual.view(B*T, 1, 1, W, H)
		#print(visual.shape)
		visual = (visual / 255 - 0.4161) / 0.1688

		visual = self.visual_frontend(visual)
		#print(visual.shape)
		visual = torch.squeeze(visual, dim=1)
		#print(visual.shape)
		visual = visual.view(B, T, 512)
		#print(visual.shape)

		# Extraction
		#mixture audio
		mixture_w = self.encoder(mixture)
		est_mask = self.separator(mixture_w, visual)

		est_source = self.decoder(mixture_w, est_mask)
		est_source = F.pad(est_source, (0, mixture.size(-1) - est_source.size(-1)))
		return est_source