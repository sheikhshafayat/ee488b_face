import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0)

	lr_step = 'epoch'

	print('Initialised step Cosine scheduler')

	return sche_fn, lr_step