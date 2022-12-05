#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=13, gamma=lr_decay)

	lr_step = 'epoch'

	print('Initialised step LR scheduler')

	return sche_fn, lr_step
