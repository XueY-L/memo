'''
CUDA_VISIBLE_DEVICES=0 python3 test_adapt.py --dataroot /home/yxue/datasets --level 5  --resume None --optimizer sgd --lr 0.00025 --weight_decay 0.0
'''
from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import sys 
sys.path.append('/home/yxue/memo/imagenet-exps')

from tqdm import tqdm
from utils.adapt_helpers import adapt_single, test_single
from utils.train_helpers import build_model, prepare_test_data

from robustbench.data import load_imagenetc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True)
parser.add_argument('--use_rvt', action='store_true')
parser.add_argument('--use_resnext', action='store_true')
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--resume', required=True)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--prior_strength', default=16, type=int)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--lr', default=0.00025, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--niter', default=1, type=int)
args = parser.parse_args()

# net = build_model(args)
base_model = load_model('Standard_R50', './ckpt', 'imagenet', ThreatModel.corruptions).cuda()
base_model.load_state_dict(torch.load('/home/yxue/model_fusion_tta/imagenet/checkpoint/ckpt_[\'gaussian_noise\']_[5].pt')['model'])


def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

if args.optimizer == 'sgd':
    optimizer = optim.SGD(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    optimizer = optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


for corrupt in ['shot_noise', 'impulse_noise', 'defocus_blur', 'motion_blur', 'zoom_blur', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate']:
    teset = load_imagenetc(5000, args.level, args.dataroot, False, [corrupt])
    correct = []
    for image, label, _ in teset:
        adapt_single(base_model, image, optimizer, marginal_entropy,
                    args.corruption, args.niter, args.batch_size, args.prior_strength)
        correct.append(test_single(base_model, image, label, args.corruption, args.prior_strength)[0])
    print(f'MEMO adapt test error {(1-np.mean(correct))*100:.2f}')
    