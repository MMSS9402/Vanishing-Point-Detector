import os
import os.path as osp
import argparse
from datetime import date
import json
import random
import time
from pathlib import Path
import numpy as np
import numpy.linalg as LA
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets
import util.misc as utils
from datasets import build_matterport_dataset
from models.matchers import build_matcher
from models import build_model
from config import cfg

cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

def c(x):
    return sm.to_rgba(x)

def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


def loss_vp1(outputs, targets,indices, **kwargs):
        #print(outputs.keys())
        assert 'pred_vp1' in outputs
        assert 'pred_vp2' in outputs
        assert 'pred_vp3' in outputs
        pred_vp1 = outputs['pred_vp1']
        pred_vp2 = outputs['pred_vp2']
        pred_vp3 = outputs['pred_vp3']
        pred_vp = torch.cat([torch.cat([pred_vp1.unsqueeze(1),pred_vp2.unsqueeze(1)],dim=1),pred_vp3.unsqueeze(1)],dim=1)
        tgt_vp = torch.cat([v["vp"] for v in targets])
        src_idx = _get_src_permutation_idx(indices)
        tgt_idx = _get_tgt_permutation_idx(indices)
        # print('src_idx & tgt_idx',src_idx,tgt_idx)
        #flag = self.calculate_index(outputs,targets)
        # if flag == 0 :                                    
        #     target_zvp = torch.stack([t['vp1'] for t in targets], dim=0)
        # if flag == 1 :
        #     target_zvp = torch.stack([t['vp3'] for t in targets], dim=0)
        #target_zvp = torch.stack([t['vp1'] for t in targets], dim=0)

        cos_sim = F.cosine_similarity(pred_vp[src_idx], tgt_vp[tgt_idx], dim=-1).abs()      
        loss_vp1_cos = (1.0 - cos_sim).mean()
                
        return loss_vp1_cos

def get_args_parser():
    parser = argparse.ArgumentParser('Set gptran', add_help=False)
    parser.add_argument('--config-file', 
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default='/home/kmuvcl/source/CTRL-C/config-files/ctrl-c.yaml')
    parser.add_argument("--opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )
    return parser


    
def to_device(data, device):
    if type(data) == dict:
        return {k: v.to(device) for k, v in data.items()}
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()} for t in data]    

def main(cfg):
    device = torch.device(cfg.DEVICE)
    
    model, _ = build_model(cfg)
    model.to(device)
    matcher = build_matcher(cfg)
    
    dataset_test = build_matterport_dataset(image_set='test', cfg=cfg)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test,
                                 drop_last=False, 
                                 collate_fn=utils.collate_fn, 
                                 num_workers=2)
    
    output_dir = Path(cfg.OUTPUT_DIR)
    
    checkpoint = torch.load('/home/kmuvcl/source/CTRL-C/hungraian_class2/checkpoint0099.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    
    # initlaize for visualization
    name = f'gsv_test_{date.today()}'
    if cfg.TEST.DISPLAY:
        fig_output_dir = osp.join(output_dir,'{}'.format(name))
        os.makedirs(fig_output_dir, exist_ok=True)
    
    dict = {'filename':[],'vp1_loss':[]}
        
    for i, (samples, extra_samples, targets) in enumerate(tqdm(data_loader_test)):
        with torch.no_grad():
            samples = samples.to(device)
            targets = to_device(targets, device)
            extra_samples = to_device(extra_samples, device)
            outputs, extra_info = model(samples, extra_samples)
            indices = matcher(outputs,targets)
            dict['vp1_loss'].append(loss_vp1(outputs,targets,indices).tolist())
            print(dict['vp1_loss'])
            #filename = targets['filename'][i]
            #dict['filename'].append(filename)
    with open('vp_loss.json','w') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)

    print('max_loss',max(dict['vp1_loss']))
    print('min_loss',min(dict['vp1_loss']))
    print('avg_loss',sum(dict['vp1_loss'])/len(dict['vp1_loss']))
        


            

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPANet training and evaluation script', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main(cfg)
