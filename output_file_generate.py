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
from models import build_model
from config import cfg

cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

def c(x):
    return sm.to_rgba(x)

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

def compute_vp_err(vp1, vp2, dim=-1):
    cos_sim = F.cosine_similarity(vp1, vp2, dim=dim).abs()
    cos_sim = np.clip(cos_sim.item(), 0.0, 1.0)    
    return np.degrees(np.arccos(cos_sim))

def compute_hl_np(hl, sz, eps=1e-6):
    (a,b,c) = hl
    if b < 0:
        a, b, c = -a, -b, -c
    b = np.maximum(b, eps)
    
    left = np.array([-1.0, (a - c)/b])        
    right = np.array([1.0, (-a - c)/b])

    # scale back to original image    
    scale = sz[1]/2
    left = scale*left
    right = scale*right
    return [np.squeeze(left), np.squeeze(right)]

def compute_up_vector(zvp, fovy, eps=1e-7):
    # image size 2 (-1~1)
    focal = 1.0/np.tan(fovy/2.0)
    
    if zvp[2] < 0:
        zvp = -zvp
    zvp = zvp / np.maximum(zvp[2], eps)
    zvp[2] = focal
    return normalize_safe_np(zvp)

def decompose_up_vector(v):
    pitch = np.arcsin(v[2])
    roll = np.arctan(-v[0]/v[1])
    return pitch, roll

def cosine_similarity(v1, v2, eps=1e-7):
    v1 = v1 / np.maximum(LA.norm(v1), eps)
    v2 = v2 / np.maximum(LA.norm(v2), eps)
    return np.sum(v1*v2)

def normalize_safe_np(v, eps=1e-7):
    return v/np.maximum(LA.norm(v), eps)

def compute_up_vector_error(pred_zvp, pred_fovy, target_up_vector):
    pred_up_vector = compute_up_vector(pred_zvp, pred_fovy)
    cos_sim = cosine_similarity(target_up_vector, pred_up_vector)

    target_pitch, target_roll = decompose_up_vector(target_up_vector)

    if cos_sim < 0:
        pred_pitch, pred_roll = decompose_up_vector(-pred_up_vector)
    else:
        pred_pitch, pred_roll = decompose_up_vector(pred_up_vector)

    err_angle = np.degrees(np.arccos(np.abs(cos_sim)))
    err_pitch = np.degrees(np.abs(pred_pitch - target_pitch))
    err_roll = np.degrees(np.abs(pred_roll - target_roll))
    return err_angle, err_pitch, err_roll

def compute_fovy_error(pred_fovy, target_fovy):
    pred_fovy = np.degrees(pred_fovy)
    target_fovy = np.degrees(target_fovy)
    err_fovy = np.abs(pred_fovy - target_fovy)
    return err_fovy.item()

def compute_horizon_error(pred_hl, target_hl, img_sz):
    target_hl_pts = compute_hl_np(target_hl, img_sz)
    pred_hl_pts = compute_hl_np(pred_hl, img_sz)
    err_hl = np.maximum(np.abs(target_hl_pts[0][1] - pred_hl_pts[0][1]),
                        np.abs(target_hl_pts[1][1] - pred_hl_pts[1][1]))
    err_hl /= img_sz[0] # height
    return err_hl
    
def draw_attention(img, weights, cmap, savepath):
    extent = [-1, 1, 1, -1]
    num_layer = len(weights)
    plt.figure(figsize=(num_layer*3,3))
    for idx_l in range(num_layer):                    
        plt.subplot(1, num_layer, idx_l + 1)
        plt.imshow(img, extent=extent)
        plt.imshow(weights[idx_l], cmap=cmap, alpha=0.3, extent=extent)
        plt.axis('off')
    plt.savefig(savepath, pad_inches=0, bbox_inches='tight')
    plt.close('all')

def draw_attention_segs(img, weights, segs, cmap, savepath):
    num_layer = len(weights)
    num_segs = len(segs)
    plt.figure(figsize=(num_layer*3,3))
    for idx_l in range(num_layer):                    
        plt.subplot(1, num_layer, idx_l + 1)
        plt.imshow(img, extent=[-1, 1, 1, -1])                 
        ws = weights[idx_l]
        ws = (ws - ws.min())/(ws.max() - ws.min())
        for idx_s in range(num_segs):
            plt.plot((segs[idx_s,0], segs[idx_s,2]), 
                     (segs[idx_s,1], segs[idx_s,3]), c=cmap(ws[idx_s]))
        plt.axis('off')
    plt.savefig(savepath, pad_inches=0, bbox_inches='tight')
    plt.close('all')    
    
def to_device(data, device):
    if type(data) == dict:
        return {k: v.to(device) for k, v in data.items()}
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()} for t in data]    

def main(cfg):
    device = torch.device(cfg.DEVICE)
    
    model, _ = build_model(cfg)
    model.to(device)
    
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
    
    dict = {'pred':{"pred_vp1":[],"pred_vp2":[],"pred_vp3":[]},'gt':{"target_vp1":[],"target_vp2":[],"target_vp3":[]}}
        
    for i, (samples, extra_samples, targets) in enumerate(tqdm(data_loader_test)):
        with torch.no_grad():
            samples = samples.to(device)
            extra_samples = to_device(extra_samples, device)
            outputs, extra_info = model(samples, extra_samples)
            img = targets[0]['org_img']
        
            pred_vp1 = outputs['pred_vp1'].to('cpu')[0].numpy()
            pred_vp2 = outputs['pred_vp2'].to('cpu')[0].numpy()
            pred_vp3 = outputs['pred_vp3'].to('cpu')[0].numpy()
            
            pred_v1weight = outputs['pred_vp1_logits'].sigmoid()
            pred_v1weight = pred_v1weight.to('cpu')[0].numpy()
            pred_v1weight = np.squeeze(pred_v1weight, axis=1)

            pred_v2weight = outputs['pred_vp2_logits'].sigmoid()
            pred_v2weight = pred_v2weight.to('cpu')[0].numpy()
            pred_v2weight = np.squeeze(pred_v2weight, axis=1)

            pred_v3weight = outputs['pred_vp3_logits'].sigmoid()
            pred_v3weight = pred_v3weight.to('cpu')[0].numpy()
            pred_v3weight = np.squeeze(pred_v3weight, axis=1)
            
            aux_outputs = outputs['aux_outputs']
                        
            img_sz = targets[0]['org_sz']
            filename = targets[0]['filename']
            filename = osp.splitext(filename)[0]
                
            target_vp1 = targets[0]['vp1'].numpy()
            target_vp2 = targets[0]['vp2'].numpy()
            target_vp3 = targets[0]['vp3'].numpy()
            
            # target_vp1 = F.normalize(torch.tensor(target_vp1), p=2, dim=-1)
            # target_vp2 = F.normalize(torch.tensor(target_vp2), p=2, dim=-1)
            # target_vp3 = F.normalize(torch.tensor(target_vp3), p=2, dim=-1)
            
            pred_vp1 = torch.tensor(pred_vp1)
            pred_vp2 = torch.tensor(pred_vp2)
            pred_vp3 = torch.tensor(pred_vp3)
            
            tgt_cross = torch.cross(target_vp3,target_vp1)
            tgt_dot = torch.dot(target_vp2,tgt_cross)
            if cfg.TEST.DISPLAY:
                dict['pred']['pred_vp1'].append(pred_vp1.tolist())
                dict['pred']['pred_vp2'].append(pred_vp2.tolist())
                dict['pred']['pred_vp3'].append(pred_vp3.tolist())
                dict['gt']['target_vp1'].append(target_vp1.tolist())
                dict['gt']['target_vp2'].append(target_vp2.tolist())
                dict['gt']['target_vp3'].append(target_vp3.tolist())
            
    print("json 파일 저장중")
    with open('z=1_non.json','w') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)
    print('json 파일 저장 완료')









if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPANet training and evaluation script', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main(cfg)