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
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader



import datasets
import util.misc as utils
from datasets import build_matterport_dataset
from models import build_model
from config import cfg
from models.matchers import build_matcher

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

def line_label( pred_v1weight,target_vp1,target_vp2,target_vp3,target_lines,target_mask):
    src_logits = torch.tensor(pred_v1weight)
    src_logits = src_logits.unsqueeze(-1)
    target_lines = torch.tensor(target_lines)
    # target_mask =torch.tensor(target_mask)
    target_vp1 = target_vp1 # [bs, 3]
    target_vp2 = target_vp2 # [bs, 3]
    target_vp3 = target_vp3 # [bs, 3]

    target_vp1 = target_vp1.unsqueeze(0)
    target_vp2 = target_vp2.unsqueeze(0)
    target_vp3 = target_vp3.unsqueeze(0)
    thresh_line_pos = np.cos(np.radians(88.0), dtype=np.float32)
    thresh_line_neg = np.cos(np.radians(85.0), dtype=np.float32)
    with torch.no_grad():

        cos_sim_zvp = F.cosine_similarity(target_lines, target_vp1, dim=-1).abs()
        cos_sim_hvp1 = F.cosine_similarity(target_lines, target_vp2, dim=-1).abs()
        cos_sim_hvp2 = F.cosine_similarity(target_lines, target_vp3, dim=-1).abs()
        cos_sim_zvp = cos_sim_zvp.unsqueeze(-1)
        cos_sim_hvp1 = cos_sim_hvp1.unsqueeze(-1)
        cos_sim_hvp2 = cos_sim_hvp2.unsqueeze(-1)
        ones = torch.ones_like(src_logits)
        zeros = torch.zeros_like(src_logits)

        cos_class_1 = torch.where(cos_sim_zvp < thresh_line_pos, ones, zeros)
        cos_class_2 = torch.where(cos_sim_hvp1 < thresh_line_pos, ones, zeros)
        cos_class_3 = torch.where(cos_sim_hvp2 < thresh_line_pos, ones, zeros)


        mask_zvp = torch.where(torch.gt(cos_class_1, thresh_line_pos) &
                        torch.lt(cos_class_1, thresh_line_neg),  
                        zeros, ones)
        mask_hvp1 = torch.where(torch.gt(cos_class_2, thresh_line_pos) &
                        torch.lt(cos_class_2, thresh_line_neg),  
                        zeros, ones)
        mask_hvp2 = torch.where(torch.gt(cos_class_3, thresh_line_pos) &
                        torch.lt(cos_class_3, thresh_line_neg),  
                        zeros, ones)
        
        cos_sim = torch.where(cos_class_1==1, 1, 0) + torch.where(cos_class_2==1, 2, 0) + torch.where(cos_class_3==1, 3, 0)

        overlaps = cos_sim >= 4
        
        if overlaps.any():
            values, indices = torch.stack([cos_sim_zvp, cos_sim_hvp1, cos_sim_hvp2], dim=-1)[overlaps].min(dim=-1)
        
            cos_sim[overlaps] = indices + 1

        cos_class_1 = cos_sim == 1
        cos_class_2 = cos_sim == 2
        cos_class_3 = cos_sim == 3
        
        cos_class_1 = cos_class_1.float()
        cos_class_2 = cos_class_2.float()
        cos_class_3 = cos_class_3.float()

    return cos_class_1, cos_class_2,cos_class_3,mask_zvp,mask_hvp1,mask_hvp2

def loss_vp_line(pred_v1weight,target_vp1,target_lines,target_mask):
    src_logits = torch.tensor(pred_v1weight)      
    target_lines = torch.tensor(target_lines)  
    # target_mask = torch.tensor(target_mask)  
    target_zvp = torch.tensor(target_vp1)  # [bs, 3]
    target_zvp = target_zvp.unsqueeze(0) # [bs, 1, 3]
    thresh_line_pos = np.cos(np.radians(88.0), dtype=np.float32)
    thresh_line_neg = np.cos(np.radians(85.0), dtype=np.float32)
    
    with torch.no_grad():

        cos_sim = F.cosine_similarity(target_lines, target_zvp, dim=-1).abs()
        # [bs, n]
        cos_sim = cos_sim.unsqueeze(-1) # [bs, n, 1]
        src_logits = src_logits.unsqueeze(-1)
        # print("thresh_line_pos",1-thresh_line_pos)
        ones = torch.ones_like(src_logits)
        zeros = torch.zeros_like(src_logits)
        target_classes = torch.where(cos_sim < thresh_line_pos, ones, zeros)
        # mask = torch.where(torch.gt(cos_sim, thresh_line_pos) &
        #                     torch.lt(cos_sim, thresh_line_neg),  
        #                     zeros, ones)    
        # mask = target_mask#*mask
        mask = target_classes

    return mask.float() 

def calculate_accuracy(y_true, y_pred):

    y_pred_binary = (y_pred > 0.8).astype(np.float32)

    y_true = np.squeeze(y_true, axis=1)

    correct =0
    for i in range(y_true.shape[0]):
        if y_pred_binary[i] == y_true[i]:
            correct += 1

    accuracy = correct / y_true.shape[0]
    
    return accuracy*100

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
    cos_sim = F.cosine_similarity(pred_vp[src_idx], tgt_vp[tgt_idx], dim=-1).abs()    
    loss_vp1_cos = (1.0 - cos_sim).mean()
    return loss_vp1_cos 

def line_label(outputs, targets):
    src_logits = outputs['pred_vp1_logits']
    target_lines = torch.stack([t['lines'] for t in targets], dim=0)
    # target_mask = torch.stack([t['line_mask'] for t in targets], dim=0)
    target_vp1 = torch.stack([t['vp1'] for t in targets], dim=0) # [bs, 3]
    target_vp2 = torch.stack([t['vp2'] for t in targets], dim=0) # [bs, 3]
    target_vp3 = torch.stack([t['vp3'] for t in targets], dim=0) # [bs, 3]

    target_vp1 = target_vp1.unsqueeze(1)
    target_vp2 = target_vp2.unsqueeze(1)
    target_vp3 = target_vp3.unsqueeze(1)

    with torch.no_grad():
        
        cos_sim_zvp = F.cosine_similarity(target_lines, target_vp1, dim=-1).abs()
        cos_sim_hvp1 = F.cosine_similarity(target_lines, target_vp2, dim=-1).abs()
        cos_sim_hvp2 = F.cosine_similarity(target_lines, target_vp3, dim=-1).abs()
        cos_sim_zvp = cos_sim_zvp.unsqueeze(-1)
        cos_sim_hvp1 = cos_sim_hvp1.unsqueeze(-1)
        cos_sim_hvp2 = cos_sim_hvp2.unsqueeze(-1)
        thresh_line_pos = np.cos(np.radians(88.0), dtype=np.float32)
        thresh_line_neg = np.cos(np.radians(85.0), dtype=np.float32)
        
        ones = torch.ones_like(src_logits)
        zeros = torch.zeros_like(src_logits)

        cos_class_1 = torch.where(cos_sim_zvp < thresh_line_pos, ones, zeros)
        cos_class_2 = torch.where(cos_sim_hvp1 < thresh_line_pos, ones, zeros)
        cos_class_3 = torch.where(cos_sim_hvp2 < thresh_line_pos, ones, zeros)
        
        mask_zvp = torch.where(torch.gt(cos_class_1, thresh_line_pos) &
                        torch.lt(cos_class_1, thresh_line_neg),  
                        zeros, ones)
        mask_hvp1 = torch.where(torch.gt(cos_class_2, thresh_line_pos) &
                        torch.lt(cos_class_2, thresh_line_neg),  
                        zeros, ones)
        mask_hvp2 = torch.where(torch.gt(cos_class_3, thresh_line_pos) &
                        torch.lt(cos_class_3, thresh_line_neg),  
                        zeros, ones)
        cos_sim = torch.where(cos_class_1==1, 1, 0) + torch.where(cos_class_2==1, 2, 0) + torch.where(cos_class_3==1, 3, 0)

        overlaps = cos_sim >= 4
        
        if overlaps.any():
            values, indices = torch.stack([cos_sim_zvp, cos_sim_hvp1, cos_sim_hvp2], dim=-1)[overlaps].min(dim=-1)
        
            cos_sim[overlaps] = indices + 1

        cos_class_1 = cos_sim == 1
        cos_class_2 = cos_sim == 2
        cos_class_3 = cos_sim == 3
        
        cos_class_1 = cos_class_1.float()
        cos_class_2 = cos_class_2.float()
        cos_class_3 = cos_class_3.float()

    return cos_class_1, cos_class_2,cos_class_3,mask_zvp,mask_hvp1,mask_hvp2


def loss_vp1_labels(outputs, targets, indices, **kwargs):
        
    src_logits1 = outputs["pred_vp1_logits"]
    src_logits2 = outputs["pred_vp2_logits"]
    src_logits3 = outputs["pred_vp3_logits"]
    src_logits = torch.cat([src_logits1.unsqueeze(1),src_logits2.unsqueeze(1),src_logits3.unsqueeze(1)],dim=1)

    
    class_zvp,class_hvp1,class_hvp2,mask_zvp,mask_hvp1,mask_hvp2 = line_label(outputs,targets)
    class_vp = torch.cat([class_zvp.unsqueeze(1),class_hvp1.unsqueeze(1),class_hvp2.unsqueeze(1)],dim=1)
    mask_vp = torch.cat([mask_zvp.unsqueeze(1),mask_hvp1.unsqueeze(1),mask_hvp2.unsqueeze(1)],dim=1)
    
    src_idx = _get_src_permutation_idx(indices)
    tgt_idx = _get_tgt_permutation_idx(indices)
    with torch.no_grad():

        target_classes = class_vp[tgt_idx]

        mask = mask_vp[tgt_idx]
    
    loss_ce = F.binary_cross_entropy_with_logits(
        src_logits[src_idx], target_classes, reduction='none')
    loss_ce = mask*loss_ce
    loss_ce = loss_ce.sum(dim=1)/mask.sum(dim=1)
    
    return loss_ce.mean()


def main(cfg):
    warnings.filterwarnings('ignore')
    device = torch.device(cfg.DEVICE)
    
    model, _ = build_model(cfg)
    matcher = build_matcher(cfg)
    model.to(device)
    
    dataset_test = build_matterport_dataset(image_set='test', cfg=cfg)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test,
                                 drop_last=False, 
                                 collate_fn=utils.collate_fn, 
                                 num_workers=2)
    
    output_dir = Path(cfg.OUTPUT_DIR)
    
    checkpoint = torch.load('/home/kmuvcl/CTRL-C/transformer_encoder/checkpoint0051.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    
    # initlaize for visualization
    name = f'gsv_test_{date.today()}'
    if cfg.TEST.DISPLAY:
        fig_output_dir = osp.join(output_dir,'{}'.format(name))
        os.makedirs(fig_output_dir, exist_ok=True)
    
    dict = {'vp_dot_product':[]}
    
    intrinsics = torch.tensor([[517.97,0,0],
                [0,517.97,0],
                [0,0,1]])
    loss_vp = []
    loss_vp1_label = []
    acc_zvp = []
    acc_hvp1 = []
    acc_hvp2 = []
    filename2 = []
    intrinsics = torch.tensor(intrinsics).float()
    for i, (samples, extra_samples, targets) in enumerate(tqdm(data_loader_test)):
        with torch.no_grad():

        
            samples = samples.to(device)
            extra_samples = to_device(extra_samples, device)
            outputs, extra_info = model(samples, extra_samples)
            img = targets[0]['org_img']
        
            pred_vp1 = outputs['pred_vp1'].to('cpu')[0].numpy()
            pred_vp2 = outputs['pred_vp2'].to('cpu')[0].numpy()
            pred_vp3 = outputs['pred_vp3'].to('cpu')[0].numpy()
        
            # print(pred_vp1,pred_vp2,pred_vp3)
            
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
            filename2.append(filename)

            #print(filename)
                
            target_vp1 = targets[0]['vp1'].numpy()
            target_vp2 = targets[0]['vp2'].numpy()
            target_vp3 = targets[0]['vp3'].numpy()
            # print("before target_vp2",target_vp2)
            target_vp1 = F.normalize(torch.tensor(target_vp1), p=2, dim=-1)
            target_vp2 = F.normalize(torch.tensor(target_vp2), p=2, dim=-1)
            target_vp3 = F.normalize(torch.tensor(target_vp3), p=2, dim=-1)
            # print("target_vp2",target_vp2)
            pred_vp1 = torch.tensor(pred_vp1)
            pred_vp2 = torch.tensor(pred_vp2)
            pred_vp3 = torch.tensor(pred_vp3)
            
            tgt_cross = torch.cross(target_vp3,target_vp1)
            tgt_dot = torch.dot(target_vp2,tgt_cross)
            
            

            if pred_vp1[2] < 0:
                pred_vp1 = -pred_vp1
            if pred_vp2[2] < 0:
                pred_vp2 = -pred_vp2
            if pred_vp3[2] < 0:
                pred_vp3 = -pred_vp3 
            
            
            
            target_segs = targets[0]['segs'].numpy()
            target_lines = targets[0]['lines'].numpy()
            # target_mask = targets[0]['line_mask'].numpy()
            # target_straight_seg = targets[0]['straight_segs'].numpy()
            num_segs = len(target_segs)
            for key in outputs:
                if isinstance(outputs[key], torch.Tensor):
                    outputs[key] = outputs[key].cpu()

            # targets = targets[0]
            # for key in targets:
            #     if isinstance(targets[key], torch.Tensor):
            #         targets[key] = targets[key].cpu()
                    

            indices = matcher(outputs, targets)

            loss_vp.append(loss_vp1(outputs,targets,indices).tolist())
            loss_vp1_label.append(loss_vp1_labels(outputs,targets,indices).tolist())

            
    idx = loss_vp.index(min(loss_vp))
    print("min(loss_vp)",min(loss_vp))
    loss_vp = sum(loss_vp)/len(loss_vp)

    idx = loss_vp1_label.index(min(loss_vp1_label))
    print("min(loss_vp1_label)",min(loss_vp1_label))
    loss_vp1_label = sum(loss_vp1_label)/len(loss_vp1_label)

    print("loss_vp",loss_vp)
    print("loss_vp1_label",loss_vp1_label)
                
  

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPANet training and evaluation script', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main(cfg)
