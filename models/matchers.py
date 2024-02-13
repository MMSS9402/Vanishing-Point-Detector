# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from config import cfg

class HungarianMatcher(nn.Module):

    def __init__(self, cost_vp: float = 1):

        super().__init__()
        self.cost_class = cost_vp

        self.thresh_line_pos = np.cos(np.radians(88.0), dtype=np.float32) # near 0.0
        self.thresh_line_neg = np.cos(np.radians(85.0), dtype=np.float32)
        
    def loss_vp(self, outputs, targets, **kwargs):
        src_zvp = outputs                                   
        target_zvp = targets

        cos_sim = F.cosine_similarity(src_zvp, target_zvp, dim=-1).abs()   

        loss_vp_cos = (1.0 - cos_sim)
            
        return loss_vp_cos
    

    @torch.no_grad()
    def forward(self, outputs, targets):
        
        pred_vp1 = outputs['pred_vp1']
        pred_vp2 = outputs['pred_vp2']
        pred_vp3 = outputs['pred_vp3']

        tgt_vp1 = torch.cat([v["vp1"] for v in targets])
        tgt_vp2 = torch.cat([v["vp2"] for v in targets])
        tgt_vp3 = torch.cat([v["vp3"] for v in targets])

        pred_vp = torch.cat([torch.cat([pred_vp1.unsqueeze(1),pred_vp2.unsqueeze(1)],dim=1),pred_vp3.unsqueeze(1)],dim=1)

        tgt_vp = torch.cat([v["vp"] for v in targets])

        bs, num_queries = pred_vp.shape[:2]
        
        cost_vp1 = self.loss_vp(pred_vp1.unsqueeze(1),tgt_vp)
        cost_vp2 = self.loss_vp(pred_vp2.unsqueeze(1),tgt_vp)
        cost_vp3 = self.loss_vp(pred_vp3.unsqueeze(1),tgt_vp)
        
        C = torch.cat([torch.cat([cost_vp1.unsqueeze(1),cost_vp2.unsqueeze(1)],dim=1),cost_vp3.unsqueeze(1)],dim=1)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = 6
        indices = [linear_sum_assignment(c) for i, c in enumerate(C.split(3, -1)[0])]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(cfg):
    return HungarianMatcher(cost_vp=1)