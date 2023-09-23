
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from config import cfg

class HungarianMatcher(nn.Module):

    def __init__(self, cost_label: float = 1):

        super().__init__()
        self.cost_class = cost_label

        self.thresh_line_pos = np.cos(np.radians(88.0), dtype=np.float32) # near 0.0
        self.thresh_line_neg = np.cos(np.radians(85.0), dtype=np.float32)
        

    def line_label(self, outputs, targets):
        src_logits = outputs['pred_vp1_logits']
        target_lines = torch.stack([t['lines'] for t in targets], dim=0)
        target_mask = torch.stack([t['line_mask'] for t in targets], dim=0)
        target_vp1 = torch.stack([t['vp1'] for t in targets], dim=0) # [bs, 3]
        target_vp2 = torch.stack([t['vp2'] for t in targets], dim=0) # [bs, 3]
        target_vp3 = torch.stack([t['vp3'] for t in targets], dim=0) # [bs, 3]

        target_vp1 = target_vp1.unsqueeze(1)
        target_vp2 = target_vp2.unsqueeze(1)
        target_vp3 = target_vp3.unsqueeze(1)
        thresh_line_pos = np.cos(np.radians(88.0), dtype=np.float32)
        thresh_line_neg = np.cos(np.radians(85.0), dtype=np.float32)
        with torch.no_grad():

            cos_sim_zvp = F.cosine_similarity(target_lines, target_vp1, dim=-1).abs()
            cos_sim_hvp1 = F.cosine_similarity(target_lines, target_vp2, dim=-1).abs()
            cos_sim_hvp2 = F.cosine_similarity(target_lines, target_vp3, dim=-1).abs()
            
            

            cos_class_1 = cos_sim_zvp < thresh_line_pos
            cos_class_2 = cos_sim_hvp1 < thresh_line_pos
            cos_class_3 = cos_sim_hvp2 < thresh_line_pos
            ones = torch.ones_like(src_logits)
            zeros = torch.zeros_like(src_logits)
            mask_zvp = torch.where(torch.gt(cos_class_1.unsqueeze(-1), thresh_line_pos) &
                            torch.lt(cos_class_1.unsqueeze(-1), thresh_line_neg),  
                            zeros, ones)
            mask_hvp1 = torch.where(torch.gt(cos_class_2.unsqueeze(-1), thresh_line_pos) &
                            torch.lt(cos_class_2.unsqueeze(-1), thresh_line_neg),  
                            zeros, ones)
            mask_hvp2 = torch.where(torch.gt(cos_class_3.unsqueeze(-1), thresh_line_pos) &
                            torch.lt(cos_class_3.unsqueeze(-1), thresh_line_neg),  
                            zeros, ones)
            
            cos_sim = torch.where(cos_class_1, 1, 0) + torch.where(cos_class_2, 2, 0) + torch.where(cos_class_3, 3, 0)

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

        return cos_class_1.unsqueeze(-1), cos_class_2.unsqueeze(-1),cos_class_3.unsqueeze(-1),mask_zvp,mask_hvp1,mask_hvp2
    
    def loss_vp_labels(self,pred_weight, outputs, targets):
        # positive < thresh_pos < no label < thresh_neg < negative
        src_logits = pred_weight
        target_lines = torch.stack([t['lines'] for t in targets], dim=0) # [bs, n, 3]
        target_mask = torch.stack([t['line_mask'] for t in targets], dim=0) # [bs, n, 1]
        target_vp2 = torch.stack([t['vp2'] for t in targets], dim=0) # [bs, 3]
        target_vp2 = target_vp2.unsqueeze(1) # [bs, 1, 3]
        target_vp3 = torch.stack([t['vp3'] for t in targets], dim=0) # [bs, 3]
        target_vp3 = target_vp3.unsqueeze(1) # [bs, 1, 3]
        _,class_hvp1,class_hvp2,_,mask_hvp1,mask_hvp2 = self.line_label(outputs,targets)

        with torch.no_grad():

            target_classes1 = class_hvp1

            target_classes2 = class_hvp2
            
            
            # [bs, n, 1]            
            mask1 = target_mask*mask_hvp1
            mask2 = target_mask*mask_hvp2
            
        loss_ce1 = F.binary_cross_entropy_with_logits(
            src_logits, target_classes1, reduction='none')
        loss_ce1 = mask1*loss_ce1#*torch.where(mask2==1,0,1)
        loss_ce1 = loss_ce1.sum(dim=1)/mask1.sum(dim=1)
        
        loss_ce2 = F.binary_cross_entropy_with_logits(
            src_logits, target_classes2, reduction='none')
        loss_ce2 = mask2*loss_ce2#*torch.where(mask1==1,0,1)
        loss_ce2 = loss_ce2.sum(dim=1)/mask2.sum(dim=1)
        losses = torch.cat([loss_ce1,loss_ce2],dim=1)
        
        return losses
        
    

    @torch.no_grad()
    def forward(self, outputs, targets):

        

        # We flatten to compute the cost matrices in a batch
        # out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        
        pred_v2weight = outputs['pred_vp2_logits']
        pred_v3weight = outputs['pred_vp3_logits']
        
        # print(pred_vp1.shape)
        
        # Also concat the target labels and boxes
        tgt_vp1 = torch.stack([t['vp1'] for t in targets], dim=0)
        tgt_vp2 = torch.stack([t['vp2'] for t in targets], dim=0)
        tgt_vp3 = torch.stack([t['vp3'] for t in targets], dim=0)
        #print("tgt_vp1",tgt_vp1.shape)
        #print(tgt_vp1)
        #print("pred_vp:",pred_vp.shape)
        # Also concat the target labels and boxes
        tgt_vp = torch.cat([v["vp"] for v in targets])
        
        target_hvps1 = torch.stack(
            [t["horizon_lines1"] for t in targets], dim=0
        )
        target_hvps2 = torch.stack(
            [t["horizon_lines2"] for t in targets], dim=0
        )
        target_hvps = torch.cat([target_hvps1.unsqueeze(1),target_hvps2.unsqueeze(1)],dim=1)

        bs, num_queries = tgt_vp1.shape[:2]
        # print("bs",bs)
        # print("num_queries",num_queries)
        #print("pred_vp.shape:",pred_vp.shape)
        #print("tgt_vp.shape",tgt_vp.shape)


        # Compute the L1 cost between boxes
        cost_vp1 = self.loss_vp_labels(pred_v2weight,outputs,targets)
        cost_vp2 = self.loss_vp_labels(pred_v3weight,outputs,targets)
        #print("label_matcher",cost_vp1.shape)
        # Final cost matrix
        C = torch.cat([cost_vp1.unsqueeze(1),cost_vp2.unsqueeze(1)],dim=1)
        # print(C)
        # print("label",C.shape)
        C = C.view(bs, 2, -1).cpu()

        
        #print(bs,num_queries)

        sizes = 6
        #print("sizes",sizes)
        indices = [linear_sum_assignment(c) for i, c in enumerate(C.split(2, -1)[0])]
        #print(indices)
        # for i,c in enumerate(C.split(2,-1)):
        #     print(i)
        #     print(i,c)
        #     print(i,c[0])
        #     print(c.shape)

        # print("indices",indices)
        # print(indices[0])
        #print([(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices])
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_label_matcher(cfg):
    return HungarianMatcher(cost_label=1)