import os
import os.path as osp
import glob
import random

import skimage.io
import skimage.transform
from scipy import io

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import functional as F
import torch.nn.functional as TF
import torchvision.transforms as transforms

import numpy as np
import numpy.linalg as LA

import cv2
import json
import csv
import matplotlib.pyplot as plt
import h5py
from einops import rearrange

from datasets import transforms as T

# 오일러 각(3개의 방위) => rotation matrix로 바꾸자
def eul2rotm_ypr(euler):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(euler[0]), -np.sin(euler[0])],
            [0, np.sin(euler[0]), np.cos(euler[0])],
        ],
        dtype=np.float32,
    )

    R_y = np.array(
        [
            [np.cos(euler[1]), 0, np.sin(euler[1])],
            [0, 1, 0],
            [-np.sin(euler[1]), 0, np.cos(euler[1])],
        ],
        dtype=np.float32,
    )

    R_z = np.array(
        [
            [np.cos(euler[2]), -np.sin(euler[2]), 0],
            [np.sin(euler[2]), np.cos(euler[2]), 0],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    return np.dot(R_z, np.dot(R_x, R_y))


def create_masks(image):
    masks = torch.zeros((1, height, width), dtype=torch.uint8)
    return masks


def read_line_file(filename, min_line_length=10):
    segs = []  # line segments
    # csv 파일 열어서 Line 정보 가져오기
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            segs.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    segs = np.array(segs, dtype=np.float32)
    lengths = LA.norm(segs[:, 2:] - segs[:, :2], axis=1)
    segs = segs[lengths > min_line_length]
    return segs

def coordinate_yup(segs,org_h,org_w):
    H = np.array([0,org_h,0,org_h])
    segs[:,1] = -segs[:,1]
    segs[:,3] = -segs[:,3]
    # segs = -segs
    return(H+segs)
    

def normalize_segs(segs, pp, rho):
    pp = np.array([pp[0], pp[1], pp[0], pp[1]], dtype=np.float32)
    return (segs - pp)/rho

def focal_length_normalize(segs):
    segs = segs/517.97
    return segs


def normalize_safe_np(v, axis=-1, eps=1e-6):
    de = LA.norm(v, axis=axis, keepdims=True)
    de = np.maximum(de, eps)
    return v / de


def segs2lines_np(segs):
    ones = np.ones(len(segs))
    ones = np.expand_dims(ones, axis=-1)
    p1 = np.concatenate([segs[:, :2], ones], axis=-1)
    p2 = np.concatenate([segs[:, 2:], ones], axis=-1)

    lines = np.cross(p1, p2)

    return normalize_safe_np(lines)



def sample_segs_np(segs, num_sample, use_prob=True):
    num_segs = len(segs)
    sampled_segs = np.zeros([num_sample, 4], dtype=np.float32)
    mask = np.zeros([num_sample, 1], dtype=np.float32)
    if num_sample > num_segs:
        sampled_segs[:num_segs] = segs
        mask[:num_segs] = np.ones([num_segs, 1], dtype=np.float32)
    # else:
    #     sampled_segs = segs[:num_sample]
    #     mask[:num_segs] = np.ones([num_sample, 1], dtype=np.float32)
    else:
        lengths = LA.norm(segs[:, 2:] - segs[:, :2], axis=-1)
        prob = lengths / np.sum(lengths)
        idxs = np.random.choice(segs.shape[0], num_sample, replace=True, p=prob)
        sampled_segs = segs[idxs]
        mask = np.ones([num_sample, 1], dtype=np.float32)
    return sampled_segs, mask

# def segs_np(segs, num_sample, use_prob=True):
#     num_segs = len(segs)
#     sampled_segs = np.zeros([num_sample, 4], dtype=np.float32)
#     mask = np.zeros([num_sample, 1], dtype=np.float32)
#     if num_sample > num_segs:
#         sampled_segs[:num_segs] = segs
#         mask[:num_segs] = np.ones([num_segs, 1], dtype=np.float32)
#     # else:
#     #     sampled_segs = segs[:num_sample]
#     #     mask[:num_segs] = np.ones([num_sample, 1], dtype=np.float32)
#     else:
#         lengths = LA.norm(segs[:, 2:] - segs[:, :2], axis=-1)
#         prob = lengths / np.sum(lengths)
#         idxs = np.random.choice(segs.shape[0], num_sample, replace=True, p=prob)
#         sampled_segs = segs[idxs]
#         mask = np.ones([num_sample, 1], dtype=np.float32)
#     return sampled_segs, mask

def straight_line(segs):

    # for i in range(len(segs)):
    #     seg = np.vstack([segs[i,2]-segs[i,0],segs[i,3]-segs[i,1]])
    num_line = segs.shape[0]
    seg = np.zeros([num_line,2])
    for i in range(len(segs)):    
        seg[i,0] = segs[i,2]-segs[i,0]
        seg[i,1] = segs[i,3]-segs[i,1]
    return seg


def sample_vert_segs_np(segs, thresh_theta=22.5):
    lines = segs2lines_np(segs)
    (a, b) = lines[:, 0], lines[:, 1]
    theta = np.arctan2(np.abs(b), np.abs(a))
    thresh_theta = np.radians(thresh_theta)
    return segs[theta < thresh_theta]

def compute_vp_similarity(lines,vp):
    lines = torch.tensor(lines)
    vp = torch.tensor(vp)
    cos_sim = TF.cosine_similarity(lines, vp.unsqueeze(0), dim=-1).abs()
    
    return cos_sim

def load_h5py_to_dict(file_path):
        with h5py.File(file_path, 'r') as f:
            return {key: torch.tensor(f[key][:]) for key in f.keys()}


class GSVDataset(Dataset):
    def __init__(self, cfg, listpath, basepath, return_masks=False, transform=None):
        self.listpath = listpath
        self.basepath = basepath
        # self.input_width = cfg.DATASETS.INPUT_WIDTH
        # self.input_height = cfg.DATASETS.INPUT_HEIGHT
        self.min_line_length = cfg.DATASETS.MIN_LINE_LENGTH
        self.num_input_lines = cfg.DATASETS.NUM_INPUT_LINES
        self.num_input_vert_lines = cfg.DATASETS.NUM_INPUT_VERT_LINE
        self.vert_line_angle = cfg.DATASETS.VERT_LINE_ANGLE
        self.return_vert_lines = cfg.DATASETS.RETURN_VERT_LINES
        self.return_masks = return_masks
        self.transform = transform
        file_path = self.basepath+self.listpath
        # dirs = np.genfromtxt(file_path, dtype=str)
        filelist = glob.glob(f"{self.basepath}/*_0.png")
        filelist.sort()
        print("YUD_Dataset load")
        print("total number of samples", len(filelist))
        
        if self.listpath == "train":
            self.filelist = filelist[0:25]
            self.size = len(self.filelist) * 4
            print("subset for training: ", self.size)  
        if self.listpath == "val":
            self.filelist = filelist[25:102]
            self.size = len(self.filelist)
            print("subset for valid: ", self.size)
        if self.listpath == "test":
            self.filelist = filelist[25:102]
            self.size = len(self.filelist)
            print("subset for test: ", self.size)
        if self.listpath == "all":
            self.filelist = filelist
            self.size = len(self.filelist)
            print("all: ", len(self.filelist))
        
        camera_params = io.loadmat(os.path.join("/home/kmuvcl/dataset/data/processed_data/YorkUrbanDB", "cameraParameters.mat"))
        f = camera_params['focal'][0, 0]
        ps = camera_params['pixelSize'][0, 0]
        pp = camera_params['pp'][0, :]
        self.K = np.matrix([[f / ps, 0, pp[0]], [0, f / ps, pp[1]], [0, 0, 1]])
        self.S = np.matrix([[2.0 / 640, 0, -1], [0, 2.0 / 640, -0.75], [0, 0, 1]])
        invmat = np.linalg.inv(self.S * self.K)
        self.invmat = np.array(invmat)
    
                
    def __getitem__(self, idx):
        target = {}
        extra = {} 
        
        if self.listpath == "train":
            iname = self.filelist[idx // 4]
            if idx % 4 == 0: iname = iname
            if idx % 4 == 1: iname = iname.replace("_0", "_1")
            if idx % 4 == 2: iname = iname.replace("_0", "_2")
            if idx % 4 == 3: iname = iname.replace("_0", "_3")
        else:
            iname = self.filelist[idx]

        image = skimage.io.imread(iname)[:, :, 0:3]
        image = np.rollaxis(image, 2).copy().astype(float)
        
        with np.load(iname.replace(".png",".npz"), allow_pickle=True) as npz:
            vpts = npz["vpts"]
            org_segs = npz['line_segments']
            # vp_inds = npz['line_vp_inds']
        
        # elements , counts = np.unique(vp_inds,return_counts = True)
        # sorted_indices = np.argsort(-counts)
        # top_3_indices = elements[sorted_indices[:3]]
        # vpts = vpts[top_3_indices]
        # vpts = vpts[0:3]

        in_vpts = self.K*vpts

        num_vp = in_vpts.shape[1]
        tvp_list = []
        for vi in range(num_vp):
            in_vpts[:, vi] /= in_vpts[2, vi]

            tVP = np.array(in_vpts[:, vi])[:, 0]
            tVP /= tVP[2]

            tvp_list += [tVP]
        
        true_vps = np.vstack(tvp_list)
        vp = true_vps

        vp[:,0] /= vp[:,2]
        vp[:,1] /= vp[:,2]
        vp[:,2] /= vp[:,2]
        vps_pixel = vp[:, 0:2]
        vps_homo = np.concatenate((vps_pixel, np.ones((len(vps_pixel),1), dtype=np.float32)), axis=-1)
        vps_homo[:, 0] -= 320
        vps_homo[:, 1] -= 240
        vps_homo[:, 1] *= -1
        vps_homo[:, 0:2] /= 320.
        vps_homo /= np.linalg.norm(vps_homo, axis=1, keepdims=True)

        org_image = image
        org_h, org_w = image.shape[1], image.shape[2]
        org_sz = np.array([org_h, org_w])
        # image = cv2.resize(image, dsize=(org_w, org_h))
        input_sz = np.array([480, 640])

        pp = (org_w / 2, org_h / 2)
        # print("pp",pp)
        # print("vpts",vpts)
        # rho = 2.0 / np.minimum(org_w, org_h)
        focal_length = 1 #674.91797516

        num_segs = org_segs.shape[0]
        # org_segs = keylines
        # print(keylines.shape)
        # print("num_segs",num_segs)
        # try:
        #     org_segs = np.copy(keylines.reshape(num_segs,-1).numpy())
        # except RuntimeError:
        #     print(iname)
        #     print(num_segs)
        #     import pdb; pdb.set_trace
            

        # org_segs = coordinate_yup(org_segs,org_h,org_w)
        try:
            s = normalize_segs(org_segs, pp=pp, rho=focal_length)
        except RuntimeError:
            print(num_segs)
            print(iname)
        segs = normalize_segs(org_segs, pp=pp, rho=focal_length)


        sampled_segs, line_mask = sample_segs_np(segs, self.num_input_lines)
        sampled_lines = segs2lines_np(sampled_segs)

        non_sampled_lines = None
        # vertical directional segs 
        vert_segs = sample_vert_segs_np(segs, thresh_theta=self.vert_line_angle)
        if len(vert_segs) < 2:
            vert_segs = segs
        sampled_vert_segs, vert_line_mask = sample_segs_np(
            vert_segs, self.num_input_vert_lines
        )
        sampled_vert_lines = segs2lines_np(sampled_vert_segs)

        # gt_vp1 = vpts[0]
        # gt_vp2 = vpts[1]
        # gt_vp3 = vpts[2]

        gt_vp1 = vps_homo[0]
        gt_vp2 = vps_homo[1]
        gt_vp3 = vps_homo[2]
        
        # gt_vp1 = gt_vp1/gt_vp1[2]
        # gt_vp2 = gt_vp2/gt_vp2[2]
        # gt_vp3 = gt_vp3/gt_vp3[2]

        gt_vp = np.array([gt_vp1,gt_vp2,gt_vp3])
        gt_vp = np.expand_dims(gt_vp,axis=0)
        gt_hvps = np.array([gt_vp2,gt_vp3])
        gt_hvps[0, :] = gt_hvps[0, :]/gt_hvps[0,2]
        gt_hvps[1, :] = gt_hvps[1, :]/gt_hvps[1,2]
        gt_horizon_lines1 = gt_hvps[0, :]
        gt_horizon_lines2 = gt_hvps[1, :]
        
        image = torch.tensor(image)

            
        if self.return_masks:
            masks = create_masks(image)


        target["vp1"] = (
            torch.from_numpy(np.ascontiguousarray(gt_vp1)).contiguous().float()
        )
        target["vp2"] = (
            torch.from_numpy(np.ascontiguousarray(gt_vp2)).contiguous().float()
        )
        target["vp3"] = (
            torch.from_numpy(np.ascontiguousarray(gt_vp3)).contiguous().float()
        )
        target["vp"] = (
            torch.from_numpy(np.ascontiguousarray(gt_vp)).contiguous().float()
        )
        target["hvps"] = (
            torch.from_numpy(np.ascontiguousarray(gt_hvps)).contiguous().float()
        )
        

        if self.return_vert_lines:
            target["segs"] = (
                torch.from_numpy(np.ascontiguousarray(sampled_vert_segs))
                .contiguous()
                .float()
            )
            target["lines"] = (
                torch.from_numpy(np.ascontiguousarray(sampled_vert_lines))
                .contiguous()
                .float()
            )
            target["line_mask"] = (
                torch.from_numpy(np.ascontiguousarray(vert_line_mask))
                .contiguous()
                .float()
            )
        else:
            target["segs"] = (
                torch.from_numpy(np.ascontiguousarray(sampled_segs))
                .contiguous()
                .float()
            )
            target["lines"] = (
                torch.from_numpy(np.ascontiguousarray(sampled_lines))
                .contiguous()
                .float()
            )
            target["line_mask"] = (
                torch.from_numpy(np.ascontiguousarray(line_mask)).contiguous().float()
            )
            
            target["horizon_lines1"] = (
            torch.from_numpy(np.ascontiguousarray(gt_horizon_lines1))
            .contiguous()
            .float()
        )
            
            target["horizon_lines2"] = (
            torch.from_numpy(np.ascontiguousarray(gt_horizon_lines2))
            .contiguous()
            .float()
        )
            
        #     target["straight_segs"] = (
        #     torch.from_numpy(np.ascontiguousarray(straight_segs))
        #     .contiguous()
        #     .float()
        # )
            
        if self.return_masks:
            target["masks"] = masks
        target["org_img"] = org_image
        target["org_sz"] = org_sz
        target["input_sz"] = input_sz
        target["img_path"] = iname
        target["filename"] = iname
        target["num_segs"] = num_segs
        # target['desc_sublines'] = desc_sublines

        extra["lines"] = target["lines"].clone()
        # extra['desc_sublines'] = target['desc_sublines'].clone()
        extra["line_mask"] = target["line_mask"].clone()

        return image, extra, target

    def __len__(self):
        return self.size


def make_transform():
    return T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )


def build_yud(image_set, cfg):
    root = "/home/kmuvcl/dataset/data/processed_data/"

    PATHS = {
        "train": "train",
        "val": "val",
        "test": "test",
    }

    img_folder = root
    ann_file = PATHS[image_set]
    dataset = GSVDataset(
        cfg,
        ann_file,
        img_folder,
        return_masks=cfg.MODELS.MASKS,
        transform=make_transform(),
    )
    return dataset
