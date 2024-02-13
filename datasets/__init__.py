from datasets.matterport import build_matterport
from datasets.gsv_dataset import build_gsv
from datasets.scannet import build_scannet
from datasets.su3 import build_su3
from datasets.yud import build_yud


def build_matterport_dataset(image_set, cfg):
    return build_matterport(image_set, cfg)

def build_gsv_dataset(image_set, cfg):
    return build_gsv(image_set, cfg)

def build_scannet_dataset(image_set,cfg):
    return build_scannet(image_set,cfg)

def build_su3_dataset(image_set,cfg):
    return build_su3(image_set,cfg)

def build_yud_dataset(image_set,cfg):
    return build_yud(image_set,cfg)