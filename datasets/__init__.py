from datasets.matterport import build_matterport
from datasets.gsv_dataset import build_gsv
def build_matterport_dataset(image_set, cfg):
    return build_matterport(image_set, cfg)

def build_gsv_dataset(image_set, cfg):
    return build_gsv(image_set, cfg)