# Vanishing Point Detector
- This deep learning model is designed to detect vanishing points in images based on the Manhattan assumption.
- We have designed an architecture optimized for vanishing point detection by employing a transformer model, learned queries, and a loss function designed using bipartite matching.
- This deep learning model has successfully reduced the weight by 64.5% compared to other vanishing point detectors.
- Additionally, we propose a deep learning model that determines the relative rotation between two images by applying the vanishing point detector. If you are interested in the Relative Rotation Regressor, please refer to this [github.](https://github.com/MMSS9402/Relative-Rotation-Regressor/tree/main).
- You can view the paper at this [Link.](https://kookmin.dcollection.net/public_resource/pdf/200000737077_20240530151846.pdf)


## installation
```shell
pip install -r requirements.txt
```

## Dataset
MatterPort3D dataset Download [Jin et al.](https://github.com/jinlinyi/SparsePlanes/blob/main/docs/data.md)

## Data Preprocessing
```shell
cd scripts/data_preprocessing
python LSD2.py
```

## train script
```shell
python run.py 
```
