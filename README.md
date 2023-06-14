# GaitGraph2: Towards a Deeper Understanding of Skeleton-based Gait Recognition

Accepted at: [17th IEEE Computer Society Workshop on Biometrics 2022](https://www.vislab.ucr.edu/Biometrics2022/index.php) (CVPRW'22)

[![arxiv](https://img.shields.io/badge/arXiv-2204.07855-red)](https://arxiv.org/abs/2204.07855)

_Previous version:_<br>
__GaitGraph: Graph Convolutional Network for Skeleton-Based Gait Recognition__ (ICIP'21)<br>
[![github](https://img.shields.io/badge/Github-GaitGraph-black)](https://github.com/tteepe/GaitGraph)
[![DOI:10.1109/ICIP42928.2021.9506717](https://img.shields.io/badge/DOI-10.1109%2FICIP42928.2021.9506717-blue)](https://doi.org/10.1109/ICIP42928.2021.9506717) 
[![arxiv](https://img.shields.io/badge/arXiv-2101.11228-red)](https://arxiv.org/abs/2101.11228)

## Preparation
```
git clone https://github.com/tteepe/GaitGraph2.git
conda create -n gait python=3.8
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch -y
pip install pytorch-lightning==1.7
pip install pytorch-metric-learning pandas
pip install -U 'jsonargparse[signatures]'

conda install pyg -c pyg # no module named torch-sparse

pip install -e .
```


### Datasets
- CASIA-B: Download from [here](https://github.com/tteepe/GaitGraph/releases/tag/v0.1) and move `casia-b_pose_coco.csv` to `data`
- OUMVLP-Pose: Download from [here](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLPPose.html)

### Pre-trained models
You can download pre-trained models from [here](https://github.com/tteepe/GaitGraph2/releases/tag/v0.1).

## Running the code
We use [PyTorch Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html) for configuration and training.

Train:
```bash
cd GaitGraph
# CASIA-B
python3 gaitgraph_casia_b.py fit --config configs/casia_b.yaml 
# OUMVLP-Pose (OpenPose)
python3 gaitgraph_oumvlp.py fit --config configs/oumvlp.yaml
# OUMVLP-Pose (AlphaPose)
python3 gaitgraph_oumvlp.py fit --config configs/oumvlp.yaml --data.keypoints alphapose
```

Test:
```bash
python3 gaitgraph_{casia_b,oumvlp}.py predict --config <path_to_config_file> --ckpt_path <path_to_checkpoint> --model.tta True

# example
python gaitgraph_oumvlp.py predict --config lightning_logs/version_0/config.yaml --ckpt_path lightning_logs/version_0/checkpoints/gaitgraph-oumvlp-epoch\=599-val_loss_epoch\=0.80.ckpt --model.tta True
```

Logs and checkpoints will be saved to `lighting_logs` and can be shown in tensorboard with:
```bash
tensorboard --logdir lightning_logs
```

## Visualization of Activations
To run the visualization add run the following command:
```bash
python3 gaitgraph_{casia_b,oumvlp}.py predict --config <path_to_config_file> --ckpt_path <path_to_checkpoint> --config configs/draw.yaml --model.tta False
```
Examples:<br>
![002-06-01](images/002-060-01.gif)
![008-255-00](images/008-255-00.gif)

### Citation
If you use GaitGraph2, please use the following BibTeX entry.

```
@misc{teepe2022gaitgraph2,
  doi = {10.48550/ARXIV.2204.07855},
  url = {https://arxiv.org/abs/2204.07855},
  author = {Teepe, Torben and Gilg, Johannes and Herzog, Fabian and H\"ormann, Stefan and Rigoll, Gerhard},
  title = {Towards a Deeper Understanding of Skeleton-based Gait Recognition},
  publisher = {arXiv},
  year = {2022}
}
```
