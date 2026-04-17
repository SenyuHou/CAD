# Noisy Multi-Label Learning through Co-Occurrence-Aware Diffusion

## 1. Preparing python environment

Install requirements.<br />

```
conda env create -f environment.yml
conda activate CAD-env
conda env update --file environment.yml --prune
```

The name of the environment is set to **CAD-env** by default. You can modify the first line of the `environment.yml` file to set the new environment's name.

## 2. Run demo script to train the CAD models

### 2.1 VOC2007<br />

Download [Pascal VOC2007] at here: [Pascal VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)<br />

Default values for input arguments are given in the code. An example command is given:

```
python train_on_Voc2007.py --device cuda:0 --noise_rate 0.3 --noise_type symmetric --lr 5e-3 --nepoch 30 --log_name voc2007
```

### 2.2 VOC2012<br />

Download [Pascal VOC2012] at here: [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)<br />
An example command is given:

```
python train_on_Voc2012.py --device cuda:0 --noise_rate 0.3 --noise_type symmetric --lr 5e-3 --nepoch 30 --log_name voc2012
```

### 2.3 MS-COCO<br />

Download [Microsoft-COCO] at here: [COCO](https://cocodataset.org/#download)<br />

An example command is given:

```
python train_on_COCO.py --device cuda:0 --noise_rate 0.3 --noise_type symmetric --nepoch 30 --lr 5e-3 --log_name coco
```

### 2.4 NUS-WIDE<br />

Download [NUS-WIDE] at here: [NUS-WIDE](https://github.com/iTomxy/data/tree/master/nuswide)<br />

An example command is given:

```
python train_on_NUSWIDE.py --gpu_devices 0 1 2 3 4 5 6 7  --nepoch 50 --lr 1e-3 --log_name nus-wide
```

## 3. Pre-trained model & Checkpoints

* ViT pre-trained models are available in the python package at [here](https://github.com/openai/CLIP). Install without dependency: <br />

```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git  --no-dependencies
```

Trained checkpoints for the CAD models are available at [here]().

