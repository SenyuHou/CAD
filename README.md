# Noisy Multi-Label Learning through Co-Occurrence-Aware Diffusion  
### 📄 [[NeurIPS 2025 Paper]](https://openreview.net/forum?id=zft0zTOFkN)

---

## 🛠️ 1. Preparing python environment

Install requirements.<br />

```
conda env create -f environment.yml
conda activate CAD-env
conda env update --file environment.yml --prune
```

> The name of the environment is set to **CAD-env** by default.  
> You can modify the first line of the `environment.yml` file to set the new environment's name.

---

## 🚀 2. Run demo script to train the CAD models

### 📊 2.1 VOC2007<br />

Download dataset:  
[Pascal VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)<br />

Example command:

```
python train_on_Voc2007.py --device cuda:0 --noise_rate 0.3 --noise_type symmetric --lr 5e-3 --nepoch 30 --log_name voc2007
```

---

### 📊 2.2 VOC2012<br />

Download dataset:  
[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)<br />

Example command:

```
python train_on_Voc2012.py --device cuda:0 --noise_rate 0.3 --noise_type symmetric --lr 5e-3 --nepoch 30 --log_name voc2012
```

---

### 🌐 2.3 MS-COCO<br />

Download dataset:  
[COCO](https://cocodataset.org/#download)<br />

Example command:

```
python train_on_COCO.py --device cuda:0 --noise_rate 0.3 --noise_type symmetric --nepoch 30 --lr 5e-3 --log_name coco
```

---

### 🏷️ 2.4 NUS-WIDE<br />

Download dataset:  
[NUS-WIDE](https://github.com/iTomxy/data/tree/master/nuswide)<br />

Example command:

```
python train_on_NUSWIDE.py --gpu_devices 0 1 2 3 4 5 6 7 --nepoch 50 --lr 1e-3 --log_name nus-wide
```

---

## 📦 3. Pre-trained model & Checkpoints

* ViT pre-trained models are available via CLIP:  
  👉 https://github.com/openai/CLIP  

Install without dependency:

```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git --no-dependencies
```

---

## 📖 Citation

If you find this work useful, please consider citing:

```
@inproceedings{CAD, 
  title={Noisy Multi-Label Learning through Co-Occurrence-Aware Diffusion},
  author={Hou, Senyu and Ren, Yuru and Jiang, Gaoxia and Wang, Wenjian},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```

---
