# GDCN: Geometry-Detail Collaborative Network for Remote Sensing Image Super-Resolution



## Install
```bash
git clone https://github.com/zxypa/GDCN.git
cd GDCN
```

## Environment
> * CUDA 11.8
> * Python 3.8
> * PyTorch 2.0.0
> * Torchvision 0.15.1
> * BasicSR 1.4.2

## Dataset
Please download the following remote sensing benchmarks:

| Data Type | [AID](https://captain-whu.github.io/AID/) | [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) | [DIOR](https://gcheng-nwpu.github.io/#Datasets) |
| :----: | :----: | :----: | :----: |
| Training | Download | None | None |
| Testing | Download | Download | Download |

## Test
**Step I.** Prepare the testing data as follows:

```text
/path/to/dataset/
├── GT/
│   ├── 000.png
│   └── ...
└── LR/
    ├── 000.png
    └── ...
```

**Step II.** Modify the dataset path in the evaluation file.

**Step III.** Run:

```bash
python eval_4x.py
```

## Train
```bash
python train.py
```

## Contact
For questions or suggestions, please contact:  
Email: hnpacv@163.com
