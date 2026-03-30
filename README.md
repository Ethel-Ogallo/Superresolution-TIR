#  Super-resolution of thermal infrared images to assess temperature spatial heterogeneity in rivers

#### Folder structure
```
Superresolution-TIR/
│
├── data/metadata.csv                         
├── data/models.csv                            
├── requirements.txt
├── README.md
│
├── scripts/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── visualize.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── edsr.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── build_metadata.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_edsr.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluate.py
│       └── compare_models.py
│
└── jobs/
    ├── train_edsr.slurm
    ├── eval_edsr.slurm
    └── eval_all.slurm
```