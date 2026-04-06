#  Super-resolution of thermal infrared images to assess temperature spatial heterogeneity in rivers

#### Folder structure
```
Superresolution-TIR/
│
├── data/
│   ├── metadata.json                               
├── requirements.txt
├── README.md
│
├── scripts/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── edsr.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── create_metadata.py
│   │   └── data_prep.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_edsr.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluate.py
│
└── jobs/
    ├── train_edsr.sh
    ├── eval_edsr.sh
    └── eval_all.sh
```
