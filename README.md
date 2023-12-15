# MotifRGC

## Get Started
First, you need to get Python environments ready:
```bash
conda create -n dl python=3.9.0
conda activate dl
pip install -r requirements.txt
```

Then, to run Link Prediction task using GCN backbone, you can use the following commands:
```bash
source ./scripts/LP/gcn/cora.sh
```
And for other backbones, datasets, tasks are in the similar way.