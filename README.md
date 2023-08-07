# Reproduce

0. Download data from kaggle

1. Unzip data

```bash
unzip 2023-summer-nycu-dl-lab3.zip -d dataset/resnet18
unzip 2023-summer-nycu-dl-lab3-2.zip -d dataset/resnet50
unzip 2023-summer-nycu-dl-lab3-3.zip -d dataset/resnet152
```

2. Create virtual environment and install requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Run the code

```bash
# resnet 18
python3 inference.py --resnet 18 --batch_size 32 --dim 256 --val --checkpoint checkpoint/resnet18-acc98.81.pt

# resnet 50
python3 inference.py --resnet 50 --batch_size 36  --dim 256 --val --checkpoint checkpoint/resnet50-epoch545-acc98.5.pt

# resnet 152
python3 main.py --resnet 152 --batch_size 32 --epoch 600 --dim 225 --val --checkpoint checkpoint/resnet152-acc89.24.pt
```
