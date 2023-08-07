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
python3 main.py --resnet 18 --batch_size 32 --dim 256 --epoch 1000 --lr 0.0001 --val --checkpoint
```
