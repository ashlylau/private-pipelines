## Implementation of small iris dataset
This script trains ~150 private models using adjacent datasets generated from the iris dataset.

### Usage:
```python3 -W ignore train_models.py --train_all --epochs 20 --batch_size 8 --num_models 30```

### Installation
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```