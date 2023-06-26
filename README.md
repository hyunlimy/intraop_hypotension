# intraop_hypotension
Official implementation of the article entitled "Comparison of deep learning-based intraoperative hypotension prediction models to the logistic regression model in unbiased real-world samples: A retrospective analysis of prospectively collected registry data"


## Reproduce our work!
You can reproduce our work using our codes. <br>
The vital signal data will be automatically downloaded from [VitalDB](https://vitaldb.net). <br>
This code is written in Python 3.8.12

Run the codes as follows:

##### (optional) 0. Install packages:
```bash
pip install -r requirements.txt
```

##### 1. Build datasets:

```python
python build_dataset.py
```
##### 2. train models:
```python
python train.py
```

