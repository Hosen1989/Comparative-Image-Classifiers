# Comparative-Image-Classifiers
Comparative Analysis of Image Classifiers on MNIST Digits Dataset

# Frist
edit `train_loop.py` with different combinations:
```python
variables = {
    "lr": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5],
    "hidden_units": [128, 256, 512, 1024, 2048],
    "optimizers": [torch.optim.SGD, torch.optim.Adam, torch.optim.RMSprop],
    "layer_1_act": [torch.relu],
    "layer_2_act": [torch.relu],
    "layer_3_act": [torch.softmax],
}
```

# Second
collect metrics from each models sub folder @ "./models_complete/**model-runid-name**/version_**x**/metrics.csv" folder,
all there metrics are merged in single csv file `data.csv` that load into simple SQLite db file @ './db/dat01'.

# Lastly
In jupyter notebook `analysis_v01.ipynb` we load the db and do our analysis ^_^
