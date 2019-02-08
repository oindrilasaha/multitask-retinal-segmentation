# multitask-retinal-segmentation

This repository contains the implementation for the paper "Learning with Multitask Adversaries using Weakly Labelled Data for Semantic Segmentation in Retinal Images"

https://openreview.net/forum?id=HJe6f0BexN

### Download Model

```
python get_model.py
```

This downloads and saves the pre-trained model in the same folder.

### Example usage
```
python test.py --model_path model.pkl --dataset idrid --img_path IDRiD_79.jpg --out_folder idrid79

```

When using above command, mention the dataset name you are using after the --dataset flag ( eg. idrid, drive ). Example image and results folder have been provided. 
