# General requirement
- Code clean, basic for easy start.
- Step by step coding with user interaction.

## Prepare dataset
[]  Write code `convert_dataset.py` to generate this kind of format for dataset

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
```
given the current structure of dataset is like normal coco structure as below
`data_dir`: `/media/vuthede/Lexar/data/oms_incabin`
`name` : `images` # note: in here there are multiple sub folders. so get the image from the annotation is better solution
`annotation_path`: `/media/vuthede/Lexar/data/oms_incabin/annotations/ver110/object_anno_7job.json`

## Finetuning
[] Basic code for fine-tuning `fintuning.py`

```python
from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(dataset_dir=<DATASET_PATH>, epochs=10, batch_size=4, grad_accum_steps=4, lr=1e-4, output_dir=<OUTPUT_PATH>)
```
There are 10 classes is as follow
object_labels = ['person', 'head', 'steering_wheel', 'laptop', 'cell_phone', 'infant', 'baby_seat', 'food', 'cigarette', 'bag']



