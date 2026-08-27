# Dataset

Images and labels are **not** committed — they are large, and most traffic
datasets carry their own licence terms. This describes the layout the code expects.

## Layout

```
dataset/
  data.yaml
  images/train/scene0001.jpg
  images/val/scene0500.jpg
  labels/train/scene0001.txt
  labels/val/scene0500.txt
```

Ultralytics locates a label by taking the image path and replacing `/images/`
with `/labels/`, then swapping the extension for `.txt`. The two trees must
mirror each other exactly. A `.txt` sitting beside its `.jpg` is silently ignored —
this is the single most common reason a run reports "0 labels found".

## Label format

One line per object, space-separated, coordinates normalised to 0–1:

```
<class_id> <x_center> <y_center> <width> <height>
```

An empty `.txt` is valid and means "no objects in this image" — a legitimate
negative sample. It is indistinguishable from a labelling failure, so treat
unexpected empty files as a bug, not as data.

## Sanity checks before training

```bash
# Every image has a label file
ls dataset/images/train | wc -l
ls dataset/labels/train | wc -l

# No label is a near-full-frame box (the classic merged-mask failure)
awk '$4 > 0.97 && $5 > 0.97 {print FILENAME": "$0}' dataset/labels/train/*.txt

# No class id exceeds the count in data.yaml
awk '{print $1}' dataset/labels/train/*.txt | sort -un
```

## How much data

Roughly 300–500 labelled instances *per class* before results are worth
publishing; 1500+ images total for a detector that generalises across lighting
and weather. Below ~100 images, a COCO-pretrained model will almost certainly
outperform your fine-tune.

## Public starting points

Rather than labelling from scratch, consider an existing labelled set —
BDD100K, KITTI, or a Roboflow Universe traffic dataset. Check each licence
before redistributing anything derived from it.
