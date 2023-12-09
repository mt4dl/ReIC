# Enhanced Follow-up Test Cases Generation

This code aims to generate follow-up images by following three MRs:

* **Cropping**: cut out a portion of the image without altering its proportions.
* **Stretching**: elongate the image along a fixed axis and discard the portion that exceeds the image boundaries.

* **Rotation**: rotate the image by a specific angle. Rotation often introduces dark borders encircling the image, which makes the image devoid of reality. Thus, we perform an additional cropping step to eliminate these dark borders.

## generate follow-up images
```
dataset='mscoco'
model='vinvl'
python3 get_object_area.py -d $dataset -m $model
python3 dynamic_mr.py -d $dataset -m $model -t 'crop'
python3 dynamic_mr.py -d $dataset -m $model -t 'stretch'
python3 dynamic_mr.py -d $dataset -m $model -t 'rotate'
```

Follow-up images will be generated in the directory `follow_up_images`.