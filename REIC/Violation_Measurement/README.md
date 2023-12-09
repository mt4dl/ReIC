# Violation Measurement

## get captions of follow-up images

* Firstly, make sure to generate follow-up images following [this script](../Follow_up_Test_Cases_Generation/README.md).

* Then, get the captions of generated follow-up images following [related codes](../README.md/#related-codes). Place them into the directory `follow_up_captions`.

## measure violation
```
dataset='mscoco'
model='vinvl'
mr='crop'
python3 violation_measurement.py -d $dataset -m $model -r $mr
```