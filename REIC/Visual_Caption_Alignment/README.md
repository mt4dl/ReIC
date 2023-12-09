# Visual-Caption Alignment in REIC

## Obtain OD results of source images
* refer to [SGG model](https://github.com/microsoft/scene_graph_benchmark) to obtain the OD result of source images and get the result *csv* file in this directory: `image_features_mscoco.csv`, `image_features_pascal.csv`.

## Generate occluded images and corresponding captions
* generate occluded images. Please refer to [image_occlusion](image_occlusion/README.md) part.

* obtain captions of occluded images. Get the corresponding captions of occluded images and then place them in the file structure shown below:
    ```
    model_captions
    └──mscoco
        └──vinvl
            ├──raw.tsv
            ├──blur.tsv
            ├──black.tsv
            └──inpainting.tsv
    ```

## Get localization results
```
dataset='mscoco'
model='vinvl'
python3 get_caption_objects.py -d $dataset -m $model
python3 OD_localization.py -d $dataset -m $model
python3 occlusion_based_localization.py -d $dataset -m $model
```