# Generate occluded images

This code aims to generate occlusion images in following methods:


* **Blurring**: The region to be occluded applies the Gaussian blur algorithm.

* **Black block filling**: The region to be occluded is filled with all pixels set to 0 (black).

* **Image inpainting**: For the region requiring occlusion, we remove its content and perform image inpainting to fill the region.

Please make sure that `image_features_mscoco.csv`, `image_features_pascal.csv` are generated in the parent directory.

## generate occluded images (blurring and black filling)
```
dataset='mscoco'
python3 generate_occlude_image.py -s 'occlusion_image' -d $dataset
```

## image inpainting
Please refer to [LaMa](https://github.com/advimman/lama) to prepare conda environment and download checkpoint.

```
cd lama
dataset='mscoco'
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/${dataset} outdir=$(pwd)/${dataset}_output
cd ..
```

## move the outputs of image inpainting model
The occluded images of image inpainting are in `lama`, we should move them to the directory `occlusion_image`.

```
dataset='mscoco'
python3 move_lama_output.py -s 'occlusion_image' -d $dataset
```