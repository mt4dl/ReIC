# **Replication package for *REIC***

## Environment requirements:

* NVIDIA RTX 2080 Ti
* python==3.7.0
* torch==1.7.1
* numpy==1.21.6
* stanza==1.4.0
* nltk==3.7
* opencv-python==4.6.0
* pickle==0.7.5

## Related codes 

In our experiment, we use the following repositories to obtain the captions of test images.

* Show, Attend and Tell (https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning).
* Oscar and VinVL (https://github.com/microsoft/Oscar).
* OFA (https://github.com/ofa-sys/ofa)
* Microsoft Azure API (https://azure.microsoft.com/en-us/services/cognitive-services/).

## File structure

* `data`:
  * images: available source images. The images employed in our paper can be accessed from [here](https://drive.google.com/file/d/1behrR2ByxtPqZT9SzRvIf8T2gw2wWmPX/view?usp=drive_link).
  * captions: corresponding source captions. You can refer to [related codes](#related-codes) to obtain the captions of source images.

* `functions`: some static functions like extracting object from caption and semantic similarity used for our experiment. 
  * please download and unzip [word embeddings](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) (`wiki-news-300d-1M.vec`) for our experiment into `functions` directory.

* `Visual_Caption_Alignment`: the codes for *Visual-Caption Alignment* module. This module aims to align the objects described in the caption with their positions in the image. Please refer to [Visual_Caption_Alignment.readme](Visual_Caption_Alignment/README.md).

* `Follow_up_Test_Cases_Generation`: the codes for *Enhanced Follow-up Test Cases Generation* module. This module this module aims to generate follow-up test cases with lower ambiguity and higher diversity. Please refer to [Follow_up_Test_Cases_Generation.readme](Follow_up_Test_Cases_Generation/README.md).

* `Violation_Measurement`: the codes for *Violation Measurement* module. This module measures whether there exists a violation by comparing the objects in the source caption and follow-up caption. Please refer to [Violation_Measurement.readme](Violation_Measurement/README.md).

