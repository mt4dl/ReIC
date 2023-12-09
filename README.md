# **Artifact of ReIC**

The Image Captioning (IC) technique is widely used to describe images in natural language. However, even state-of-the-art IC systems can still produce incorrect captions and lead to misunderstandings. In this paper, we propose ReIC to perform metamorphic testing for IC systems with some reduction-based transformations (e.g., cropping and stretching). We employ ReIC to test five popular IC systems. The results demonstrate that ReIC can sufficiently leverage the provided test images to generate follow-up cases of good reality, and effectively detect a great number of distinct violations, without the need for any pre-annotated information.
This repository is the supplementary material for our tool *ReIC*. It contains:

1. **The replication package for our experiments:** the scripts to replicate our experiments.
2. **The datasets used in our experiments:** the download linkage of datasets used in our paper and the raw data related to the results reported in thr paper.
3. **The complete experimental results for this paper:** the complete experimental results on our experiment.

## Datasets

The images utilized in our experiments are sourced from two datasets: the *MSCOCO* and the *PASCAL* dataset. The complete datasets can be obtained from the following links:
* MSCOCO: http://images.cocodataset.org/zips/train2014.zip, http://images.cocodataset.org/zips/val2014.zip
* PASCAL: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

For reproducibility purposes, the images employed in our experiments can be accessed from [here](https://drive.google.com/file/d/1behrR2ByxtPqZT9SzRvIf8T2gw2wWmPX/view?usp=drive_link). You can download the images and unzip them into the directory `/ReIC/data`.


## Complete experimental results

Due to the space limit for the paper, we have not included all the detailed experimental results (the results on the whole models and datasets of RQ2, RQ3 and RQ4) in our paper. Here we provide the complete experimental results in the directory `/complete-results`.


## Replication package

The replication package can be found in the directory `/REIC`. Please refer to our [replication package](REIC/README.md).