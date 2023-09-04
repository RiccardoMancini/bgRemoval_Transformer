# Fashion Images Background Removal with Transformers - (Computer Vision & Deep Learning project)
## Abstract
This article presents two methods for identifying
and removing background from fashion images using advanced
semantic segmentation techniques. Two different neural network
based on transformers have been used. In contrast to convolutionbased methods, these approach allows to model global context
already at the first layer and throughout the network. The
reference dataset for this project consisting of 45,600 fashion
images, each containing at least one cloth, not necessary worn
by person, and a .csv file containing the encoded pixel, for
every image, to generate the masks. After that, every single
image with his mask generated was given in input to some
chosen architectures of Segmenter and SETR transformer-based
network. These networks was pre-trained on ADE20K dataset
to avoid the training on the fashion dataset from scratch. To
evaluate the models after training, the metrics used were IoU
(Intersection over Union) and Pixel Accuracy, the most common
evaluation metrics for semantic image segmentation. In the end,
if the results obtained were satisfactory, it procedeed with the
removal of the background from another set of images other
than those used for training phases.
&nbsp;&nbsp;
[Read more here](./project-paper.pdf)


## Requirements
- pytorch
- mmcv; install the correct version based on your cuda version -> (https://github.com/open-mmlab/mmcv)
- install mmsegmentation library in dev mode (https://mmsegmentation.readthedocs.io/en/latest/get_started.html);
- after cloning mmsegmentation repository, move into the mmsegmentation directory and then run the command:
  ```pip install -v -e .```
