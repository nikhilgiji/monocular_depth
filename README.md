## Monocular Depth Estimation 

Real time monocular depth estimation using intel MiDaS 

[Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer. René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun](https://arxiv.org/abs/1907.01341v3)


### Download Model 

Create a model directory and cd into it

```shell
    mkdir model && cd model
```
Download the Intel MiDaS Lite model from tensorflow hub [here](https://tfhub.dev/intel/lite-model/midas/v2_1_small/1/lite/1) inside the model folder. 


### Run Image 

Run depth estimation on a single image

```shell
    python monocular_img.py image_name.jpg 
``` 

For example 

```shell
    python monocular_img.py test_img.jpg 
``` 
