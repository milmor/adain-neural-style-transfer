# AdaIN Neural Style Transfer

Implementation of the paper:

> Xun Huang and Serge Belongie. [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) (ICCV 2017).

![Architecture](./images/architecture.jpg)


## Examples
<p align='center'>
  <img src='images/content_img/chicago_cropped.jpg' width="160px">
  <img src='images/style_img/ashville_cropped.jpg' width="160px">
  <img src='images/output_img_mixed7/step_152000_512x512/chicago_cropped_ashville_cropped.jpeg' width="160px">
<br>
  <img src='images/content_img/avril_cropped.jpg' width="160px">
  <img src='images/style_img/picasso.png' width="160px">
  <img src='images/output_img_mixed7/step_152000_512x512/avril_cropped_picasso.jpeg' width="160px">
<br>
  <img src='images/content_img/cornell_cropped.jpg' width="160px">
  <img src='images/style_img/woman_with_hat_matisse_cropped.jpg' width="160px">
  <img src='images/output_img_mixed7/step_152000_512x512/cornell_cropped_woman_with_hat_matisse_cropped.jpeg' width="160px">
</p>


## Dependencies
- Python 3.8
- Tensorfow 2.3


## Usage
### Train
1. Download [MSCOCO images](http://mscoco.org/dataset/#download) and [Wikiart images](https://www.kaggle.com/c/painter-by-numbers).
2. Use `--name=<model_name>`, `--content_dir=<coco_path>` and `--style_dir=<wikiart_path>` to provide model name and datasets paths. 
```
python train.py --name=<model_name>  --content_dir=<coco_path> --style_dir=<wikiart_path>
```

### Test
Run `test.py`. It will save every possible combination of content and styles to the output directory.
```
python test.py --name=<model_name> --test_content_img=<content_path> --test_style_img=<style_path>
```

### Tensorboard
Run `tensorboard --logdir ./`


## Licence
Copyright (c) 2020 Emilio Morales. Free to use, copy and modify for academic research purposes, as long as proper attribution is given and this copyright notice is retained. Contact me for any use that is not academic research. (email: mil.mor.mor at gmail.com).


## Citation
```
@software{morales2020adain,
  author = {Morales, Emilio},
  title = {Adain neural style transfer},
  url = {https://github.com/milmor/adain-neural-style-transfer},
  year = {2020},
}
```
```


