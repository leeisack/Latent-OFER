# Latent-OFER: Detect, Mask, and Reconstruct with Latent Vectors for Occluded Facial Expression Recognition

background paper implementation

MAE: https://github.com/facebookresearch/mae

deep-SVDD: https://github.com/lukasruff/Deep-SVDD-PyTorch

SVDD: https://github.com/iqiukp/SVDD-Python

ViT: https://github.com/lucidrains/vit-pytorch


## Preparation
- Download pre-trained model of [MSCeleb](https://drive.google.com/file/d/1u2NtY-5DVlTunfN4yxfxys5n8uh7sc3n/view?usp=sharing) and move the file to `./models`

dataset
- Download [RAF-DB](http://www.whdeng.cn/raf/model1.html) dataset and extract the `raf-basic` dir to `./datasets`
- Download [AffectNet](http://mohammadmahoor.com/affectnet/) dataset and extract the `AffectNet` dir  to `./datasets` 
- Download [KDEF](https://www.kdef.se/download-2/index.html) dataset and extract the `KDEF` dir  to `./datasets` 

- The dataset for test(syn-KDEF) recon-image only is here. https://drive.google.com/file/d/1woKlRn6PuvfNGYD1941I8SFw5oR0lrEF/view?usp=sharing
- The dataset for test(syn-KDEF) latent vector(.pt) only is here. https://drive.google.com/file/d/1L6mAI1I_y65nPzBWePlPP90MqlTsQ23T/view?usp=share_link
- The pre-trained model is here. https://drive.google.com/file/d/1S_cTYML_HUnRZ6P1CC8gTK9dRT4oEohv/view?usp=share_link

### Setting
The location of the label is here. "./datasets/EmoLael/label.txt
The location of the images is here. "./datasets/Images/
The location of the PT files is here. "./datasets/PT_files/
The location of the pre-trained model is here. "./checkpoints/


## Training

run:
```
CUDA_VISIBLE_DEVICES=0 python main.py
```


## only test

There is a simple test to Latent-OFER model for a emotion inference:
```
CUDA_VISIBLE_DEVICES=0 python only_test.py
```


## Grad CAM++ Reproduction
Our  experiment of grad cam++ was based on the package `grad-cam 1.3.1`, which could be pulled by:

```
pip install grad-cam==1.3.1
```

Then, run the following code to dump the visual results. (Need to replace several  variables manually.)

```
python run_grad_cam.py
```
