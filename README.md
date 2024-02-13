### CV-Final-Project--MaskGIT



##### Data Preparation:

You can download the TinyImageNet dataset that we use from this link: http://cs231n.stanford.edu/tiny-imagenet-200.zip, simply unzip the file and place it under `/data` folder, forming structure like:

```
/data/tiny-imagenet-200/
```



##### Prerequisites

You can use the following command to install the required python libraries:

```
pip install -r requirements.txt
```



##### Run the Code

You can change the configurations in `configs/maskgit.yaml` .

Pretrained .pth files can be downloaded from this link:

https://drive.google.com/drive/folders/1-7PQ_HRmfpVpme4YzHvFhlaZt0BMKKOf?usp=sharing

You can set the `run` param in `configs/maskgit.yaml` to `test` and set the path to the downloaded pretrained weights and run

 ````python
python vqvae_main.py 
#or 
python transformer_main.py
 ````

to get two test data folders containing images before and after running the model, and calculate FID using 

```
pip install pytorch-fid
python -m pytorch_fid path/to/dataset1 path/to/dataset2
```

before the testing process, you may need to first adjust the .yaml file's params: `vqvae_test_dataset, transformer_test_dataset` accordingly.





##### Referenced Works

https://github.com/CompVis/taming-transformers

https://github.com/google-research/maskgit

https://github.com/dome272/MaskGIT-pytorch/

https://github.com/dome272/VQGAN-pytorch/

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
