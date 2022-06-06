[![Pytorch Image Embeddings test with Github Actions](https://github.com/omartinez182/Image-Embeddings-CLI-Tool/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/omartinez182/Image-Embeddings-CLI-Tool/actions/workflows/main.yml)

# Image Embeddings
 
This CLI tool generates image embeddings from either URLs or a local folder. The tool uses a pre-trained [ResNet-18](https://pytorch.org/hub/pytorch_vision_resnet/) model to generate the embeddings. The embeddings can then be used as features for different types of modeling tasks, one example would be visually-aware recommendations, such as in [VBPR](https://arxiv.org/pdf/1510.01784.pdf).


## Getting Started


1) Create a virtual environment.

```
cd Image-Embeddings-CLI-Tool/
virtualenv venv
```

2) Activate the virtual environment.

```
source venv/bin/activate
```

3) Then run: (recommended)

```
make all
```

or manually install of the requirements.

```
pip install -r requirements.txt
```

### Usage

If you'd like to create embeddings for images in a specific folder, you just need to specify the path as an argument. For example:

```
python3 make_embeddings.py --inputDir 'data/images'
```

If you'd like to create embeddings for an image using a URL, just pass the URL as an argument. For example:

```
python3 make_embeddings.py --URL 'https://media.wired.com/photos/5c18253a9fe42d6b6e532fb7/master/w_1600%2Cc_limit/%2520hornless%2520heritage_nikita%2520teryoshin_1.jpg'
```

Make sure that you only have the desired images in the folder ```data/output/inputImagesCNN```. 

**Output:** The embeddings will be saved in the folder with the same name (```embeddings```) as a pickle file which you can then load and use as features for your modeling task of choice.

```NOTE: Currently the library to create the embeddings using PyTorch supports only jpg images.```


## Contributions

If you'd like to contribute, please make sure that before you do a push, you run:

```make all```