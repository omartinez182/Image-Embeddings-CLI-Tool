import os
import pandas as pd
import numpy as np
import pickle
import argparse
from PIL import Image
from torchvision import transforms
from img2vec_pytorch import Img2Vec
from tqdm import tqdm
import traceback
import logging

def main(args):
    """
    Generate 512 embeddings for a given image(s).
    """
    # Load images
    inputDim = (224,224)
    inputDir = args.inputDir
    inputDirCNN = "data/output/inputImagesCNN"

    os.makedirs(inputDirCNN, exist_ok = True)

    transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])

    for imageName in os.listdir(inputDir):
        try:
            I = Image.open(os.path.join(inputDir, imageName))
            newI = transformationForCNNInput(I)

            newI.save(os.path.join(inputDirCNN, imageName))
            
            newI.close()
            I.close()
        except Exception as e:
            pass
            print(e)
            logging.error(traceback.format_exc())

    # Generate embeddings
    img2vec = Img2Vec(cuda=False, model='resnet-18')

    allVectors = {}
    print("Converting images to feature vectors:")
    for image in tqdm(os.listdir("data/output/inputImagesCNN")):
        try:
            I = Image.open(os.path.join("data/output/inputImagesCNN", image))
            vec = img2vec.get_vec(I)
            allVectors[image] = vec
            I.close() 
        except:
            pass

    # Save embedding vectors
    with open('models/allEmbeddings.pkl', 'wb') as handle:
        pickle.dump(allVectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(allVectors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputDir',
                        type=str,
                        default='data/images')
    args = parser.parse_args()
    main(args)