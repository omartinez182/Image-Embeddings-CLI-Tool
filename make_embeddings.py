import os
import pickle
import argparse
from PIL import Image
from torchvision import transforms
from img2vec_pytorch import Img2Vec
from tqdm import tqdm
import requests
import traceback
import logging

def main(args):
    """
    Generate 512 embeddings for a given image(s).
    """

    if args.URL is not None:
        # Create directory for downloaded images
        inputDir = "data/downloaded_images"
        os.makedirs(inputDir, exist_ok=True)
        image_name = args.URL.rsplit('/', 1)[-1]
        path_and_name = os.path.join(inputDir, image_name)
        # Download the image
        response = requests.get(args.URL)
        # Save the image
        file = open(path_and_name, "wb")
        file.write(response.content)
        file.close()
    else:
        inputDir = args.inputDir

    # Load images
    inputDim = (224,224)
    inputDirCNN = "data/output/inputImagesCNN"

    os.makedirs(inputDirCNN, exist_ok=True)

    transformationForCNNInput = transforms.Compose([transforms.Resize(inputDim)])

    for imageName in os.listdir(inputDir):
        try:
            I = Image.open(os.path.join(inputDir, imageName))
            newI = transformationForCNNInput(I)

            newI.save(os.path.join(inputDirCNN, imageName))
            
            newI.close()
            I.close()
        except Exception as e:
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
    with open('embeddings/allEmbeddings.pkl', 'wb') as handle:
        pickle.dump(allVectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return allVectors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--URL',
                    type=str)
    parser.add_argument('--inputDir',
                        type=str,
                        default='data/images')
    arguments = parser.parse_args()
    main(arguments)