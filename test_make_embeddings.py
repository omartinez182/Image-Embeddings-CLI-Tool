from make_embeddings import main
import argparse

def test_embeddings_size():
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--URL',
                        type=str)
        parser.add_argument('--inputDir',
                            type=str,
                            default='data/images')
        arguments = parser.parse_args()
        allVectors = main(arguments)

        for vec in allVectors:
            assert len(allVectors[vec]) == 512