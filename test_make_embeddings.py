from make_embeddings import main

def test_embeddings_size():
    allVectors = main()
    for vec in allVectors:
        assert len(allVectors[vec]) == 512