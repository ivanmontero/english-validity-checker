import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from gensim.models import KeyedVectors
import os
from gensim.models import Word2Vec
from functools import reduce
from argparse import ArgumentParser
import random
from multiprocessing import Pool

# A utility function to take a batch of encoded sentences (an "encoded"
# word is the integer mapping of the word.) and, for each, return a list
# of tuples, where the first entry is the original sentence, and the
# second entry is its corruption. This method will be ran as a map
# function.
def corrupt(encoded_sentences, p1tree, p1embed, w2vembed, nearest_n_max):
    # The return list we will populate.
    pairs = []
    # The positions of the words we will corrupt. Index mapped with to_process
    poss = []
    # The encoded sentences to process. Index mapped with poss.
    to_process = []
    # The query to pass to the k-d tree.
    query = []

    # For each encoded sentence, we will determine if we will process it, and if
    # so, determine the position the corruption will occur.
    for s in encoded_sentences:
        pos = random.randint(0, len(s)-1)
        if random.randint(0, 999) == 0:
            pairs.append((s, s[:pos] + s[pos+1:]))
            continue
        to_process.append(s)
        query.append(p1embed[s[pos]])
        poss.append(pos)

    # Grab the nearest words in the part 1 embedding space for each word in
    # this batch. The number of neighbors is random.
    nearest_p1_candidates = p1tree.query(
        query, random.randint(2, nearest_n_max), return_distance=False, dualtree=True)

    # Finally, populate the return value with the original-corruption pairs.
    for i in range(len(to_process)):
        candidates_i = nearest_p1_candidates[i]
        s = to_process[i]
        pairs.append((s,
            s[:poss[i]] + [
            candidates_i[np.argmax(np.sqrt(np.sum(
                (w2vembed[s[poss[i]]]
                - w2vembed[candidates_i])**2,axis=-1)))]] 
            + s[poss[i]+1:]))

    # Return the original-corruption pairs.
    return pairs

# The main entry point of the program. In a main method since this program uses
# multiprocessing since the embedding searches can be done in parallel.
if __name__ == "__main__":
    # To prevent multiple imports of tensorflow.
    from keras.models import load_model

    # Parse the necessary arguments.
    parser = ArgumentParser()
    parser.add_argument("-n", "--name")
    parser.add_argument("-tr", "--train")
    parser.add_argument("-b", "--batches", type=int)
    args = parser.parse_args()

    # Load the Word2Vec embeddings/mappings created during the part 1 training.
    wv = KeyedVectors.load(args.name + ".wv")
    w2vweights = wv.vectors
    vocab = wv.vocab

    # Grab the trained embeddings from the part 1 model.
    part1model = load_model(args.name + ".h5")
    p1weights = part1model.layers[0].get_weights()[0]

    # Load the training file and grab all of the natural training sentences.
    train_file = open(args.train, "r", encoding="utf8", errors="surrogateescape")
    train_data = train_file.read().lower()
    train_sentences = list(filter(None, train_data.replace("\n", "\t").split("\t")))
    train_natural_sentences = [train_sentences[i*2] for i in range(len(train_sentences) // 2)]

    # Encode the sentences (e.g., give them their integer mappings)
    encoded_sentences = [[vocab[w].index for w in s.split(" ")] for s in train_natural_sentences]

    # Use a k-d tree to optimally search for the k-nearest part 1 words for a
    # given word.
    p1tree = KDTree(p1weights)

    # Split the corruption generation into as many workers as the machine will allow.
    # This greatly speeds up the process.
    pool = Pool()
    batch_size = len(encoded_sentences) // args.batches
    corrupted_sentences_batches = []
    print(f"{os.cpu_count()} workers, {batch_size} batch size with {batch_size//os.cpu_count()} work per worker")
    for i in range(args.batches):
        # Grab the amount of sentences to corrupt in this batch, and split it into
        # work for each of the workers.
        batch = encoded_sentences[i*batch_size:(i+1)*batch_size]
        work_size = len(batch)//os.cpu_count()
        work = [batch[work_size*i:(i+1)*work_size] for i in range(os.cpu_count())]
        
        # Begin sentence corruption and populate a list of the corruptions.
        print(f"Starting batch {i+1} of {args.batches}, batch size {len(batch)}")
        corrupted_sentences_batches.extend(
            pool.starmap(corrupt, [(w, p1tree, p1weights, w2vweights, 100) for w in work]))
        print(f"Finished batch {i+1} of {args.batches}. Writing to file")

    # Write the corruptions in a file where the first line is the original sentnece, and
    # the second line is the corruption.
    with open("corruptions.txt", "w+", encoding="utf8", errors="surrogateescape") as f:
        for pairs in corrupted_sentences_batches:
            for pair in pairs:
                assert " ".join([wv.index2word[i] for i in pair[0]]) != " ".join([wv.index2word[i] for i in pair[1]])
                f.write(" ".join([wv.index2word[i] for i in pair[0]]))
                f.write("\t")
                f.write(" ".join([wv.index2word[i] for i in pair[1]]))
                f.write("\n")