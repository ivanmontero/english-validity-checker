import pandas as pd
import numpy as np
from keras.models import Sequential, load_model, Model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import keras.backend as K
from keras.layers import LSTM, Bidirectional, Embedding, Dense, Flatten, Add, Concatenate, Input
from keras.optimizers import Adam
from gensim.models import KeyedVectors
import os
from gensim.models import Word2Vec
from functools import reduce
from argparse import ArgumentParser

# Parse all command-line arguments.
parser = ArgumentParser()
parser.add_argument("-n", "--name")
parser.add_argument("-tr", "--train")
parser.add_argument("-te", "--test")
parser.add_argument("-e", "--epochs", type=int, default=1)
parser.add_argument("-vs", "--validation_split", type=float, default=0.01)
parser.add_argument("-v", "--validation", action="store_true")
parser.add_argument("-bs", "--batch_size", type=int, default=32)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
args = parser.parse_args()

# Open training and test files, and sanitize.
train_file = open(args.train, "r", encoding="utf8", errors="surrogateescape")
test_file = open(args.test, "r", encoding="utf8", errors="surrogateescape")
train_data = train_file.read().lower()
test_data = test_file.read().lower()

# Extract all sentences from the test and train files.
train_sentences = list(filter(None, train_data.replace("\n", "\t").split("\t")))
train_natural_sentences = [train_sentences[i*2] for i in range(len(train_sentences) // 2)]
print(f"# of train sentences: {len(train_sentences)}")
test_sentences = list(filter(None, test_data.replace("\n", "\t").split("\t")))
print(f"# of test sentences: {len(test_sentences)}")

# Get the length of the longest sentence (for model input size).
all_sentences = train_sentences + test_sentences
max_sentence_length = reduce(lambda x, y: max(x,y), list(map(lambda x: len(x.split(" ")), all_sentences)))
print(f"Max sentence length: {max_sentence_length}")

# Extract every word from the train/test files, so we can create an encoding
# for each word.
all_words = set([w for s in all_sentences for w in s.split(" ")])
print(f"# of total words: {len(all_words)}")
extra_words = all_words.difference(set([w for s in train_natural_sentences for w in s.split(" ")]))
print(f"# of words not in training: {len(extra_words)}")

# We will use Gensim's implementation of Google's Word2Vec to get a decent
# starting embedding for each word. We will train Word2Vec using natural
# sentences found in the training file. We will treat extra words (which are
# absent from our training set) as stand-alone sentences. Our embeddings
# will each be a 100-dimensional float vector.
gen_model_sentences = [[w for w in s.split(" ")] for s in train_natural_sentences] + [[word] for word in extra_words]
print("Training Word2Vec word embeddings...")
gen_model = Word2Vec(gen_model_sentences,
                     size=128,
                     workers=os.cpu_count(),
                     min_count=0,
                     iter=5,
                     window=10)
wv = gen_model.wv
weights = wv.vectors
vocab = wv.vocab

# We will now encode the words using the dictionary of indices created by the
# Word2Vec model, each which map to a trained embedding in the model. We will
# also create our training/testing sets.
print("Preprocessing classifier input...")
encoded_sentences = [[vocab[w].index for w in s.split(" ")] for s in train_sentences]
X_train = np.array(pad_sequences(encoded_sentences, maxlen=max_sentence_length))
y_train = np.array([(i + 1) % 2 for i in range(len(X_train))])
print(len(encoded_sentences))
print(X_train.shape)

# Our classifier will be a Keras model composed of simply:
#   An embedding input layer.
#   Two bidirectional LSTMs w/ 100 cells each.
#   A dense layer with one single output for classification.
print("Creating model...")
model = Sequential()
model.add(Embedding(weights.shape[0],
                    weights.shape[1],
                    weights=[weights],
                    mask_zero=True,
                    input_length=max_sentence_length))
model.add(Bidirectional(LSTM(128, return_sequences=True), merge_mode="sum"))
model.add(Bidirectional(LSTM(128), merge_mode="sum"))
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])

print("Model summary:")
print(model.summary())
wv.save(args.name + ".wv")

# We will set several callbacks to checkpoint the model, and monitor progress.
callbacks = [
    ModelCheckpoint(args.name + ".h5", save_best_only=True),
    TensorBoard(log_dir=f"{args.name}_logs", write_graph=False, update_freq="batch")]
if args.validation:
    test_size = int(X_train.shape[0] * args.validation_split)
    test_idx = np.random.choice(X_train.shape[0], test_size, replace=False)
    train_idx = np.ones((X_train.shape[0],), bool)
    train_idx[test_idx] = False
    X_test = X_train[test_idx]
    X_train = X_train[train_idx]
    y_test = y_train[test_idx]
    y_train = y_train[train_idx]
    callbacks.append(ReduceLROnPlateau(factor=0.5, cooldown=1, verbose=1, min_lr=0.0001))

# We will then fit the model.
print("Training model...")
model.fit(
        X_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        validation_data=(X_test, y_test) if args.validation else None,
        callbacks=callbacks)

# We then save the model, and word2vec data.
print("Saving model!")
model.save(args.modelfile)
wv.save(args.wvfile)

# We will then evaluae the model on the test data.
print("Preprocessing classifier input (Test)...")
test_encoded_sentences = [[vocab[w].index for w in s.split(" ")] for s in test_sentences]
X_test = np.array(pad_sequences(test_encoded_sentences, maxlen=max_sentence_length))

print("Running model (Test)...")
predictions = model.predict(X_test, verbose=1)

# Write the probability values for each sentence.
print("Writing Test Output...")
with open("predictions.txt", "w+") as f:
    for i in range(len(predictions)):
        f.write(str(predictions[i]))
        f.write("\n")