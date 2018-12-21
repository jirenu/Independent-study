#imports
import gensim
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
'''
Obtained from https://medium.com/@aneesha/using-tsne-to-plot-a-subset-of-similar-words-from-word2vec-bb8eeaea6229

Modified by: Jishnu Renugopal
'''


twitter_pre_trained = "/Users/JishnuRenugopal/Downloads/word2vec_twitter_model/word2vec_twitter_model.bin"
google_pre_trained = "/Users/JishnuRenugopal/Downloads/GoogleNews-vectors-negative300.bin"
def display_closestwords_tsnescatterplot(model, word, ind,dimensions=400):
    plt.figure(ind)
    arr = np.empty((0, 300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)

#vec1 onto vec2
def projection(vec1, vec2):
    mag = np.dot(vec1, vec2)/(np.linalg.norm(vec2)**2)
    return mag*vec2, np.dot(vec1, vec2)/(np.linalg.norm(vec2))

#Finds bias
def projectedOnDim(dim1, dim2, wordVec):
    dim = dim1-dim2
    vec, scalar = projection(wordVec,dim)
    return vec, scalar

def similarWordsplots(model,contexts):
    ind = 0
    for i in contexts:
        for j in i:
            if j in model.wv.vocab:
                print("yes!")
                display_closestwords_tsnescatterplot(model, j, ind)
            ind+=1
    plt.show()



def similarWords(contexts):
    pass
#Main
ind = 0
baselines = [["folks", "people", "guys"]]
model = gensim.models.KeyedVectors.load_word2vec_format(twitter_pre_trained, binary=True, unicode_errors='ignore')
contexts_guys = ["you", "u", "these", "hey", "those", "sorry"]
contexts_guys = [i+" guys" for i in contexts_guys]
dim_guys = ["man", "woman"]
contexts_folks = [i+" folks" for i in contexts_guys]
dim_folks = ["we", "you"]
contexts_people = [i+" people" for i in contexts_guys]
list_contexts = [contexts_guys, contexts_folks, contexts_people]
similarWordsplots(model,baselines)
