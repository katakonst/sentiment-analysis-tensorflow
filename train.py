import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import string
import codecs
import collections
import glob
import pickle
import argparse

def readFile(fileName, allWords):

    file = codecs.open(fileName, encoding='utf-8')
    for line in file:
        line = line.lower().encode('utf-8')
        words = line.split()
        for word in words:
            word = word.translate(None, string.punctuation.encode('utf-8'))
            if word != '':
                allWords.append(word)
    file.close()


def readFileToConvertWordsToIntegers(dictionary, fileName, allDocuments, allLabels, label, size):

    file = codecs.open(fileName, encoding='utf-8')
    document = []
    for line in file:
        line = line.lower().encode('utf-8')
        words = line.split()
        for word in words:
            word = word.translate(None, string.punctuation.encode('utf-8'))
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
            document.append(index)
    file.close()
    if len(document)<sentence_size:
       allDocuments.append(document)
       allLabels.append(label)


def build_dictionary(words, vocabulary_size):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0
    data.append(index)
  return dictionary

def buildDictionary(language, vocabulary_size):
    allWords = []
    fileList = glob.glob("./datasets/"+language+"/train/neg/*.txt")
    for file in fileList:
        readFile(file, allWords)

    fileList = glob.glob("./datasets/"+language+"/train/pos/*.txt")
    for file in fileList:
        readFile(file, allWords)

    fileList = glob.glob("./datasets/"+language+"/test/neg/*.txt")
    for file in fileList:
        readFile(file, allWords)

    fileList = glob.glob("./datasets/"+language+"/test/pos/*.txt")
    for file in fileList:
        readFile(file, allWords)
    dictionary = build_dictionary(allWords, vocabulary_size)
    del allWords
    pickle.dump( dictionary, open('./dictionaries/'+language+"dictionary.pickle", "wb" ) )
    return dictionary


def train(dictionary,language, sentSize):
    trainDocuments = []
    trainLabels = []
    testDocuments = []
    testLabels = []

    print("Converting train neg...")
    fileList = glob.glob("./datasets/"+lang+"/train/neg/*.txt")
    for file in fileList:
        readFileToConvertWordsToIntegers(dictionary, file, trainDocuments, trainLabels, 0,sentSize)
    print("total train: "+str(len(trainDocuments)))

    print("Converting train pos...")
    fileList = glob.glob("./datasets/"+language+"/train/pos/*.txt")
    for file in fileList:
        readFileToConvertWordsToIntegers(dictionary, file, trainDocuments, trainLabels, 1, sentSize)
    print("total train: "+str(len(trainDocuments)))

    print("Converting test neg...")
    fileList = glob.glob("./datasets/"+language+"/test/neg/*.txt")
    for file in fileList:
        readFileToConvertWordsToIntegers(dictionary, file, testDocuments, testLabels, 0, sentSize)
    print("total train: "+str(len(testDocuments)))

    print("Converting test pos...")
    fileList = glob.glob("./datasets/"+language+"/test/pos/*.txt")
    for file in fileList:
        readFileToConvertWordsToIntegers(dictionary, file, testDocuments, testLabels, 1, sentSize)
    print("total test: "+str(len(testDocuments)))

    return trainDocuments, testDocuments, trainLabels, testLabels

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train with lstm')
    parser.add_argument('language')

    args = parser.parse_args()
    lang = args.language

    sentence_size = 150
    vocabulary_size = 10000

    dictionary = buildDictionary(lang, vocabulary_size)
    trainDocuments,testDocuments,trainLabels,testLabels = train(dictionary, lang, sentence_size)

    trainX = pad_sequences(trainDocuments, maxlen=150, value=0.)
    testX = pad_sequences(testDocuments, maxlen=150, value=0.)
    trainY = to_categorical(trainLabels, nb_classes=2)
    testY = to_categorical(testLabels, nb_classes=2)

    net = tflearn.input_data([None, 150])
    net = tflearn.embedding(net, input_dim=vocabulary_size, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=32)
    model.save("./checkpoints/"+lang + "/"+lang+"tf.tfl")
