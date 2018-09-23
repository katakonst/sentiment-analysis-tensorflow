import tflearn
import string
import pickle
import argparse
import numpy as nm

def convertTextToIndex(dictionary, text):
    document = []
    text = text.lower().encode('utf-8')
    words = text.split()
    for word in words:
        word = word.translate(None, string.punctuation.encode('utf-8'))
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
        document.append(index)

    ln = 150 - len(document)
    if ln>0 :
        document = nm.pad(document, (0, ln), 'constant')
    return document

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train with lstm')
    parser.add_argument('language')
    parser.add_argument('text')

    args = parser.parse_args()
    lang = args.language
    text = args.text
    f = open('./dictionaries/'+lang+'dictionary.pickle', 'rb')
    dictionary = pickle.load(f)
    f.close()
    net = tflearn.input_data([None, 150])
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                                 loss='categorical_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.load("checkpoints/"+lang+"/"+lang+"tf.tfl")
    result = model.predict([convertTextToIndex(dictionary, text)])
    print("negative="+str(result[0][0]))
    print("positive="+ str(result[0][1]))