import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np 
import tflearn
import tensorflow
import random as rd
import json
import pickle


with open("intents.json") as file:
	data = json.load(file)
    
try:
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)
except:
    

    feature_vectors = []
    target_labels = []
    categories = []
    stemmed_words = []
    for i in data["intents"]:
        for pattern in i["patterns"]:
            words_temp = pattern.split()
            for words in words_temp:
                stemmed_words.append(words)

            temp = pattern.split()
            feature_vectors.append(temp)
            target_labels.append(i["tag"])
        
      
    #for categories
    for i in data["intents"]:
        categories.append(i["tag"])        

    words = list(stemmer.stem(w.lower()) for w in stemmed_words if w not in "?")
    words = sorted(list(set(words)))
    categories = sorted(categories)

    training = []
    output = []

    for i,fv in enumerate(feature_vectors):

        bow = list()
        wrds = [stemmer.stem(w) for  w in fv]
        for w in words:
            if w in wrds:
                bow.append(1)
            else:
                bow.append(0)

        output_row = [0 for _ in range(len(categories))]
        output_row[categories.index(target_labels[i])] = 1
        #Converting the output into one hot encoding format
        training.append(bow)
        #the bag of words(representing each stemmed word as 0 or 1, based on whether it is present in the 
        #feature vector or not)
        output.append(output_row)
    
    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
             pickle.dump((words, categories, training, output), f)     
    #Data is properly structured now for machine learning.

def bag_of_words(input_string,words):
    #Every Input will be converted into Bag of Words.
    #Then, based on the model, the Neural Network will predict the proper response.
    iss = input_string.split()
    input_string = list(stemmer.stem(w.lower()) for w in iss)
    bagofwords = [ 0 for _ in range(len(words)) ]
    for i in input_string:
        if i in words:
            bagofwords[words.index(i)] = 1
    return np.array(bagofwords)

#training the data; Can be adjusted or tweaked later.
net = tflearn.input_data(shape = [None, training.shape[1]])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size = 8, show_metric=True)
model.save("model.tflearn")

def ai_chat():
	print("Hi, I am an automated ITU chat bot, ask me anything (type quit to stop)!")
	print("Consult itu.edu.pk if i fail to address your queries.")
	while True:
		inp = input("You: ")
		if inp.lower() == "quit":
			break
		results = model.predict([bag_of_words(inp, words)])
		results_index = np.argmax(results) #The category with the highest probability
		tag = categories[results_index]


		for tg in data["intents"]:
			if tg['tag'] == tag:
				responses = tg['responses']
                #Generate any random response.
                #Random responses will give a more humanlike experience.
                
		print(rd.choice(responses)) #AI giving response

ai_chat()




