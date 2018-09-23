from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
import coremltools

np.random.seed(1337)  # for reproducibility

mapping = {
    "terrible": 1,
    "great": 2,
    "bad": 3,
    "good": 4,
    "awful": 5,
    "awesome": 6
}

# max number of words we're mapping plus the unknown character
max_features = len(mapping) + 1

trainSentences = [
    "that was great",
    "that movie is terrible",
    "that movie was bad",
    "that movie is terrible and it was really bad",
    "that was awesome and great",
    "wow that was so good it was great",
    "that was good",
    "wow you are awful"
]

trainSentiment = [1, 0, 0, 0, 1, 1, 1, 0]

testSentences = [
    "wow how good was that",
    "i am feeling great",
    "it is terrible to feel bad",
    "i feel really awful today"
]

testSentiment = [1, 1, 0, 0]

def convert(model):
	coreml_model = coremltools.converters.keras.convert(model, input_names=['tokenizedString'], output_names=['sentiment'])
	coreml_model.author = 'Evan Compton'
	coreml_model.license = 'MIT'
	coreml_model.short_description = 'Gets the sentiment based on a tokenized string'
	coreml_model.input_description['tokenizedString'] = 'A String mapped according to the pre-deifned mapping'
	coreml_model.output_description['sentiment'] = 'Whether the sentence was positive or negative'
	coreml_model.save('sentiment_model.mlmodel')

def tokenize(sent):
    words = sent.split(' ')
    tokenized = []
    for word in words:
        if mapping.get(word) != None:
            tokenized.append(mapping[word])
        else:
            tokenized.append(0)
    return tokenized


print('Creating data...')
tokenizedTrain = []
for sent in trainSentences:
    tokenizedTrain.append(tokenize(sent))

X_train = np.array(tokenizedTrain)
y_train = np.array(trainSentiment)


tokenizedTest = []
for sent in testSentences:
    tokenizedTest.append(tokenize(sent))

X_test = np.array(tokenizedTest)
y_test = np.array(testSentiment)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
maxlen = 80  # max length of a sentence
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen, dropout=0.2))
model.add(GRU(128, recurrent_dropout=0.2, dropout=0.2))
model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
print(X_train.shape)
print(y_train.shape)
batch_size = 2
model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)

print('------------------------')
print(model.predict(sequence.pad_sequences(np.array([[0, 0, 0, 2]]), maxlen=maxlen)))
model.save('sentiment_model.h5')

convert(model)
