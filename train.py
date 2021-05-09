import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH

OUTPUT_UNITS = 38
LOSS = "sparse_categorical_crossentropy"
NUM_UNITS = [256] # hidden layer neurons
'''
num_units is a list containing num of neurons in each layer
Ex. [256, 256] -> two LSTM layers with 256 neurons each
'''
LEARNING_RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 64 # no. of samples 
SAVE_MODEL_PATH = "model.h5" # *.h5 is the extension

def build_model(output_units, loss, num_units, learning_rate):
	'''
	:param: output_units: no. of neurons in the output layer
	:param: loss: Loss function
	'''
	# create model architecture
	Input = keras.layers.Input(shape=(None, output_units))
	'''
	:param: None: enables us to have any number of time stamps (sequences)
	:param: output_units: vocabulary size no. of neuron in output layer
	'''
	x = keras.layers.LSTM(num_units[0])(Input)
	x = keras.layers.Dropout(0.2)(x)

	output = keras.layers.Dense(output_units, activation="softmax")(x)

	model = keras.Model(Input, output)

	# compile model
	model.compile(loss=loss,
		optimizer=keras.optimizers.Adam(lr=learning_rate),
		metrics=["accuracy"])

	model.summary() # print the details of the model

	return model

def train():

	# generate the training sequences
	inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

	# build the network model
	model = build_model(output_units=OUTPUT_UNITS, loss=LOSS, num_units=NUM_UNITS, learning_rate=LEARNING_RATE)

	# train the model
	model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

	# save the model
	model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
	train()