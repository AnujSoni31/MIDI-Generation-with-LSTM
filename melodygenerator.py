import json
import numpy as np
import music21 as m21
import tensorflow.keras as keras
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:
	def __init__(self, model_path="model.h5"):

		self.model_path = model_path 
		self.model = keras.models.load_model(model_path)

		with open(MAPPING_PATH, "r") as fp:
			self._mappings = json.load(fp)

		self._start_symbols = ["/"] * SEQUENCE_LENGTH

	def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
	# :param seed(str): melody piece e.g. '64 _ 63 _ _'
		
		# create seed with start symbol
		seed = seed.split() # convert to list to easily manipulate
		melody = seed
		seed = self._start_symbols + seed

		# map seed(list) to int
		seed = [self._mappings[symbol] for symbol in seed]

		for _ in range(num_steps):

			# limit the seed to max sequence length
			seed = seed[-max_sequence_length:]

			# one-hot encode the seed
			onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
			# (1, max_sequence_length, num of symbols in the vocabulary) -> single sample note
			# 1 -> batch_size
			onehot_seed = onehot_seed[np.newaxis, ...] # adds an extra dimension for 3D array

			# make a prediction
			probabilities = self.model.predict(onehot_seed)[0] # [0] -> index for probability distribution
			# [0.1, 0.2, 0.1, 0.6] -> Example probability distribution (PD)
			output_int = self._sample_with_temperature(probabilities, temperature)

			# update seed
			seed.append(output_int)

			# map int to our encoding
			output_symbol = [key for key, val in self._mappings.items() if val == output_int][0]

			# check if the end of melody -> slash (/) symbol
			if output_symbol == "/":
				break

			# else update the melody
			melody.append(output_symbol)

			return melody

	def _sample_with_temperature(self, probabilities, temperature): #  picking notes for predicted seed
		# temperature -> infinity; PD becomes homogeneous
		# temperature -> 0; highest probability is picked
		# temperature = 1; nothing changes
		predictions = np.log(probabilities) / temperature
		probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

		choices = range(len(probabilities)) # [0, 1, 2, 3]
		index = np.random.choice(choices, p=probabilities)

		return index

	def save_melody(self, melody, step_duration=0.25, format="midi", file_name="C:/Users/anujs/Music/mel.mid"):
		# create a music21 stream
		stream = m21.stream.Stream() # THIS IS THE ONLY ISSUE NOW

		# parse all the symbols in the melody and create note/rest objects
		start_symbol = None
		step_counter = 1

		for i, symbol in enumerate(melody):
			# handle case in which we have a note/rest
			if symbol != "_" or i+1==len(melody):
				# ensure we're dealing with note/rest beyond first one
				if start_symbol is not None:
					quarter_length_duration = step_duration * step_counter

					# handle rest
					if start_symbol == "r":
						m21_event = m21.note.Rest(quarterLength = quarter_length_duration)

					# handle note
					else:
						m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

					stream.append(m21_event)

					# reset the step counter
					step_counter = 1

				start_symbol = symbol

			# handle case in which we have a prolongation "_"
			else:
				step_counter += 1

		# write the m21 stream to a midi file
		stream.write(format, file_name)

if __name__ == '__main__':
	mg = MelodyGenerator()
	seed = input("Enter a sample seed: ")
	seed = """69 _ _ _ 69 _ _ _ _ _ _ _ 72 _ _ _ 71 _ _ _ _ _ _ _ 69 _ _ _ 71 _ _ _ _ _ 72 _ _ _ 
	69 _ _ _ 69 _ _ _ _ _ _ _ 72 _ _ _ 71 _ _ _ _ _ _ _ 69 _ _ _ 71 _ _ _ _ _ 72 _ _ _ 
	69 _ _ _ 69 _ _ _ _ _ _ _ 72 _ _ _ 71 _ _ _ _ _ _ _ 69 _ _ _ 71 _ _ _ _ _ 72 _ _ _ 
	69 _ _ _ 69 _ _ _ _ _ _ _ 72 _ _ _ 71 _ _ _ _ _ _ _ 69 _ _ _ 71 _ _ _ _ _ 72 _ _ _ 
	69 _ _ _ 69 _ _ _ _ _ _ _ 72 _ _ _ 71 _ _ _ _ _ _ _ 69 _ _ _ 71 _ _ _ _ _ 72 _ _ _ 
	69 _ _ _ 69 _ _ _ _ _ _ _ 72 _ _ _ 71 _ _ _ _ _ _ _ 69 _ _ _ 71 _ _ _ _ _ 72 _ _ _ """ # example seed -> 214
	melody = mg.generate_melody(seed, 600, SEQUENCE_LENGTH, 0.8)
	print(melody)
	mg.save_melody(melody)