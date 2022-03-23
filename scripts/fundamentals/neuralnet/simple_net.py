# Imports
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K

class SimpleNet:
	@staticmethod
	# Build simple model
	def build_model(width, height, depth, classes):
		# Initialize model as sequential
		model = Sequential()
		# Format shape into tuple
		input_shape = (height, width, depth)

		# This a simple CNN... so let's keep it real simple
		# Add the only CONV => RELU layer
		# The 32, (3, 3) below says have 32 convultion kernels with a 3x3 size
		# "same" means the output layer H and W will be same as input
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=input_shape))
		# Add relu activation
		model.add(Activation("relu"))

		# Add softmax for classification
		# Need to flatten and add dense (FC) layer prior
		model.add(Flatten())
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# Return model
		return model