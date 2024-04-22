from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_simple_linear_model(input_dim, output_dim):
    model = Sequential([
        Dense(output_dim, input_shape=(input_dim,), activation='linear')
    ])
    return model