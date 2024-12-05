import numpy as np
import pandas as pd

# Character mapping utilities
def create_char_mappings(text):
    """Create character to index and index to character mappings"""
    unique_chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
    return char_to_idx, idx_to_char

# Activation functions
class Tanh:
    @staticmethod
    def activation(z):
        return np.tanh(z)
    
    @staticmethod
    def derivative(z):
        tanh_z = np.tanh(z)
        return 1 - tanh_z ** 2

class Linear:
    @staticmethod
    def activation(z):
        return z
    
    @staticmethod
    def derivative(z):
        return np.ones_like(z)

def init_weights(n_input_features, hidden_layers, output_neurons):
    """Initialize weights for the neural network with multiple hidden layers
    hidden_layers: list of integers representing neurons in each hidden layer"""
    weights = []
    biases = []
    
    # Input layer to first hidden layer
    prev_size = n_input_features
    for layer_size in hidden_layers:
        W = np.random.uniform(-0.5, 0.5, size=(prev_size, layer_size))
        b = np.ones((1, layer_size))
        weights.append(W)
        biases.append(b)
        prev_size = layer_size
    
    # Last hidden layer to output layer
    W_out = np.random.uniform(-0.5, 0.5, size=(prev_size, output_neurons))
    b_out = np.ones((1, output_neurons))
    weights.append(W_out)
    biases.append(b_out)
    
    return weights, biases

def forward(X, weights, biases):
    """Forward pass through multiple layers"""
    activations = [X]
    Z_values = []
    
    # Through all but the last layer
    for i in range(len(weights) - 1):
        Z = np.dot(activations[-1], weights[i]) + biases[i]
        Z_values.append(Z)
        A = Tanh.activation(Z)
        activations.append(A)
    
    # Output layer
    Z_out = np.dot(activations[-1], weights[-1]) + biases[-1]
    Z_values.append(Z_out)
    A_out = Linear.activation(Z_out)
    activations.append(A_out)
    
    return Z_values, activations

class CharacterPredictor:
    def __init__(self, input_size=199, hidden_layers=[128, 64, 32], output_size=1, 
                 alpha=0.01, batch_size=32, epochs=100):
        self.input_size = input_size
        self.hidden_layers = hidden_layers  # Now a list of layer sizes
        self.output_size = output_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Initialize weights for all layers
        self.weights, self.biases = init_weights(
            input_size, hidden_layers, output_size
        )
        
    def prepare_sequence(self, text, char_to_idx, training=True):
        """Prepare input sequences and target outputs"""
        print(f"Preparing sequences from text of length {len(text)}...")
        X = []
        y = []
        
        # For prediction, we only need one sequence if text is shorter than input_size
        if len(text) < self.input_size:
            if not training:
                # Pad the text if it's too short
                text = text.ljust(self.input_size, ' ')
        
        # Process sequences
        sequences_to_process = len(text) - self.input_size if training else 1
        for i in range(sequences_to_process):
            sequence = text[i:i + self.input_size]
            X_sequence = [char_to_idx.get(char, 0) for char in sequence]  # Use get() with default value
            X.append(X_sequence)
            
            if training:
                target = text[i + self.input_size]
                y_target = char_to_idx[target]
                y_one_hot = np.zeros(len(char_to_idx))
                y_one_hot[y_target] = 1
                y.append(y_one_hot)
        
        print(f"Created {len(X)} sequences")
        return np.array(X), np.array(y) if training else None
    
    def fit(self, X, y):
        """Train the network"""
        print(f"Starting training on {X.shape[0]} samples...")
        m = X.shape[0]
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            batch_count = 0
            
            # Mini-batch training
            for i in range(0, m, self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]
                
                # Forward pass
                Z_values, activations = forward(batch_X, self.weights, self.biases)
                
                # Backward pass
                dZ = activations[-1] - batch_y  # Error at output layer
                dW_list = []
                db_list = []
                
                # Handle output layer first
                dW = (1/m) * np.dot(activations[-2].T, dZ)
                db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
                dW_list.insert(0, dW)
                db_list.insert(0, db)
                
                # Handle hidden layers
                for layer in range(len(self.hidden_layers), 0, -1):
                    dA = np.dot(dZ, self.weights[layer].T)
                    dZ = dA * Tanh.derivative(Z_values[layer-1])
                    dW = (1/m) * np.dot(activations[layer-1].T, dZ)
                    db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
                    dW_list.insert(0, dW)
                    db_list.insert(0, db)
                    
                # Update weights and biases
                for layer in range(len(self.weights)):
                    self.weights[layer] -= self.alpha * dW_list[layer]
                    self.biases[layer] -= self.alpha * db_list[layer]
                
                epoch_loss += np.mean((activations[-1] - batch_y) ** 2)
                batch_count += 1
            
            avg_epoch_loss = epoch_loss / batch_count
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Average Loss: {avg_epoch_loss:.4f}")
    
    def predict(self, text, char_to_idx, idx_to_char):
        """Make predictions for a text input"""
        X, _ = self.prepare_sequence(text, char_to_idx, training=False)
        Z_values, activations = forward(X, self.weights, self.biases)
        
        # Convert predictions to characters
        predictions = activations[-1]
        # Get indices of top 5 most likely characters
        top_k_indices = np.argsort(predictions[0])[-5:][::-1]
        
        # Create list of (character, probability) pairs
        char_probs = []
        for idx in top_k_indices:
            char = idx_to_char[idx]
            prob = predictions[0][idx]
            char_probs.append((char, prob))
        
        return char_probs

# Add prints for the main execution
print("Loading text data...")
with open('data/redditJokesProcessed.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"Loaded {len(text)} characters")

print("Creating character mappings...")
char_to_idx, idx_to_char = create_char_mappings(text)
print(f"Found {len(char_to_idx)} unique characters")

print("Initializing predictor...")
predictor = CharacterPredictor(
    input_size=199,
    hidden_layers=[128, 64, 32],
    output_size=len(char_to_idx),
    alpha=0.01,
    batch_size=32,
    epochs=2
)

# Prepare sequences
X, y = predictor.prepare_sequence(text, char_to_idx)

# Train the network
predictor.fit(X, y)

print("Making predictions...")
test_text = "What do you call a redditor that doesn't use the search button in /r/jokes?"
predictions = predictor.predict(test_text, char_to_idx, idx_to_char)
print("\nTop 5 predicted next characters:")
for char, prob in predictions:
    print(f"'{char}': {prob:.4f}")