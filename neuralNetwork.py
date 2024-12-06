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
    """Initialize weights for the neural network with multiple hidden layers"""
    weights = []
    biases = []
    
    # First layer needs special handling for the flattened input
    W = np.random.uniform(-0.5, 0.5, size=(n_input_features, hidden_layers[0])) / np.sqrt(n_input_features)
    b = np.zeros((1, hidden_layers[0]))
    weights.append(W)
    biases.append(b)
    
    # Hidden layers
    for i in range(1, len(hidden_layers)):
        W = np.random.uniform(-0.5, 0.5, size=(hidden_layers[i-1], hidden_layers[i])) / np.sqrt(hidden_layers[i-1])
        b = np.zeros((1, hidden_layers[i]))
        weights.append(W)
        biases.append(b)
    
    # Output layer
    W = np.random.uniform(-0.5, 0.5, size=(hidden_layers[-1], output_neurons)) / np.sqrt(hidden_layers[-1])
    b = np.zeros((1, output_neurons))
    weights.append(W)
    biases.append(b)
    
    return weights, biases

def forward(X, weights, biases):
    """Forward pass through multiple layers"""
    batch_size = X.shape[0]
    X_flat = X.reshape(batch_size, -1)  # Flatten the input
    
    activations = [X_flat]
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
    def __init__(self, input_size=199, hidden_layers=[128, 64], output_size=74, 
                 alpha=0.01, batch_size=32, epochs=100):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Initialize weights with the correct input size
        input_features = input_size * 74  # input_size * vocab_size
        self.weights, self.biases = init_weights(
            input_features,
            hidden_layers,
            output_size
        )
        
    def prepare_sequence(self, text, char_to_idx, training=True):
        """Prepare input sequences and target outputs"""
        print(f"Preparing sequences from text of length {len(text)}...")
        X = []
        y = []
        
        vocab_size = len(char_to_idx)
        sequences_to_process = len(text) - self.input_size if training else 1
        
        # Add progress tracking
        update_interval = max(1, sequences_to_process // 20)
        
        for i in range(sequences_to_process):
            if i % update_interval == 0:
                progress = (i / sequences_to_process) * 100
                print(f"Processing sequences: {progress:.1f}% complete")
            
            sequence = text[i:i + self.input_size]
            
            # One-hot encode each character in the sequence
            X_sequence = np.zeros((self.input_size, vocab_size))
            for j, char in enumerate(sequence):
                char_idx = char_to_idx.get(char, 0)
                X_sequence[j, char_idx] = 1
            
            X.append(X_sequence)
            
            if training:
                target = text[i + self.input_size]
                y_target = char_to_idx[target]
                y_one_hot = np.zeros(vocab_size)
                y_one_hot[y_target] = 1
                y.append(y_one_hot)
        
        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y) if training else None
        
        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape}, X memory: {X.nbytes / 1e9:.2f} GB")
        if y is not None:
            print(f"y shape: {y.shape}, y memory: {y.nbytes / 1e9:.2f} GB")
        
        return X, y
    
    def fit(self, X, y):
        """Train the network"""
        print(f"Starting training on {X.shape[0]} samples...")
        m = X.shape[0]
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            batch_count = 0
            
            # Add progress tracking for batches
            total_batches = m // self.batch_size
            update_interval = max(1, total_batches // 10)
            
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # Mini-batch training
            for i in range(0, m, self.batch_size):
                if batch_count % update_interval == 0:
                    progress = (batch_count / total_batches) * 100
                    print(f"Batch progress: {progress:.1f}% - Loss: {epoch_loss/(batch_count+1e-10):.4f}")
                
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]
                
                # Forward pass
                Z_values, activations = forward(batch_X, self.weights, self.biases)
                
                # Backward pass
                dZ = activations[-1] - batch_y
                dW_list = []
                db_list = []
                
                # Output layer
                dW = np.dot(activations[-2].T, dZ)
                db = np.sum(dZ, axis=0, keepdims=True)
                dW_list.insert(0, dW)
                db_list.insert(0, db)
                
                # Hidden layers
                for layer in range(len(self.hidden_layers), 0, -1):
                    dA = np.dot(dZ, self.weights[layer].T)
                    dZ = dA * Tanh.derivative(Z_values[layer-1])
                    
                    if layer > 1:
                        dW = np.dot(activations[layer-1].T, dZ)
                    else:
                        dW = np.dot(activations[0].T, dZ)
                        
                    db = np.sum(dZ, axis=0, keepdims=True)
                    dW_list.insert(0, dW)
                    db_list.insert(0, db)
                
                # Update weights and biases
                for layer in range(len(self.weights)):
                    self.weights[layer] -= self.alpha * dW_list[layer] / self.batch_size
                    self.biases[layer] -= self.alpha * db_list[layer] / self.batch_size
                
                epoch_loss += np.mean((activations[-1] - batch_y) ** 2)
                batch_count += 1
            
            avg_epoch_loss = epoch_loss / batch_count
            print(f"Epoch {epoch + 1}/{self.epochs} complete - Average Loss: {avg_epoch_loss:.4f}")
    
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
    hidden_layers=[128, 64],
    output_size=74,
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