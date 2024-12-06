import numpy as np
import pandas as pd
import time
import json
import gc

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

def clip_gradients(gradients, max_norm=5.0):
    """Clip gradients to prevent exploding gradients"""
    total_norm = 0
    for grad in gradients:
        total_norm += np.sum(np.square(grad))
    total_norm = np.sqrt(total_norm)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for i in range(len(gradients)):
            gradients[i] = gradients[i] * clip_coef
    return gradients

class CharacterPredictor:
    def __init__(self, input_size=199, hidden_layers=[128, 64], output_size=74, 
                 alpha=0.01, batch_size=32, epochs=100):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.initial_alpha = alpha
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = 0.0001
        self.momentum = 0.9
        self.velocity_weights = None
        self.velocity_biases = None
        self.validation_losses = []
        
        # Initialize weights with the correct input size
        input_features = input_size * 74  # input_size * vocab_size
        self.weights, self.biases = init_weights(
            input_features,
            hidden_layers,
            output_size
        )
        
        # Initialize velocity arrays for momentum
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]
        
    def prepare_sequence(self, text, char_to_idx, training=True):
        """Prepare input sequences and target outputs using chunked processing"""
        print("Preparing sequences...")
        start_time = time.time()
        last_check_time = start_time
        last_check_sequences = 0
        
        sequences_to_process = len(text) - self.input_size if training else 1
        vocab_size = len(char_to_idx)
        
        # Process in smaller chunks to manage memory
        chunk_size = 10000
        num_chunks = (sequences_to_process + chunk_size - 1) // chunk_size
        
        # Create lookup array for faster char to index conversion
        char_to_idx_arr = np.zeros(256, dtype=np.int32)
        for char, idx in char_to_idx.items():
            char_to_idx_arr[ord(char)] = idx
        
        # Create temporary directory for chunks
        import tempfile
        import os
        temp_dir = tempfile.mkdtemp()
        chunk_files_X = []
        chunk_files_y = []
        
        try:
            for chunk in range(num_chunks):
                chunk_start = chunk * chunk_size
                chunk_end = min((chunk + 1) * chunk_size, sequences_to_process)
                chunk_size_actual = chunk_end - chunk_start
                
                # Pre-allocate arrays for this chunk
                X = np.zeros((chunk_size_actual, self.input_size, vocab_size), dtype=np.float32)
                y = np.zeros((chunk_size_actual, vocab_size), dtype=np.float32) if training else None
                
                # Process sequences in this chunk
                for i in range(chunk_size_actual):
                    if (chunk_start + i) % 10000 == 0:
                        current_time = time.time()
                        sequences_processed = chunk_start + i
                        
                        # Calculate speed based on last check point
                        time_delta = current_time - last_check_time
                        sequences_delta = sequences_processed - last_check_sequences
                        
                        if sequences_processed == 0 or time_delta < 0.001:
                            print(f"Processing sequence {sequences_processed}/{sequences_to_process} "
                                  f"({(sequences_processed/sequences_to_process)*100:.1f}%) - "
                                  f"Calculating speed...")
                        else:
                            speed = sequences_delta / time_delta
                            remaining_sequences = sequences_to_process - sequences_processed
                            remaining_time = remaining_sequences / speed
                            
                            remaining_minutes = int(remaining_time // 60)
                            remaining_seconds = int(remaining_time % 60)
                            
                            print(f"Processing sequence {sequences_processed}/{sequences_to_process} "
                                  f"({(sequences_processed/sequences_to_process)*100:.1f}%) - "
                                  f"Speed: {speed:.0f} sequences/sec - "
                                  f"Est. remaining time: {remaining_minutes}m {remaining_seconds}s")
                        
                        last_check_time = current_time
                        last_check_sequences = sequences_processed
                    
                    # Get sequence indices using vectorized operation
                    text_idx = chunk_start + i
                    sequence = np.frombuffer(text[text_idx:text_idx + self.input_size].encode(), dtype=np.uint8)
                    sequence_indices = char_to_idx_arr[sequence]
                    
                    # Set one-hot encodings
                    X[i, np.arange(self.input_size), sequence_indices] = 1
                    
                    if training:
                        target_idx = char_to_idx_arr[ord(text[text_idx + self.input_size])]
                        y[i, target_idx] = 1
                
                # Save chunk to disk
                x_filename = os.path.join(temp_dir, f'chunk_X_{chunk}.npy')
                np.save(x_filename, X)
                chunk_files_X.append(x_filename)
                
                if training:
                    y_filename = os.path.join(temp_dir, f'chunk_y_{chunk}.npy')
                    np.save(y_filename, y)
                    chunk_files_y.append(y_filename)
                
                # Clear memory
                del X
                if training:
                    del y
                gc.collect()
            
            # Load and concatenate chunks in smaller groups
            print("\nCombining chunks...")
            group_size = 5
            final_X = []
            final_y = [] if training else None
            
            for i in range(0, len(chunk_files_X), group_size):
                group_X = [np.load(f) for f in chunk_files_X[i:i+group_size]]
                final_X.append(np.concatenate(group_X, axis=0))
                
                if training:
                    group_y = [np.load(f) for f in chunk_files_y[i:i+group_size]]
                    final_y.append(np.concatenate(group_y, axis=0))
                
                # Clear group memory
                del group_X
                if training:
                    del group_y
                gc.collect()
            
            X = np.concatenate(final_X, axis=0)
            y = np.concatenate(final_y, axis=0) if training else None
        
        finally:
            # Clean up temporary files
            for f in chunk_files_X + chunk_files_y:
                try:
                    os.remove(f)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
        
        return X, y
    
    def fit(self, X, y, validation_split=0.1):
        """Train the network"""
        print(f"Starting training on {X.shape[0]} samples...")
        
        # More frequent batch progress updates
        print_interval = max(1, (X.shape[0] // self.batch_size) // 50)  # Print ~50 times per epoch
        
        # Split data into training and validation sets
        split_idx = int(X.shape[0] * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        m = X_train.shape[0]
        
        for epoch in range(self.epochs):
            # Learning rate decay
            self.alpha = self.initial_alpha / (1 + epoch * 0.1)
            
            epoch_loss = 0
            batch_count = 0
            current_loss = float('inf')
            
            print(f"\nEpoch {epoch + 1}/{self.epochs} - Learning Rate: {self.alpha:.6f}")
            
            # Mini-batch training
            for i in range(0, m, self.batch_size):
                batch_X = X_train[i:i + self.batch_size]
                batch_y = y_train[i:i + self.batch_size]
                
                # Forward pass
                Z_values, activations = forward(batch_X, self.weights, self.biases)
                
                # Calculate batch loss before updates
                batch_loss = np.mean((activations[-1] - batch_y) ** 2)
                epoch_loss += batch_loss
                batch_count += 1
                current_loss = epoch_loss / batch_count
                
                if batch_count % print_interval == 0:
                    progress = (i / m) * 100
                    print(f"Batch {batch_count}: {progress:.1f}% complete - Current loss: {current_loss:.4f}")
                
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
                
                # Clip gradients
                dW_list = clip_gradients(dW_list)
                db_list = clip_gradients(db_list)
                
                # Update weights and biases with momentum and L2 regularization
                for layer in range(len(self.weights)):
                    # Add L2 regularization gradient
                    dW_list[layer] += self.weight_decay * self.weights[layer]
                    
                    # Apply momentum
                    self.velocity_weights[layer] = (self.momentum * self.velocity_weights[layer] - 
                                                  self.alpha * dW_list[layer] / self.batch_size)
                    self.velocity_biases[layer] = (self.momentum * self.velocity_biases[layer] - 
                                                 self.alpha * db_list[layer] / self.batch_size)
                    
                    self.weights[layer] += self.velocity_weights[layer]
                    self.biases[layer] += self.velocity_biases[layer]
            
            # Calculate validation loss every epoch
            val_Z_values, val_activations = forward(X_val, self.weights, self.biases)
            val_loss = np.mean((val_activations[-1] - y_val) ** 2)
            self.validation_losses.append(val_loss)
            
            avg_epoch_loss = epoch_loss / batch_count
            print(f"Epoch {epoch + 1}/{self.epochs} complete - Train Loss: {avg_epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    def predict(self, text, char_to_idx, idx_to_char):
        """Make predictions for a text input"""
        # Take last input_size characters if text is too long
        if len(text) > self.input_size:
            text = text[-self.input_size:]
        
        # Create one-hot encoding for single sequence
        vocab_size = len(char_to_idx)
        X = np.zeros((1, self.input_size, vocab_size))
        
        # Fill in one-hot encoding
        for i, char in enumerate(text):
            if char in char_to_idx:
                X[0, i, char_to_idx[char]] = 1
        
        # Make prediction
        Z_values, activations = forward(X, self.weights, self.biases)
        
        # Get indices of top 5 most likely characters
        predictions = activations[-1]
        top_k_indices = np.argsort(predictions[0])[-5:][::-1]
        
        # Create list of (character, probability) pairs
        char_probs = []
        for idx in top_k_indices:
            char = idx_to_char[idx]
            prob = predictions[0][idx]
            char_probs.append((char, prob))
        
        return char_probs
    
    def save_weights(self, filename='neuralNetworkWeights.json'):
        """Save weights and biases to a JSON file"""
        weights_data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'architecture': {
                'input_size': self.input_size,
                'hidden_layers': self.hidden_layers,
                'output_size': self.output_size
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(weights_data, f)
        print(f"Weights saved to {filename}")
    
    def load_weights(self, filename='neuralNetworkWeights.json'):
        """Load weights and biases from a JSON file"""
        with open(filename, 'r') as f:
            weights_data = json.load(f)
        
        # Verify architecture matches
        arch = weights_data['architecture']
        if (arch['input_size'] != self.input_size or 
            arch['hidden_layers'] != self.hidden_layers or 
            arch['output_size'] != self.output_size):
            raise ValueError("Model architecture in file doesn't match current model")
        
        # Convert lists back to numpy arrays
        self.weights = [np.array(w) for w in weights_data['weights']]
        self.biases = [np.array(b) for b in weights_data['biases']]
        
        # Reinitialize velocity arrays
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]
        print(f"Weights loaded from {filename}")

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
    hidden_layers=[256, 256, 256],  
    output_size=74,
    alpha=0.001, 
    epochs=5,   
    batch_size=256  
)
# Load pre-trained weights or train based on training flag
training = True  # Set to True to train new model, False to load existing weights

if not training:
    try:
        predictor.load_weights()
        
        print("Making predictions...")
        test_text = "You are a joke teller. You are a joke teller. You are a joke teller. You are a joke teller. You tell jokes, Okay start!!!!: What do you call a redditor that doesn't use the search button in /r/jokes?"
        current_text = test_text
        
        print("\nInput:")
        print(test_text)
        print("\nGenerating outputs...")

        # Generate text
        generated_text_greedy = ""
        generated_text_weighted = ""
        current_text_greedy = current_text
        current_text_weighted = current_text

        for i in range(400):
            predictions_greedy = predictor.predict(current_text_greedy[-199:], char_to_idx, idx_to_char)
            predictions_weighted = predictor.predict(current_text_weighted[-199:], char_to_idx, idx_to_char)
            
            # Greedy approach
            next_char_greedy = predictions_greedy[0][0]
            
            # Weighted random approach
            top_3_chars, top_3_probs = zip(*predictions_weighted[:3])
            top_3_probs = np.array(top_3_probs)
            top_3_probs = top_3_probs ** 2
            top_3_probs = top_3_probs / np.sum(top_3_probs)
            next_char_weighted = np.random.choice(top_3_chars, p=top_3_probs)
            
            generated_text_greedy += next_char_greedy
            generated_text_weighted += next_char_weighted
            current_text_greedy = current_text_greedy + next_char_greedy
            current_text_weighted = current_text_weighted + next_char_weighted

        # Print final outputs only
        print("\nGreedy Output (most likely):")
        print(generated_text_greedy)
        print("\nWeighted Random Output (probability^2 weighted):")
        print(generated_text_weighted)

    except FileNotFoundError:
        print("No pre-trained weights found. Preparing sequences for training...")
        X, y = predictor.prepare_sequence(text, char_to_idx)
        predictor.fit(X, y)
        predictor.save_weights()
else:
    print("Training new model...")
    X, y = predictor.prepare_sequence(text, char_to_idx)
    predictor.fit(X, y)
    predictor.save_weights()