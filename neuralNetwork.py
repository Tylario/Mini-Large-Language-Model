import numpy as np
import pandas as pd
import time
import json
import gc

# Character mapping utilities
def create_char_mappings(text):
    """Create character to index and index to character mappings for encoding/decoding text"""
    unique_chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
    return char_to_idx, idx_to_char

# Activation functions
class ReLU:
    @staticmethod
    def activation(z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    @staticmethod
    def derivative(z):
        """Derivative of ReLU activation function"""
        return np.where(z > 0, 1, 0)

class Linear:
    @staticmethod
    def activation(z):
        """Linear activation function (identity)"""
        return z
    
    @staticmethod
    def derivative(z):
        """Derivative of linear activation function"""
        return np.ones_like(z)

def init_weights(n_input_features, hidden_layers, output_neurons):
    """Initialize weights and biases for all layers using Xavier initialization"""
    weights = []
    biases = []
    
    # Input layer weights
    W = np.random.uniform(-0.5, 0.5, size=(n_input_features, hidden_layers[0])) / np.sqrt(n_input_features)
    b = np.zeros((1, hidden_layers[0]))
    weights.append(W)
    biases.append(b)
    
    # Hidden layer weights
    for i in range(1, len(hidden_layers)):
        W = np.random.uniform(-0.5, 0.5, size=(hidden_layers[i-1], hidden_layers[i])) / np.sqrt(hidden_layers[i-1])
        b = np.zeros((1, hidden_layers[i]))
        weights.append(W)
        biases.append(b)
    
    # Output layer weights
    W = np.random.uniform(-0.5, 0.5, size=(hidden_layers[-1], output_neurons)) / np.sqrt(hidden_layers[-1])
    b = np.zeros((1, output_neurons))
    weights.append(W)
    biases.append(b)
    
    return weights, biases

def forward(X, weights, biases):
    """Forward propagation through the network
    Returns intermediate Z values and activations for backpropagation"""
    batch_size = X.shape[0]
    X_flat = X.reshape(batch_size, -1)  # Flatten input
    
    activations = [X_flat]
    Z_values = []
    
    # Hidden layers with ReLU activation
    for i in range(len(weights) - 1):
        Z = np.dot(activations[-1], weights[i]) + biases[i]
        Z_values.append(Z)
        A = ReLU.activation(Z)
        activations.append(A)
    
    # Output layer with linear activation
    Z_out = np.dot(activations[-1], weights[-1]) + biases[-1]
    Z_values.append(Z_out)
    A_out = Linear.activation(Z_out)
    activations.append(A_out)
    
    return Z_values, activations

def clip_gradients(gradients, max_norm=5.0):
    """Clip gradients to prevent exploding gradients by scaling if norm exceeds threshold"""
    total_norm = 0
    for grad in gradients:
        total_norm += np.sum(np.square(grad))
    total_norm = np.sqrt(total_norm)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for i in range(len(gradients)):
            gradients[i] = gradients[i] * clip_coef
    return gradients

def sequence_generator(text, batch_size, input_size, char_to_idx):
    """Generator that yields batches of one-hot encoded sequences and targets
    Optimized for memory efficiency by processing text directly"""
    vocab_size = len(char_to_idx)
    n_sequences = len(text) - input_size
    
    # Fast char to index lookup array
    char_to_idx_arr = np.zeros(256, dtype=np.int32)
    for char, idx in char_to_idx.items():
        char_to_idx_arr[ord(char)] = idx
    
    indices = np.arange(n_sequences)
    
    while True:
        np.random.shuffle(indices)
        
        for i in range(0, n_sequences, batch_size):
            batch_indices = indices[i:i + batch_size]
            actual_batch_size = len(batch_indices)
            
            X_batch = np.zeros((actual_batch_size, input_size, vocab_size), dtype=np.float32)
            y_batch = np.zeros((actual_batch_size, vocab_size), dtype=np.float32)
            
            for j, idx in enumerate(batch_indices):
                # Convert sequence to indices efficiently
                sequence = np.frombuffer(text[idx:idx + input_size].encode(), dtype=np.uint8)
                sequence_indices = char_to_idx_arr[sequence]
                
                # Set one-hot encodings
                X_batch[j, np.arange(input_size), sequence_indices] = 1
                
                # Set target
                target_idx = char_to_idx_arr[ord(text[idx + input_size])]
                y_batch[j, target_idx] = 1
            
            yield X_batch, y_batch

class CharacterPredictor:
    def __init__(self, input_size=199, hidden_layers=[128, 64], output_size=74, 
                 alpha=0.01, batch_size=32, epochs=100, 
                 decay_rate=0.1, weight_decay=0.0001):
        """Initialize character prediction neural network
        
        Args:
            input_size: Length of input sequences
            hidden_layers: List of hidden layer sizes
            output_size: Size of vocabulary (number of unique characters)
            alpha: Initial learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            decay_rate: Learning rate decay factor
            weight_decay: L2 regularization coefficient
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.initial_alpha = alpha
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.decay_rate = decay_rate
        self.weight_decay = weight_decay
        self.momentum = 0.9
        self.velocity_weights = None
        self.velocity_biases = None
        self.validation_losses = []
        
        # Initialize network parameters
        input_features = input_size * output_size
        self.weights, self.biases = init_weights(
            input_features,
            hidden_layers,
            output_size
        )
        
        # Initialize momentum arrays
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]
        
    def prepare_sequence(self, text, char_to_idx, training=True):
        """Prepare input sequences and targets using memory-efficient chunked processing
        
        Processes text in chunks to avoid memory issues with large datasets.
        Uses temporary files to store intermediate results.
        
        Args:
            text: Input text to process
            char_to_idx: Character to index mapping
            training: Whether preparing for training (True) or prediction (False)
        
        Returns:
            X: One-hot encoded input sequences
            y: One-hot encoded targets (if training=True)
        """
        print("Preparing sequences...")
        start_time = time.time()
        last_check_time = start_time
        last_check_sequences = 0
        
        sequences_to_process = len(text) - self.input_size if training else 1
        vocab_size = len(char_to_idx)
        
        # Process in chunks to manage memory
        chunk_size = 10000
        num_chunks = (sequences_to_process + chunk_size - 1) // chunk_size
        
        # Fast char to index lookup
        char_to_idx_arr = np.zeros(256, dtype=np.int32)
        for char, idx in char_to_idx.items():
            char_to_idx_arr[ord(char)] = idx
        
        # Create temporary storage
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
                
                X = np.zeros((chunk_size_actual, self.input_size, vocab_size), dtype=np.float32)
                y = np.zeros((chunk_size_actual, vocab_size), dtype=np.float32) if training else None
                
                for i in range(chunk_size_actual):
                    if (chunk_start + i) % 10000 == 0:
                        current_time = time.time()
                        sequences_processed = chunk_start + i
                        
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
                    
                    text_idx = chunk_start + i
                    sequence = np.frombuffer(text[text_idx:text_idx + self.input_size].encode(), dtype=np.uint8)
                    sequence_indices = char_to_idx_arr[sequence]
                    
                    X[i, np.arange(self.input_size), sequence_indices] = 1
                    
                    if training:
                        target_idx = char_to_idx_arr[ord(text[text_idx + self.input_size])]
                        y[i, target_idx] = 1
                
                # Save chunk
                x_filename = os.path.join(temp_dir, f'chunk_X_{chunk}.npy')
                np.save(x_filename, X)
                chunk_files_X.append(x_filename)
                
                if training:
                    y_filename = os.path.join(temp_dir, f'chunk_y_{chunk}.npy')
                    np.save(y_filename, y)
                    chunk_files_y.append(y_filename)
                
                del X
                if training:
                    del y
                gc.collect()
            
            # Combine chunks efficiently
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
                
                del group_X
                if training:
                    del group_y
                gc.collect()
            
            X = np.concatenate(final_X, axis=0)
            y = np.concatenate(final_y, axis=0) if training else None
        
        finally:
            # Cleanup temporary files
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
    
    def fit(self, text, char_to_idx, validation_split=0.1):
        """Train the network using batched sequences
        
        Args:
            text: Training text
            char_to_idx: Character to index mapping
            validation_split: Fraction of data to use for validation
        """
        import time
        start_time = time.time()
        last_batch_time = start_time
        
        n_sequences = len(text) - self.input_size
        n_train = int(n_sequences * (1 - validation_split))
        
        train_text = text[:n_train + self.input_size]
        val_text = text[n_train:]
        
        train_gen = sequence_generator(train_text, self.batch_size, self.input_size, char_to_idx)
        val_gen = sequence_generator(val_text, self.batch_size, self.input_size, char_to_idx)
        
        steps_per_epoch = n_train // self.batch_size
        val_steps = (len(val_text) - self.input_size) // self.batch_size
        
        print(f"Starting training with {n_train} training sequences...")
        print(f"Steps per epoch: {steps_per_epoch}")
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.alpha = self.initial_alpha / (1 + epoch * self.decay_rate)
            
            epoch_loss = 0
            batch_count = 0
            print(f"\nEpoch {epoch + 1}/{self.epochs} - Learning Rate: {self.alpha:.6f}")
            
            # Training loop
            for step in range(steps_per_epoch):
                batch_X, batch_y = next(train_gen)
                
                # Forward pass
                Z_values, activations = forward(batch_X, self.weights, self.biases)
                
                batch_loss = np.mean((activations[-1] - batch_y) ** 2)
                epoch_loss += batch_loss
                batch_count += 1
                
                if batch_count % 100 == 0:
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    batch_time = current_time - last_batch_time
                    last_batch_time = current_time
                    
                    progress = step / steps_per_epoch
                    batches_remaining_epoch = steps_per_epoch - step
                    time_per_batch = batch_time / 100
                    
                    epoch_remaining = batches_remaining_epoch * time_per_batch
                    epochs_remaining = self.epochs - epoch - progress
                    total_remaining = epoch_remaining + (epochs_remaining * steps_per_epoch * time_per_batch)
                    
                    def format_time(seconds):
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        return f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                    
                    current_loss = epoch_loss / batch_count
                    print(f"Batch {batch_count}: {progress*100:.1f}% complete - "
                          f"Loss: {current_loss:.4f} - "
                          f"Elapsed: {format_time(elapsed_time)} - "
                          f"Epoch remaining: {format_time(epoch_remaining)} - "
                          f"Total remaining: {format_time(total_remaining)}")
                
                # Backward pass
                dZ = activations[-1] - batch_y
                dW_list = []
                db_list = []
                
                # Output layer gradients
                dW = np.dot(activations[-2].T, dZ)
                db = np.sum(dZ, axis=0, keepdims=True)
                dW_list.insert(0, dW)
                db_list.insert(0, db)
                
                # Hidden layer gradients
                for layer in range(len(self.hidden_layers), 0, -1):
                    dA = np.dot(dZ, self.weights[layer].T)
                    dZ = dA * ReLU.derivative(Z_values[layer-1])
                    
                    if layer > 1:
                        dW = np.dot(activations[layer-1].T, dZ)
                    else:
                        dW = np.dot(activations[0].T, dZ)
                        
                    db = np.sum(dZ, axis=0, keepdims=True)
                    dW_list.insert(0, dW)
                    db_list.insert(0, db)
                
                # Gradient clipping
                dW_list = clip_gradients(dW_list)
                db_list = clip_gradients(db_list)
                
                # Update weights with momentum and L2 regularization
                for layer in range(len(self.weights)):
                    dW_list[layer] += self.weight_decay * self.weights[layer]
                    
                    self.velocity_weights[layer] = (self.momentum * self.velocity_weights[layer] - 
                                                  self.alpha * dW_list[layer] / self.batch_size)
                    self.velocity_biases[layer] = (self.momentum * self.velocity_biases[layer] - 
                                                 self.alpha * db_list[layer] / self.batch_size)
                    
                    self.weights[layer] += self.velocity_weights[layer]
                    self.biases[layer] += self.velocity_biases[layer]
            
            # Validation
            val_loss = 0
            val_batch_count = 0
            
            for _ in range(val_steps):
                val_X, val_y = next(val_gen)
                val_Z_values, val_activations = forward(val_X, self.weights, self.biases)
                val_loss += np.mean((val_activations[-1] - val_y) ** 2)
                val_batch_count += 1
            
            avg_val_loss = val_loss / val_batch_count
            avg_epoch_loss = epoch_loss / batch_count
            self.validation_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch + 1}/{self.epochs} complete - Train Loss: {avg_epoch_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    def predict(self, text, char_to_idx, idx_to_char):
        """Generate predictions for input text
        
        Returns list of (character, probability) pairs for top 5 most likely next characters
        """
        if len(text) > self.input_size:
            text = text[-self.input_size:]
        
        vocab_size = len(char_to_idx)
        X = np.zeros((1, self.input_size, vocab_size))
        
        for i, char in enumerate(text):
            if char in char_to_idx:
                X[0, i, char_to_idx[char]] = 1
        
        Z_values, activations = forward(X, self.weights, self.biases)
        
        predictions = activations[-1]
        top_k_indices = np.argsort(predictions[0])[-5:][::-1]
        
        char_probs = []
        for idx in top_k_indices:
            char = idx_to_char[idx]
            prob = predictions[0][idx]
            char_probs.append((char, prob))
        
        return char_probs
    
    def save_weights(self, filename='neuralNetworkWeights.json'):
        """Save model weights and architecture to JSON file"""
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
        """Load model weights and verify architecture from JSON file"""
        with open(filename, 'r') as f:
            weights_data = json.load(f)
        
        arch = weights_data['architecture']
        if (arch['input_size'] != self.input_size or 
            arch['hidden_layers'] != self.hidden_layers or 
            arch['output_size'] != self.output_size):
            raise ValueError("Model architecture in file doesn't match current model")
        
        self.weights = [np.array(w) for w in weights_data['weights']]
        self.biases = [np.array(b) for b in weights_data['biases']]
        
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]
        print(f"Weights loaded from {filename}")

    def layer_norm(self, x, epsilon=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(variance + epsilon)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward_propagation(self, X):
        self.activations = [X]
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            z_norm = self.layer_norm(z)  # Add normalization
            self.activations.append(self.activation_fn(z_norm))
        
        # Change final layer to use softmax
        self.final_output = self.softmax(np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1])
        return self.final_output

    def sample_with_temperature(self, probabilities, temperature=0.8):
        """
        Sample from probability distribution with temperature adjustment.
        - Lower temperature (< 1.0) = More conservative/focused
        - Higher temperature (> 1.0) = More random/creative
        """
        # Adjust probabilities with temperature
        logits = np.log(probabilities + 1e-10)  # Add small epsilon to avoid log(0)
        exp_logits = np.exp(logits / temperature)
        probs = exp_logits / np.sum(exp_logits)
        
        return np.random.choice(len(probs), p=probs)

# Main execution
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
    hidden_layers=[199, 199, 199, 100],
    output_size=74,
    alpha=0.001,
    epochs=10,
    batch_size=128,
    decay_rate=0.1,
    weight_decay=0.0001
)

training = False  # Set to True to train new model, False to load existing weights

if not training:
    try:
        predictor.load_weights()
        
        print("Making predictions...")
        test_text = "You are a joke teller. You are a joke teller. You are a joke teller. You are a joke teller. You tell jokes, Okay start!!!!: What do you call a redditor that doesn't use the search button in /r/jokes?"
        current_text = test_text
        
        print("\nInput:")
        print(test_text)
        print("\nGenerating outputs...")

        # Generate text using two different sampling methods
        generated_text_greedy = ""
        generated_text_weighted = ""
        current_text_greedy = current_text
        current_text_weighted = current_text

        for i in range(400):
            predictions_greedy = predictor.predict(current_text_greedy[-199:], char_to_idx, idx_to_char)
            predictions_weighted = predictor.predict(current_text_weighted[-199:], char_to_idx, idx_to_char)
            
            # Greedy: Always choose most likely character
            next_char_greedy = predictions_greedy[0][0]
            
            # Weighted random: Sample from all chars with squared probabilities
            chars, probs = zip(*predictions_weighted)
            probs = np.array(probs)
            probs = probs ** 2
            probs = probs / np.sum(probs)
            next_char_weighted = np.random.choice(chars, p=probs)
            
            generated_text_greedy += next_char_greedy
            generated_text_weighted += next_char_weighted
            current_text_greedy = current_text_greedy + next_char_greedy
            current_text_weighted = current_text_weighted + next_char_weighted

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
    predictor.fit(text, char_to_idx)
    predictor.save_weights()