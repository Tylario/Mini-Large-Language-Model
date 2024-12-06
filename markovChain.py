import numpy as np
from collections import defaultdict
import random
import json

class MultiOrderMarkovChain:
    def __init__(self, orders, use_classes=True):
        """
        Initialize Markov Chain with multiple orders
        
        Args:
            orders (list): Required list of sequence lengths to consider
            use_classes (bool): Whether to use character classes (default: True)
        """
        self.orders = orders
        self.use_words = True
        self.use_classes = use_classes
        self.transitions = {
            order: defaultdict(lambda: defaultdict(float)) 
            for order in orders
        }
        self.starts = []
        
    def train(self, text):
        """Train the model on input text"""
        print(f"Training on {len(text)} {'words' if self.use_words else 'characters'}...")
        
        if self.use_words:
            tokens = text.replace('\n', ' \n ').split(' ')
        else:
            tokens = list(text)
            
        # Train each order
        for order in self.orders:
            print(f"Training order {order}...")
            for i in range(len(tokens) - order):
                current = tuple(tokens[i:i+order]) if self.use_words else tokens[i:i+order]
                next_token = tokens[i+order]
                
                # Weight by position in training data
                position_weight = 1.0 + (i / len(tokens))
                self.transitions[order][current][next_token] += position_weight
                
                # Store sentence starts
                if i == 0 or any(t.strip() in '.!?\n' for t in tokens[i-1:i]):
                    self.starts.append(current)
        
        print("Training complete!")
        
    def generate(self, length=200, temperature=1.0, order_weights=None):
        """
        Generate text using multiple orders
        
        Args:
            length (int): Length of text to generate
            temperature (float): Randomness of output (0.0-1.0, lower = more conservative)
            order_weights (dict): Weight for each order (e.g., {2: 0.35, 4: 0.3, ...})
        """
        if not self.transitions:
            raise Exception("Model must be trained first!")
            
        # Default weights if none provided
        if order_weights is None:
            total = sum(1/order for order in self.orders)  # Higher weight to shorter orders
            order_weights = {
                order: (1/order)/total for order in self.orders
            }
        
        # Start with longest sequence
        current = random.choice(self.starts)
        if self.use_words:
            result = list(current)
        else:
            result = current
        
        for _ in range(length):
            next_token_probs = defaultdict(float)
            
            # Get predictions from each order
            for order in self.orders:
                # Get the appropriate length sequence for this order
                if self.use_words:
                    current_seq = tuple(result[-order:]) if len(result) >= order else tuple(result)
                else:
                    current_seq = result[-order:] if len(result) >= order else result
                
                if current_seq in self.transitions[order]:
                    possibilities = self.transitions[order][current_seq]
                    
                    # Apply temperature
                    counts = np.array(list(possibilities.values())) ** (1.0 / temperature)
                    probs = counts / counts.sum()
                    
                    # Add weighted probabilities to total
                    for token, prob in zip(possibilities.keys(), probs):
                        next_token_probs[token] += prob * order_weights[order]
            
            if not next_token_probs:
                # If stuck, pick a new start
                current = random.choice(self.starts)
                if self.use_words:
                    result.extend(current)
                else:
                    result += current
                continue
            
            # Normalize probabilities
            total_prob = sum(next_token_probs.values())
            for token in next_token_probs:
                next_token_probs[token] /= total_prob
            
            # Choose next token
            tokens = list(next_token_probs.keys())
            probs = list(next_token_probs.values())
            next_token = np.random.choice(tokens, p=probs)
            
            if self.use_words:
                result.append(next_token)
            else:
                result += next_token
        
        return ' '.join(result) if self.use_words else result

    def save(self, filename='markov_model.json'):
        """Save the model to a file"""
        # Convert transitions dictionary to use string keys
        transitions_str = {}
        for order, trans in self.transitions.items():
            transitions_str[str(order)] = {}
            for k, v in trans.items():
                key_str = str(k) if not isinstance(k, tuple) else '|||'.join(k)
                transitions_str[str(order)][key_str] = dict(v)
        
        # Convert starts list to strings if they're tuples
        starts_str = []
        for start in self.starts:
            if isinstance(start, tuple):
                starts_str.append('|||'.join(start))
            else:
                starts_str.append(start)
        
        model_data = {
            'orders': self.orders,
            'use_words': self.use_words,
            'use_classes': self.use_classes,
            'transitions': transitions_str,
            'starts': starts_str
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load(self, filename='markov_model.json'):
        """Load the model from a file"""
        with open(filename, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.orders = model_data['orders']
        self.use_words = model_data.get('use_words', True)
        self.use_classes = model_data.get('use_classes', True)
        
        # Convert string keys back to proper types
        self.transitions = {}
        for order, trans in model_data['transitions'].items():
            order = int(order)
            self.transitions[order] = defaultdict(lambda: defaultdict(float))
            for k, v in trans.items():
                if self.use_words and '|||' in k:
                    key = tuple(k.split('|||'))
                else:
                    key = k
                self.transitions[order][key] = defaultdict(float, v)
        
        # Convert starts back to tuples if using words
        self.starts = []
        for start in model_data['starts']:
            if self.use_words and '|||' in start:
                self.starts.append(tuple(start.split('|||')))
            else:
                self.starts.append(start)
                
        print(f"Model loaded from {filename}")

# Example usage
if __name__ == "__main__":
    print("Loading text data...")
    with open('data/redditJokesProcessed.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded {len(text)} characters")
    
    # Create model with orders 1-8
    model = MultiOrderMarkovChain(
        orders=[1, 2, 3, 4, 5, 6, 7, 8],  # Changed to match weights
    )
    model.train(text)
    
    print("\nGenerating with emphasis on short-term patterns:")
    
    # Weights for orders 1-8
    weights = {
        1: 0.1,   # Single word context
        2: 0.2,   # Two word context
        3: 0.1,   # Three word context
        4: 0.05,  # Four word context
        5: 0.025, # Five word context
        6: 0.0125,# Six word context
        7: 0.00625,# Seven word context
        8: 0.003125# Eight word context
    }
    
    print("\nGenerated text:")
    print(model.generate(
        length=200, 
        temperature=1.0,
        order_weights=weights
    ))