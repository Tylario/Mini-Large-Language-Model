import json
import os
import time

# Load data from reddit_jokes.json
file_path = os.path.join(os.path.dirname(__file__), 'data', 'reddit_jokes.json')
with open(file_path, 'r') as file:
    jokes = json.load(file)  # This is a list of joke dictionaries

MaxJokesToProcess = float('inf')  # infinity
redditJokesProcessed = []
joke_texts = []
start_time = time.time()

# Initialize a list to store processing times
processing_times = []

def clean_text(text):
    # First, explicitly replace any newlines or carriage returns with spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Define allowed characters:
    allowed_chars = set(
        # Lowercase letters (a-z)
        'abcdefghijklmnopqrstuvwxyz'
        # Uppercase letters (A-Z)
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # Numbers (0-9)
        '0123456789'
        # Space for word separation
        ' '
        # Basic punctuation:
        '.'   # Period for sentences
        ','   # Comma for clauses
        '!'   # Exclamation for emphasis
        '?'   # Question mark
        '\''  # Apostrophe for contractions (don't, it's)
        '"'   # Quotation marks
        '-'   # Hyphen for compound words
        ':'   # Colon for lists or explanations
        ';'   # Semicolon for complex sentences
        '('   # Opening parenthesis
        ')'   # Closing parenthesis
        '...' # Ellipsis for trailing thoughts
    )
    
    # Keep only allowed characters and normalize whitespace
    cleaned_text = ' '.join(''.join(char for char in text if char in allowed_chars).split())
    
    return cleaned_text.strip()

for i, joke in enumerate(jokes):
    if i >= MaxJokesToProcess:
        break
        
    joke_start_time = time.time()
    
    # Clean and combine title and body
    title = clean_text(joke['title'])
    body = clean_text(joke['body'])
    combined = f"{title} {body}".strip()
    
    # Instead of concatenating strings, append to list
    joke_texts.append(combined)
    
    # Add to processed data
    redditJokesProcessed.append({
        'text': combined,
        'score': joke['score']
    })
    
    # Record the processing time for this joke
    processing_times.append(time.time() - joke_start_time)
    
    if (i + 1) % 1000 == 0:
        avg_processing_time = sum(processing_times[-1000:]) / 1000
        remaining_jokes = len(jokes) - (i + 1)
        remaining_seconds = remaining_jokes * avg_processing_time
        
        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)
        seconds = int(remaining_seconds % 60)
        
        time_str = ""
        if hours > 0:
            time_str += f"{hours}h "
        if minutes > 0:
            time_str += f"{minutes}m "
        if seconds > 0:
            time_str += f"{seconds}s"
            
        print(f"Processed {i + 1} of {len(jokes)} jokes... Estimated time left: {time_str}")

print(f"Finished processing {len(redditJokesProcessed)} jokes")

# Save processed data to new file
output_path = os.path.join(os.path.dirname(__file__), 'data', 'redditJokesProcessed.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(' '.join(joke_texts))

print(f"Saved processed data to {output_path}")