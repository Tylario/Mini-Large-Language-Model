import json
import os

# Load data from reddit_jokes.json
file_path = os.path.join(os.path.dirname(__file__), 'data', 'reddit_jokes.json')
with open(file_path, 'r') as file:
    jokes = json.load(file)  # This is a list of joke dictionaries

MaxJokesToProcess = 5000
redditJokesProcessed = []
combined_text = ""

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
    
    # Replace various types of whitespace with a single space
    text = ' '.join(text.split())
    
    # Keep only allowed characters
    cleaned_text = ''.join(char for char in text if char in allowed_chars)
    
    # Remove multiple spaces that might have been created
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text.strip()

for i, joke in enumerate(jokes):
    if i >= MaxJokesToProcess:
        break
        
    # Process the joke
    if i > 0:
        combined_text += "   "  # Add separator between entries
    combined_text += f"{joke['title']} {joke['body']}"
    
    # Add to processed data
    redditJokesProcessed.append({
        'text': f"{joke['title']} {joke['body']}",
        'score': joke['score']
    })
    
    # Print progress every 1000 jokes
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1} jokes...")

print(f"Finished processing {len(redditJokesProcessed)} jokes")
print("First joke as sample:", combined_text[:200] + "...")  # Print first 200 chars as sample

# Save processed data to new file
output_path = os.path.join(os.path.dirname(__file__), 'data', 'redditJokesProcessed.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    for joke in redditJokesProcessed:
        f.write(f"{joke['text']}\n")  # Write each joke text on a new line

print(f"Saved processed data to {output_path}")