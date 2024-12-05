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
    
    # Keep only allowed characters and normalize whitespace
    cleaned_text = ' '.join(''.join(char for char in text if char in allowed_chars).split())
    
    return cleaned_text.strip()

for i, joke in enumerate(jokes):
    if i >= MaxJokesToProcess:
        break
        
    # Clean and combine title and body
    title = clean_text(joke['title'])
    body = clean_text(joke['body'])
    combined = f"{title} {body}".strip()
    
    # Only add separator if there's existing text
    if i > 0:
        combined_text += "   "  # Add separator between entries
    combined_text += combined
    
    # Add to processed data
    redditJokesProcessed.append({
        'text': combined,
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
    # Join all joke texts with double spaces and write as a single line
    f.write('  '.join(joke['text'] for joke in redditJokesProcessed))

print(f"Saved processed data to {output_path}")