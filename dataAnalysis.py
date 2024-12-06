def analyze_text_file(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Get total character count
    total_chars = len(text)
    
    # Get unique characters
    unique_chars = sorted(set(text))
    num_unique_chars = len(unique_chars)
    
    # Calculate character frequencies
    char_freq = {}
    for char in text:
        char_freq[char] = char_freq.get(char, 0) + 1
    
    # Sort character frequencies by count (descending)
    sorted_freq = dict(sorted(char_freq.items(), key=lambda x: x[1], reverse=True))
    
    # Print results
    print(f"Total number of characters: {total_chars}")
    print(f"Number of unique characters: {num_unique_chars}")
    print("\nUnique characters:")
    print(''.join(unique_chars))
    
    print("\nCharacter frequencies:")
    for char, freq in sorted_freq.items():
        if char == '\n':
            char_display = '\\n'
        elif char == '\r':
            char_display = '\\r'
        elif char == ' ':
            char_display = 'SPACE'
        else:
            char_display = char
        percentage = (freq / total_chars) * 100
        print(f"'{char_display}': {freq} ({percentage:.2f}%)")

# Run the analysis
file_path = 'data/redditJokesProcessed.txt'
analyze_text_file(file_path)
