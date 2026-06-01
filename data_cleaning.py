import pandas as pd
import re
import os

def clean_lyrics(text):
    if not isinstance(text, str):
        return ""
    
    # Removing song structure tags like [Chorus], [Verse 1], etc.
    text = re.sub(r'\[.*?\]', ' ', text)

    # Replacing newlines with spaces to create a continuous string of text
    text = text.replace('\n', ' ')

    # Replacing multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# Loading raw data
df = pd.read_csv('data/raw/lyrics.csv')

# Appling cleaning function to the entire 'lyrics' column
df['clean_lyrics'] = df['lyrics'].apply(clean_lyrics)

# Creating samples
chunk_size = 500
processed_data = []

for artist, group in df.groupby('artist'):

    # Joining all cleaned songs into one string
    all_artist_text = " ".join(group['clean_lyrics'].tolist())
    
    # Spliting into list of words
    words = all_artist_text.split()
    
    # Iterating through the list of words, appending only chunks that have exactly 500 words
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        
        if len(chunk_words) == chunk_size:
            processed_data.append({
                'artist': artist,
                'lyrics': " ".join(chunk_words)
            })

# Saving data
os.makedirs("data/processed", exist_ok=True)

final_df = pd.DataFrame(processed_data)
final_df.to_csv('data/processed/clean_lyrics.csv', index=False, encoding='utf-8')

print(f"Success! Generated {len(final_df)} text chunks ({chunk_size} words each).")
print("Data are saved")