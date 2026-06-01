import lyricsgenius
import pandas as pd
import os

# Initializing Genius API
GENIUS_TOKEN = "V9f-qAvh_tyoNaw_TjkJCC6P06XvMr_X3wHi1FxVsVCiaLvUgr8AMUwy7M5uF67D"

genius = lyricsgenius.Genius(GENIUS_TOKEN, skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], timeout=30, retries=3)

# Listing target artist
artists_list = ["Taco Hemingway", "sanah", "Bedoes", "Dawid Podsiadło", "Mata", "Quebonafide"]
all_songs_data = []

# Iterating over each artist to fetch their top tracks
for artist_name in artists_list:
    print(f"\nDownloading for: {artist_name}")
    
    # Searching for top 15 most popular songs
    artist = genius.search_artist(artist_name, max_songs=15, sort="popularity")
    
    if artist is not None:
        for song in artist.songs:
            all_songs_data.append({
                "artist": artist.name,
                "title": song.title,
                "lyrics": song.lyrics
            })

# Saving raw data in csv file
os.makedirs("data/raw", exist_ok=True)

df = pd.DataFrame(all_songs_data)
df.to_csv("data/raw/lyrics.csv", index=False, encoding="utf-8")

print("Success! All lyrics are downloaded")