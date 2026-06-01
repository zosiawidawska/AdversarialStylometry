import pandas as pd
import spacy

# Loading the Polish NLP model.
nlp = spacy.load("pl_core_news_sm", disable=["ner"])

# Defining list of function words
function_words = ['i', 'w', 'na', 'z', 'do', 'że', 'o', 'a', 'nie', 'bo', 'jak', 'to', 'co', 'ja', 'ty']

def extract_features(text):
    """
    Extracting stylometric features:
    - Average sentence length
    - Part of Speech (POS) frequencies (Verbs, Adjectives, Pronouns)
    - Function word frequencies
    - A modified text string with all nouns masked
    """
    doc = nlp(text)

    # Calculating average sentence length
    sentences = list(doc.sents)
    num_sentences = len(sentences) if len(sentences) > 0 else 1
    words_count = len([token for token in doc if not token.is_punct])
    avg_sentence_length = words_count / num_sentences
    
    # POS counting
    pos_counts = {'VERB': 0, 'ADJ': 0, 'PRON': 0}
    text_without_nouns = []
    
    for token in doc:
        # Increment POS counters if the token matches our target categories
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
            
        # For control experiment replacing all nouns with a generic placeholder.    
        if token.pos_ != 'NOUN':
            text_without_nouns.append(token.text)
        else:
            text_without_nouns.append("[NOUN]")

    # Calculating frequencies of specific function words        
    text_lower = text.lower().split()
    func_word_freq = {f"freq_{word}": text_lower.count(word) for word in function_words}
    
    # Compiling all extracted features into a dictionary
    features = {
        'avg_sentence_length': round(avg_sentence_length, 2),
        'count_verbs': pos_counts['VERB'],
        'count_adjectives': pos_counts['ADJ'],
        'count_pronouns': pos_counts['PRON'],
        'text_no_nouns': " ".join(text_without_nouns)
    }
    
    # Merge the function word frequencies into the main features dictionary
    features.update(func_word_freq)
    return pd.Series(features)

# Saving data
df = pd.read_csv('data/processed/clean_lyrics.csv')

features_df = df['lyrics'].apply(extract_features)

final_df = pd.concat([df, features_df], axis=1)

output_path = 'data/processed/features.csv'
final_df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Success! Stylistic features extracted and saved")