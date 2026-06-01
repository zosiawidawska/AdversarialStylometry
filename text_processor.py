import pandas as pd

FUNCTION_WORDS = ['i', 'w', 'na', 'z', 'do', 'że', 'o', 'a', 'nie', 'bo', 'jak', 'to', 'co', 'ja', 'ty']

def extract_text_features(text, nlp_model):
    doc = nlp_model(text)

    sentences = list(doc.sents)
    num_sentences = len(sentences) if len(sentences) > 0 else 1
    words_count = len([token for token in doc if not token.is_punct])
    avg_sentence_length = words_count / num_sentences
    
    pos_counts = {'VERB': 0, 'ADJ': 0, 'PRON': 0}
    
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
            
    text_lower = text.lower().split()
    func_word_freq = {f"freq_{word}": text_lower.count(word) for word in FUNCTION_WORDS}
    
    features = {
        'lyrics': text,
        'avg_sentence_length': round(avg_sentence_length, 2),
        'count_verbs': pos_counts['VERB'],
        'count_adjectives': pos_counts['ADJ'],
        'count_pronouns': pos_counts['PRON']
    }
    features.update(func_word_freq)
    
    return pd.DataFrame([features])