import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def run_experiment(df, text_column, model_name):
    print(f"Running Experiment: {model_name}")
    print(f"Target text column: {text_column}")

    # Numeric features
    numeric_features = ['avg_sentence_length', 'count_verbs', 'count_adjectives', 'count_pronouns']
    
    # Adding all function word frequency columns
    freq_features = [col for col in df.columns if col.startswith('freq_')]
    numeric_features.extend(freq_features)

    # X (features) and y (target)
    X = df[numeric_features + [text_column]]
    y = df['artist']

    # Spliting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # TF-IDF for char n-grams (2 to 4 characters)
    text_transformer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=5000)
    
    numeric_transformer = StandardScaler()

    # Transformers into a single preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, text_column),
            ('num', numeric_transformer, numeric_features)
        ]
    )

    # Classification pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    # Training the model
    print("Training model...")
    model_pipeline.fit(X_train, y_train)

    # Evaluate and print results to the console
    print(f"\n EVALUATION RESULTS FOR: {model_name.upper()}")
    y_pred = model_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name}.joblib"
    joblib.dump(model_pipeline, model_path)
    print(f"Success! Model saved to {model_path}")

    return model_pipeline

if __name__ == "__main__":
    # Load the processed dataset
    print("Loading data...")
    data_path = 'data/processed/features.csv'
    df = pd.read_csv(data_path)

    df['lyrics'] = df['lyrics'].fillna("")
    df['text_no_nouns'] = df['text_no_nouns'].fillna("")

    # Main model using full text
    run_experiment(
        df=df, 
        text_column='lyrics', 
        model_name='logistic_regression_main'
    )

    # Control experiment using text without nouns
    run_experiment(
        df=df, 
        text_column='text_no_nouns', 
        model_name='logistic_regression_control'
    )