import streamlit as st
import numpy as np
import joblib
import spacy
import matplotlib.pyplot as plt

# Import the extractor function
from text_processor import extract_text_features

# Configurete the main Streamlit page settings
st.set_page_config(page_title="Author Attribution Demo", layout="centered")

@st.cache_resource
def load_resources():
    print("Loading spaCy model and scikit-learn pipeline")
    nlp_model = spacy.load("pl_core_news_sm", disable=["ner"])
    pipeline = joblib.load('models/logistic_regression_main.joblib')
    return nlp_model, pipeline

# Initialize resources
nlp, model_pipeline = load_resources()

def get_top_features_for_class(pipeline, class_idx, top_n=5):
    # Extracting the most important stylistic features

    # Extract specific steps from the scikit-learn pipeline
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get raw feature names and clean them up
    feature_names = preprocessor.get_feature_names_out()
    clean_feature_names = [name.split('__')[-1] for name in feature_names]
    
    # Retrieve the weights for the winning class and sort them
    coefs = classifier.coef_[class_idx]
    top_indices = np.argsort(coefs)[-top_n:][::-1]
    
    return [(clean_feature_names[i], coefs[i]) for i in top_indices]

# UI Setup
st.title("Adversarial Stylometry Classifier")
st.markdown("Write a short text (approx. 50-200 words) and check which artist's style it matches.")

# Multiline text input for the user
user_input = st.text_area("Input Text:", height=200)

# Trigger analysis when primaty button is clicked
if st.button("Analyze Stylometry", type="primary"):
    # Validation if the text is long enough for NLP analysis
    if len(user_input.split()) < 10:
        st.warning("Please provide a longer text (at least 10 words)")
    else:
        # Extracting features using the imported function
        df_input = extract_text_features(user_input, nlp)
        
        # Predict
        predicted_author = model_pipeline.predict(df_input)[0]
        probabilities = model_pipeline.predict_proba(df_input)[0]
        classes = model_pipeline.classes_
        
        st.success(f"Predicted Author: **{predicted_author}**")
        
        # Visualize results
        col1, col2 = st.columns(2)
        
        # Pie chart
        with col1:
            st.markdown("Probability Distribution")
            
            fig, ax = plt.subplots(figsize=(7, 5)) 
            filtered_probs = [p for p in probabilities if p > 0.01]
            filtered_classes = [c for c, p in zip(classes, probabilities) if p > 0.01]
            
            wedges, texts, autotexts = ax.pie(
                filtered_probs, 
                autopct='%1.1f%%', 
                startangle=140, 
                colors=plt.cm.tab10.colors
            )
            
            ax.legend(
                wedges, 
                filtered_classes,
                title="Probability:",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
            
            ax.axis('equal')
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            st.pyplot(fig, use_container_width=True)
            
        # Model Explainability    
        with col2:
            st.markdown("Why this result?")
            class_idx = list(classes).index(predicted_author)
            top_features = get_top_features_for_class(model_pipeline, class_idx)
            
            for feature_name, weight in top_features:
                st.markdown(f"- **{feature_name}** *(weight: {weight:.2f})*")