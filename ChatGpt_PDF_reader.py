import streamlit as st
import PyPDF2
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# Download NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Tokenize the text
    words = text.split()
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

# Function to generate and display the word cloud
def generate_wordcloud(words):
    wordcloud = WordCloud(width=800, height=400, max_words=500, background_color="white").generate(" ".join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# Streamlit App
def main():
    st.title("PDF WordCloud Generator")
    st.write("Upload a PDF file to extract text, preprocess it, and generate a word cloud.")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        st.write("Processing the uploaded PDF...")
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(uploaded_file)
        if not pdf_text.strip():
            st.error("No text found in the PDF. Please upload a valid text-based PDF.")
            return
        
        st.write("Extracted text:")
        st.write(pdf_text[:1000])  # Show first 1000 characters of extracted text

        # Preprocess the text
        preprocessed_words = preprocess_text(pdf_text)
        st.write(f"Number of words after preprocessing: {len(preprocessed_words)}")
        
        # Generate word cloud
        st.subheader("Word Cloud")
        generate_wordcloud(preprocessed_words)

if __name__ == "__main__":
    main()