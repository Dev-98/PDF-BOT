import streamlit as st
import fitz  # PyMuPDF
import re
import nltk
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data if not already downloaded
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

helping_verbs = ["how", "why", "is", "are", "was", "were", "will", "can", "could", "should", "would", "different", "types", "difference", "similarity", "common" ,"similar"] 
words_to_ignore = helping_verbs + stopwords.words("english")



def extract_and_process_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    text = re.sub(r'[^A-Za-z\s\d+]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def is_meaningful_word(word):
    synsets = wordnet.synsets(word)
    return len(synsets) > 0


def replace_non_meaningful_words(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    for i in range(len(tokens)):
        token = tokens[i]
        if not is_meaningful_word(token.lower()):
            lemma = lemmatizer.lemmatize(token)
            synonyms = wordnet.synsets(lemma)
            if synonyms:
                closest_synonym = synonyms[0].lemmas()[0].name()
                tokens[i] = closest_synonym
    return ' '.join(tokens)


def calculate_tfidf_vectors(documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    return tfidf_vectorizer, tfidf_matrix


def answer_question_with_context(user_question, tfidf_vectorizer, tfidf_matrix, sentences, word_limit):
    user_question = re.sub(r'[^A-Za-z\s]', '', user_question)
    user_question = re.sub(r'\s+', ' ', user_question).strip()
    user_question = ' '.join([word for word in word_tokenize(user_question.lower()) if word not in words_to_ignore])
    user_question_vector = tfidf_vectorizer.transform([user_question])
    similarity_scores = cosine_similarity(user_question_vector, tfidf_matrix)
    max_similarity_index = np.argmax(similarity_scores)
    if similarity_scores[0][max_similarity_index] == 0:
        return 
    context_sentence = sentences[max_similarity_index]
    context_words = word_tokenize(context_sentence)
    context_index = context_words.index(user_question.split()[0])
    context_length = len(context_words)
    context = ' '.join(context_words[context_index:context_index + context_length])
    context = replace_non_meaningful_words(context)
    if len(context.split()) > 550:
        context = ' '.join(context.split()[:550])
    merged_context = merge_consecutive_words(context)
    limited_context = limit_answer_length(merged_context, word_limit)
    return limited_context


def merge_consecutive_words(text):
    tokens = word_tokenize(text)
    merged_tokens = []
    i = 0
    while i < len(tokens) - 1:
        combined_word = tokens[i] + tokens[i + 1]
        if is_meaningful_word(combined_word):
            merged_tokens.append(combined_word)
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1
    while i < len(tokens):
        merged_tokens.append(tokens[i])
        i += 1
    return ' '.join(merged_tokens)


def limit_answer_length(text, word_limit):
    tokens = word_tokenize(text)
    if len(tokens) > word_limit:
        return ' '.join(tokens[:word_limit])
    return text


def format_conversation_response(raw_response):
    formatted_response = re.sub(r'[^A-Za-z0-9\s.]', '', raw_response)
    sentences = sent_tokenize(formatted_response)
    sentences = [sentence.capitalize() for sentence in sentences]
    formatted_response = ' '.join(sentences)
    return formatted_response.strip()


def main():
    st.title("PDF Chatbot with Streamlit")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        st.sidebar.title("Settings")
        word_limit = st.sidebar.slider("Select answer word limit:", 10, 200, 20, step=10)

        pdf_path = "uploaded_pdf.pdf"
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(uploaded_file.read())

        processed_text = extract_and_process_text(pdf_path)
        sentences = sent_tokenize(processed_text)
        tfidf_vectorizer, tfidf_matrix = calculate_tfidf_vectors(sentences)

        st.write("Bot: Hi! I'm your PDF Chatbot. How can I assist you today?")

        question_counter = 0  # Initialize a counter for unique widget keys

        while True:
            user_input = st.text_input(f"Question {question_counter + 1}:", key=f"question_{question_counter}")
            if not user_input:
                break

            main_answer = ""
            user_question_words = user_input.split()
            normie = ['hi', 'hey', 'hello' 'halo']
            for word in user_question_words:
                if word.lower() in normie:
                    main_answer = "Hello World"
                else:    
                    context = answer_question_with_context(word, tfidf_vectorizer, tfidf_matrix, sentences, word_limit)
                    if context:
                        formatted_context = format_conversation_response(context)
                        if main_answer != formatted_context:
                            main_answer += "\n" + formatted_context + ". "
            if main_answer:
                st.write("Bot:", main_answer)
            else :
                st.write("Bot:", "Sorry couldn't find anything related throughout PDF")

            question_counter += 1

if __name__ == "__main__":
    main()
