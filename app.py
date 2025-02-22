import streamlit as st
import nltk
from transformers.pipelines import pipeline


# Download NLTK dependencies
nltk.download("punkt")
nltk.download("stopwords")

# Load a healthcare-related question-answering model
qa_model = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Define medical context for better responses
medical_context = """
Doctors recommend staying hydrated and taking proper rest during fever. 
Paracetamol is used for reducing fever. 
For headaches, painkillers like ibuprofen can be taken in moderation. 
Consult a doctor if symptoms persist or worsen.
"""

def generate_response(user_input):
    """Uses a question-answering model to generate healthcare-related responses."""
    response = qa_model(question=user_input, context=medical_context)
    return response["answer"]

def main():
    st.title("Healthcare Assistant Chatbot")

    # User input
    user_input = st.text_input("How can I help you today?")
    
    if st.button("Submit"):
        if user_input:
            st.write("**User:**", user_input)
            with st.spinner("Processing your query, please wait..."):
                bot_response = generate_response(user_input)
            st.write("**Chatbot:**", bot_response)
        else:
            st.write("⚠️ Please enter a message to get a response.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
