__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Import the necessary modules
from llm_setup import get_vectors_store, setup_conversational_chain
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
from streamlit_chat import message
import openai
import os

# Set the layout of the page to wide
st.set_page_config(layout="wide")
# Set the OpenAI API key from environment variables
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the path where FAISS index will be stored
chroma_db_path = './db'
# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings()
# Get the vectors store from the FAISS index
vectors = get_vectors_store(chroma_db_path, embeddings)

# Setup the conversational chain using the vectors
chain = setup_conversational_chain(vectors)

# Function to process a query and get a response
def conversational_chat(query):

    # Get the result from the chain by providing the query and the chat history
    result = chain({"question": query, "chat_history": st.session_state['history']})
    # Append the query and response to the chat history
    st.session_state['history'].append((query, result["answer"]))

    # Return the response
    return result["answer"]

# Initialize the session state variables for history, generated responses and past messages
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about IFRS17 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

# Set up containers for the chat history and the user's text input
response_container = st.container()
container = st.container()

with container:
    # Create a form for user input
    with st.form(key='my_form', clear_on_submit=True):
        # Get user input as a text input
        user_input = st.text_input("Query:", placeholder="What do you wanna talk about?", key='input')
        # Set up a submit button for the form
        submit_button = st.form_submit_button(label='Send')

    # If the submit button is pressed and the user input is not empty, get a response
    if submit_button and user_input:
        output = conversational_chat(user_input)

        # Append the user input and the response to the respective session state variables
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# If there are generated responses, display them
if st.session_state['generated']:
    with response_container:
        # Loop over all the generated responses
        for i in range(len(st.session_state['generated'])):
            # Display the user message and the response from the bot
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

# To run the chatbot, use: streamlit run name_of_your_chatbot.py
