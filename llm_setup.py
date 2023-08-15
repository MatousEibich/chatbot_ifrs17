# import the necessary modules
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain import PromptTemplate
import chromadb

# This function is used to get vectors store. It takes in the path for the ChromaDB and the embeddings as input.
def get_vectors_store(chroma_db_path, embeddings):
    try:
        # Try to load the ChromaDB from disk
        vectors = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)
    except (RuntimeError, TypeError) as e:
        # If an error occurs, raise an error
        raise RuntimeError("Unable to load ChromaDB")
    # Return the vectors
    return vectors



# This function sets up a conversational chain. It takes in the vectors as input.
def setup_conversational_chain(vectors):
    # # Initialize the OpenAI model with specified parameters
    # # Using 0 temperature for low randomness and gpt-3.5-turbo for fast compression
    # llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # # Create the compressor
    # # https: // python.langchain.com / docs / modules / data_connection / retrievers / contextual_compression /
    # compressor = LLMChainExtractor.from_llm(llm)
    #
    # # Create a retriever with the vector store and set its search parameters
    # # https://python.langchain.com/docs/modules/model_io/prompts/example_selectors/mmr
    # retriever = vectors.as_retriever(search_type = "mmr")
    # # This is kinda a parameter, with higher number results could be better, but slower
    # retriever.search_kwargs = {'k': 8}
    #
    # # Create a contextual compression retriever with the base compressor and base retriever
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=retriever
    # )

    # Define the prompt template for the conversation
    template = """The following is a conversation between a human and an AI. The AI acts as an assistant for IFRS17 legislature. 
     The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    Both the question and context are about IFRS17, which is a very complicated legal topic. 
    You may often need to consider multiple pieces of context together to come up with the final answer.

    Current conversation:
    {chat_history}
    Context: 
    {context}
    Human: {question}
    AI:"""
    # Initialize the PromptTemplate with specified input variables and the template
    PROMPT = PromptTemplate(input_variables=["chat_history", "question", "context"], template=template)

    # Initialize the chat model with specified parameters
    # Using 0 temperature for low randomness and gpt-4 for final answer
    chat_llm = ChatOpenAI(temperature=0.0, model_name='gpt-4')

    # Create a conversational retrieval chain with the chat model, retriever, and other specified parameters
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_llm,
        retriever=vectors.as_retriever(search_kwargs={'k': 8}),
        combine_docs_chain_kwargs={"prompt": PROMPT},
        # Using gpt-3.5-turbo for fast condesing of question
        # https://python.langchain.com/docs/modules/chains/popular/chat_vector_db
        condense_question_llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo'),
        verbose=True
    )

    # Return the chain
    return chain
