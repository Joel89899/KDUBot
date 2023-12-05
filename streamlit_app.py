import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Kyundong University Global Chatbot", page_icon="🤖", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with the KDU Global Bot assistant")
st.info("This is an assistant built on Data about Kyungdong University Global")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Kyungdong University Global"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the course docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
chat_engine = index.as_chat_engine(chat_mode="context", 
                                   verbose=True, 
                                   system_prompt= """You are a personal Bot assistant for answering any questions about Kyungdong University Global.
You are given a question and a set of documents. 
If the user's question requires you to provide specific information from the documents, give your answer based only on the examples provided below. DON'T generate an answer that is NOT written in the provided examples.
If you don't find the answer to the user's question with the examples provided to you below, answer that you didn't find the answer in the documentation and propose him to rephrase his query with more details. 
Provide polite greetings if user greets you.
Make your replies short and clear like in a text message chat. 
Do not give a long and general answer, dependinding the information you have you can ask for specifics to provide the specific answer. 
Use bullet points if you have to make a list, your lists have to br spacious and readable.

QUESTION: {question}

DOCUMENTS:
=========
{context}
=========
 
""")

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history


from trubrics.integrations.streamlit import FeedbackCollector

collector = FeedbackCollector(
    project="default",
    email=st.secrets.TRUBRICS_EMAIL,
    password=st.secrets.TRUBRICS_PASSWORD,
)

collector.st_feedback(
    component="default",
    feedback_type="thumbs",
    model="gpt-3.5-turbo",
    prompt_id=None,  # see prompts to log prompts and model generations
    open_feedback_label='[Optional] Provide additional feedback'
)