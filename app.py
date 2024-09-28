import streamlit as st
import os
import re
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

import shutil
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = 'your-api-key'

# Constants
CHROMA_PATH = "chroma5"

Embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# App Titlepip install streamlit langchain openai chromadb

st.title("Conversational Log Assistant")

# Introduction message
st.write("Upload a log file to start the troubleshooting process!")

# Session state to store conversation history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Function to display chat history
def display_chat(messages):
    for role, message in messages:
        if role == 'user':
            st.write(f"**🧑 You:** {message}")
        else:
            st.write(f"**🤖 Assistant:** {message}")

# Function to parse and format logs
def parse_log_line(line):
    pattern = r"^(?P<Level>\w+)\s+(?P<DateTime>\d{2}/\d{2}/\d{4} \d{1,2}:\d{2}:\d{2} [APM]{2})\s+(?P<Source>[^\t]+)\s+(?P<EventID>\d+)\s+(?P<TaskCategory>[^\t]+)\t?(?P<Message>.*)$"
    match = re.match(pattern, line.strip())
    if match:
        return match.groupdict()
    return None

def format_log_entry(entry):
    return (
        f"Level: {entry['Level']}\n"
        f"Date and Time: {entry['DateTime']}\n"
        f"Source: {entry['Source']}\n"
        f"Event ID: {entry['EventID']}\n"
        f"Task Category: {entry['TaskCategory']}\n"
        f"Message: {entry['Message']}\n"
        f"{'-'*40}\n"
    )

def process_logs(input_content):
    output_content = ""
    for line in input_content.splitlines():
        log_entry = parse_log_line(line)
        if log_entry:
            output_content += format_log_entry(log_entry)
    return output_content

# Function to split the documents
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# Function to process file, create Chroma DB, and query LLM
def run_llm_query(user_input, file_content):
    # 1. Parse and format the logs
    formatted_logs = process_logs(file_content)

    # 2. Save formatted logs with UTF-8 encoding
    formatted_file_path = 'formatted_logs.txt'
    with open(formatted_file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_logs)

    # 3. Load the formatted log file
    doc_loader = TextLoader(formatted_file_path)
    documents = doc_loader.load()

    # 4. Split the document into chunks
    chunks = split_documents(documents)

    # 5. Add the chunks to Chroma DB (only once, reusing for subsequent queries)
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)  # Ensure the directory exists

    # If Chroma has not been initialized, initialize it
    if not hasattr(run_llm_query, 'chroma_db'):
        run_llm_query.chroma_db = Chroma.from_documents(
            chunks, persist_directory=CHROMA_PATH, embedding=OpenAIEmbeddings()
        )
        run_llm_query.chroma_db.persist()  # Persist the DB after first initialization

    # 6. Query the Chroma DB with the user's input
    db = run_llm_query.chroma_db
    results = db.similarity_search_with_relevance_scores(user_input, k=5)

    # 7. Prepare context for the LLM
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    # Format the prompt with the context and the question

    
    PROMPT_TEMPLATE = """
    You are an expert in system logs and records. Use the provided system logs below to answer the question:

    {context}

    ---

    Provide precise ans specific answers. Give the user all the infrmation that they require.
    Keep the asnwers short.
    Answer the question based on the above context of the system logs: {question}
    """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=user_input)

    # Call the LLM to generate a response
    model = ChatOpenAI(model="gpt-4o-mini")
    response_text = model.predict(prompt)
    
    return response_text

# User input text
user_input = st.text_input("Type your message here:", "")

# File upload
uploaded_file = st.file_uploader("Upload a log file", type=["txt"])

# Handle user input and file upload
if user_input and uploaded_file:
    # Read the uploaded file content
    file_content = uploaded_file.read().decode("utf-8")
    
    # Add user input and file info to the chat history
    st.session_state['messages'].append(('user', f"{user_input} (with file: {uploaded_file.name})"))

    # Process logs and get LLM response
    llm_response = run_llm_query(user_input, file_content)
    
    # Add assistant's response to the chat history
    st.session_state['messages'].append(('assistant', llm_response))
    print(llm_response)

# Display conversation
st.write("### Chat History")
display_chat(st.session_state['messages'])

# Clear conversation
if st.button("Clear Conversation"):
    st.session_state['messages'] = []
