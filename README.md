# Conversational Log Assistant

This Streamlit-based application enables users to upload log files and engage in a conversational interface to query and troubleshoot issues in the logs. It utilizes **OpenAI's language models** and **LangChain** to process and respond to queries based on the log data.

## Features

- Upload log files for real-time analysis
- Query logs using natural language
- Provides accurate and context-aware responses from logs
- Embedding-based search using OpenAI's API

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Chat-with-Logs.git
   ```

2. Install the required dependencies:
   ```bash
   pip install streamlit langchain openai chromadb
   ```

3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key'
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

- Open the app in your browser.
- Upload a log file and interact with it using conversational queries.
- The app will analyze and retrieve relevant parts of the logs to answer your questions.

