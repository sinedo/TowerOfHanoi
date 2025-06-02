# Retrieval-Augmented Generation

## 1. Setup

To setup the LangChain RAG system, the instructions on https://python.langchain.com/v0.2/docs/tutorials/local_rag/ are followed. This includes running the following commands in the docker container:

```bash
# General purpose model
ollama pull llama3.1:8b
# Text embedding model
ollama pull nomic-embed-text
# Document loading, retrieval methods and text splitting
pip3 install -U langchain langchain_community
# Local vector store via Chroma
pip3 install -U langchain_chroma
# Local inference and embeddings via Ollama
pip3 install -U langchain_ollama
# Web Loader
pip3 install -U beautifulsoup4
```

To get `langchain_chroma` working, you must perform the following commands

```bash
# remove python3 version of protobuf to use python3.8 version instead
rm -r /usr/lib/python3/dist-packages/google/protobuf/
# make sure protobuf is installed
pip3 install protobuf
pip3 install sqlite3-binary
```

Before executing any commands, initialize `conda` using

```bash
/opt/conda/bin/conda init
```

This command could be added to the `Dockerfile`. After that, the environment can be activated using `conda activate llm_env`.

Then execute the following commands in `python3` after switching to the conda environment:

```python
# set sqlite3 to pysqlite3, to make the version compatible
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Import an example document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Initialize a vector store
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)
```

## 2. Creating a knowledge base

## 3. Evaluation

### Who is the current pope?

#### Base Model

As of today, November 2, 2023, the current Pope is **Pope Francis**.

He was elected in March 2013 and continues to lead the Catholic Church.

You can find more information about him and the Vatican on the official Vatican website: [https://www.vatican.va/](https://www.vatican.va/)
