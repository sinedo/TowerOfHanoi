from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# local files to search in rag.
file_paths = [
    "rowing/Harvard_University.pdf",
    "rowing/Harvard–Yale_Regatta.pdf",
    "rowing/Rowing_sport.pdf",
    "rowing/Yale_University.pdf"
]
# laod and merge sources
all_docs = []
for path in file_paths:
    loader = PyPDFLoader(path)
    all_docs.extend(loader.load())
   
   
# split text, good for preprocessing ? 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(all_docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(splits, embedding=embeddings)


prompt_template = """
You are a helpful assistant. Use the context below to answer the question.
If unsure, say you don’t know.

<context>
{context}
</context>

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

model = ChatOllama(model="gemma3:4b")

qa_chain = (
    {
        "context": vectorstore.as_retriever() | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

question = "Most consecutive victories yale or harvard?"
answer = qa_chain.invoke(question)
print(answer)