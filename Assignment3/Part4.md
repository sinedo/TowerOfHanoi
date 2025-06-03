install conda from /opt/conda/bin  

```bash
/opt/conda/bin/conda init
```

create and activate new conda environment 

``` bash 
conad create -n rag_env python=3.11
conad activate rag_env

```

install python packages.

``` bash
# Document loading, retrieval methods and text splitting
pip install -qU langchain langchain_community

# Local vector store via Chroma
pip install -qU langchain_chroma
pip install -qU langchain_ollama
pip install -qU beautifulsoup4

```
now run the python script
``` bash
python3 rowing_rag.py

```

## Comparison 

### The question "who is faster yale or harvard" was given to both the native gemma3:4b and the gemma3:4b+rag.

#### Native gemma3:4b

**Answer:**

<SOLUTION>Based on recent data and rankings, Yale is generally considered the faster school academically. Yale has a significantly higher graduation rate and a faster pace of completion for its degrees compared to Harvard.  

However, both are incredibly prestigious universities!  

Do you want me to elaborate on specific metrics that demonstrate this difference? 

#### Rag: gemma3:4b

**Answer:**

<SOLUTION>Yale is faster. The context states Yale holds the upstream course record with a time of 18:35.8, while Harvard’s record is 18:22.4.

---
### The question "who won more consecutive regattas ?" was given to both the native gemma3:4b and the gemma3:4b+rag.

#### Native gemma3:4b

**Answer:**

<SOLUTION>Please provide me with the context! I need the text you’re referring to in order to answer the question about who won more consecutive regattas.

#### Rag: gemma3:4b

**Answer:**

<SOLUTION>Harvard won more consecutive regattas, winning 18 consecutively from 1963 to 1980.


---
### The question "who won the last rowing regatta yale or harvard ? (2024)" was given to both the native gemma3:4b and the gemma3:4b+rag.

#### Native gemma3:4b

**Answer:**

<SOLUTION> As of today, November 2, 2024, Harvard won the 2024 Harvard-Yale Regatta.  

You can find the full results here: https://harvardcrimson.com/news/2024/09/29/harvard-beats-yale-in-annual-rowing-race/  

#### Rag: gemma3:4b

**Answer:**

<SOLUTION> The text doesn’t provide information about the 2024 Yale-Harvard Regatta. It states there was no regatta in 2020 due to the COVID-19 pandemic, and doesn’t offer any details about recent races.


### Analysis:

Both queries were wrapped in a bigger prompts
``` 
#without rag
You are a helpful assistant. If unsure, say you don’t know.
question: {question}

#with rag
You are a helpful assistant. Use the context below to answer the question.
If unsure, say you don’t know.

<context>
{context}
</context>

Question: {question}
"""

``` 

The system with the rag correctly reads most of the data and answers with the context of the input files. However it did not read the tables well, even if specifically prompted and therefore fails to answer the last question while the native model did answer it correctly but provided a dummy link

When asked about facts not mentioned in the documents it fails to give an answer. Who is the current pope was answered with i dont know, while the native model answered with the outdated info "francis" => prompting can be done better