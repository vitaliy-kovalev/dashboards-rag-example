### What is it?
It's a simple CLI implementation of RAG to search through reports. It's made to support this Medium article:

### Project structure and steps:
```text
├── README.md ................... 0. Read the docs
├── requirements.txt ............ 1. Install all those packages: pip install -r requirements.txt
├── docs ........................ 2. Here you could put your own documents with the same structure to test it out
│ ├── finance_tab.json
│ ├── logistics_tab.json
│ └── sales_tab.json
├── prompt.md ................... 3. Here you could add your own business context and template for the answer
├── splitter_and_converter.py ... 4. Launch this to split your documents and create FAISS vectorstore
└── ask_me_anything.py .......... 5. Use this CLI to test the results
```

### Details for step 4: how to create local vectorstore and what are the options?

```commandline
Usage: python3 -m splitter_and_converter

Options:
  -mp,  --metadata_path       PATH     Path to the folder with test metadata of the dashboards
  -m,   --hf_embedding_name   TEXT     Path to the embedding model on Hugging Face, default is E5 by Microsoft Reasearch Lab
  -vo,  --vectorstore_output  PATH     Where to save FAISS vector base
  -cs,  --chunk_size          INTEGER  Embedding model limitation (depends on the model), default 512
  -ams, --add_meta_size       INTEGER  How much additional meta would be added before the next chunk, default 90
```

### Details for step 5: how to ask questions and which options could you use?
Before using it you should put your [OpenAI API key](https://platform.openai.com/api-keys) to the environment variable: 
```commandline
export OPENAI_API_KEY=<your_key_here>
```
Then just run it:
```commandline
Usage: python3 -m ask_me_anything -q 'Where to find sales by Regions?'

Options:
  -q,  --question           TEXT     Question
  -k,  --retriever_k        INTEGER  Top-k documents to use it in context
  -v,  --vectorstore        PATH     Vector storage folder
  -pl, --prompt_location    PATH     Prompt folder
  -m,  --hf_embedding_name  TEXT     Hugging Face model name
```
