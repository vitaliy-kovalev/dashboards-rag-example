import os
import time
from datetime import datetime, timezone
import json
import click
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
import hashlib


def convert(
        base: list[dict]
) -> list[Document]:
    docs = []
    for row in base:
        text = row.get('md')
        metadata = row.get('meta')
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def enrich_with_metadata(doc: Document) -> Document:
    meta = doc.metadata
    if "source" in meta and "start_index" in meta:
        meta["uid"] = f"{meta['source']}:{meta['start_index']}"
    else:
        meta["uid"] = hashlib.md5(
            doc.page_content.encode("utf-8")
        ).hexdigest()

    meta_text = '\n'.join([
        "--- PART ---",
        f"Domain: {doc.metadata.get('domain', '')}",
        f"Tab Name: {doc.metadata.get('tab_name', '-')}",
        f"URL: {doc.metadata.get('url', '-')}",
        f"Number of views: {doc.metadata.get('number_of_views', 0)}",
        '---'
    ]).strip()
    if '## About this tab:' in doc.page_content:
        concatenation = doc.page_content
    else:
        concatenation = meta_text + "\n" + doc.page_content
    return Document(
        page_content=concatenation,
        metadata=meta
    )


@click.command()
@click.option(
    '--metadata_path', '-mp',
    default='./docs',
    type=click.Path(exists=True),
    help='Path to the folder with test metadata of the dashboards'
)
@click.option(
    '--hf_embedding_name', '-m',
    default='intfloat/multilingual-e5-base',
    type=click.STRING,
    help='Path to the embedding model on Hugging Face'
)
@click.option(
    '--vectorstore_output', '-vo',
    default=f"./rag",
    type=click.Path(),
    help='Where to save FAISS vector base'
)
@click.option(
    '--chunk_size', '-cs',
    default=512,
    type=click.INT,
    help='Embedding model limitation (depends on model)'
)
@click.option(
    '--add_meta_size', '-ams',
    default=90,
    type=click.INT,
    help='How much additional meta would be added before the next chunk'
)
def main(
    metadata_path: str,
    hf_embedding_name: str,
    vectorstore_output: str,
    chunk_size: int,
    add_meta_size: int
):
    click.echo(f"{datetime.now(timezone.utc)} Started {__package__} with parameters {vars()}")
    start = time.time()

    click.echo(f"{datetime.now(timezone.utc)} Getting base...")
    base = []

    for filename in os.listdir(metadata_path):
        if filename.endswith(".json"):
            file_path = os.path.join(metadata_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    base.append(json.load(f))
                except json.JSONDecodeError as e:
                    click.echo(f"{datetime.now(timezone.utc)} Error in file {filename}: {e}")

    docs = convert(base)

    click.echo(f"{datetime.now(timezone.utc)} Got {len(docs)} documents. Splitting...")
    tokenizer = AutoTokenizer.from_pretrained(hf_embedding_name)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=int(chunk_size - add_meta_size),
        chunk_overlap=0,
        separators=["\n## ", "\n### "],
        strip_whitespace=True,
        add_start_index=True,
        keep_separator="start"
    )
    chunks = splitter.split_documents(docs)

    chunks = [enrich_with_metadata(chunk) for chunk in chunks]
    click.echo(f"{datetime.now(timezone.utc)} 🔪 Split into {len(chunks)} chunks")

    too_big_details = [f"{doc.metadata.get('url')}: {len(doc.page_content)} / {chunk_size}" for doc in chunks
                       if len(tokenizer.encode(doc.page_content)) > chunk_size]
    too_big_cnt = len(too_big_details)

    if too_big_cnt > 0:
        click.echo(f"{datetime.now(timezone.utc)} 🔪 {round(float(too_big_cnt / len(chunks)) * 100, 2)}% ({too_big_cnt} / "
                   f"{len(chunks)}) chunks is too big for context of {chunk_size} tokens!")
        click.echo(f"{datetime.now(timezone.utc)} 🔪 Check those dashboard tabs: ")
        click.echo('\n'.join(list(set(too_big_details))))

    click.echo(f"{datetime.now(timezone.utc)} Building vector storage...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=hf_embedding_name,
        encode_kwargs={
            "normalize_embeddings": True
        }
    )
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(vectorstore_output)

    finish = time.time()
    click.echo(f"{datetime.now(timezone.utc)} Done. Elapsed: {round(finish - start, 1)} sec")


if __name__ == '__main__':
    main()
