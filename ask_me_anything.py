import os
import time
from datetime import datetime, timezone
from uuid import uuid4

import click
from jinja2 import Environment, FileSystemLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def render_template(
    env: Environment,
    template_name: str,
    **kwargs
) -> str:
    template = env.get_template(template_name)
    return template.render(**kwargs)


@click.command()
@click.option(
    '--question', '-q',
    type=click.STRING,
    default='Which report contains table with sales?',
    help='Question for RAG'
)
@click.option(
    '--retriever_k', '-k',
    type=click.INT,
    default=40,
    help='top-k documents to use it in context'
)
@click.option(
    '--vectorstore', '-v',
    default=f"./rag",
    type=click.Path(exists=True),
    help='Vector storage folder'
)
@click.option(
    '--prompt_location', '-pl',
    default='./',
    type=click.Path(exists=True),
    help='Prompt folder'
)
@click.option(
    '--hf_embedding_name', '-m',
    default='intfloat/multilingual-e5-base',
    type=click.STRING,
    help='Hugging Face model name'
)
def main(
    question: str,
    retriever_k: int,
    vectorstore: str,
    prompt_location: str,
    hf_embedding_name: str
):
    click.echo(f"{datetime.now(timezone.utc)} Started {__package__} with parameters {vars()}")
    this_id = str(uuid4())
    started_at_utc = datetime.now(timezone.utc)
    start = time.time()

    # Jinja2 Environment to load the prompt text and expand it with additional context later on
    jinja_loader = FileSystemLoader(prompt_location)
    jinja_env = Environment(loader=jinja_loader)

    # Same model that have been used in vectorstore creation
    embedding_model = HuggingFaceEmbeddings(
        model_name=hf_embedding_name,
        encode_kwargs={"normalize_embeddings": True}
    )

    click.echo("Retriever started...")

    retriever_start = time.time()

    # Similarity search stage
    db = FAISS.load_local(
        folder_path=vectorstore,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": retriever_k})

    unique_docs = retriever.invoke(question)

    click.echo(f"Done for {round(time.time() - retriever_start, 0)}. Unique docs cnt: {len(unique_docs)}")

    # Generation stage
    # Initializing the llm api (temperature=0 means that gpt would be less creative in the answers)
    click.echo("Choosing final answer...")
    llm = ChatOpenAI(
        model="gpt-4o",
        openai_api_key=os.environ.get('OPENAI_API_KEY', ''),
        temperature=0.0
    )

    prompt = render_template(
        env=jinja_env,
        template_name='prompt.md',
    )

    final_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt
    )

    chain = create_stuff_documents_chain(
        llm,
        final_prompt,
        document_variable_name="context"
    )

    result = chain.invoke({
        "question": question,
        "context": unique_docs
        }
    )

    click.echo(f"\n{result}\n")
    finished_dttm = datetime.now(timezone.utc)
    finish = time.time()
    click.echo(f"{finished_dttm} Request {this_id} started at {started_at_utc} Done at {finished_dttm}. "
               f"Elapsed: {round(finish - start, 1)} sec")


if __name__ == '__main__':
    main()
