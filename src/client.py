import logging
from typing import List, Tuple

from fitz import Document
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import get_llm_model

store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    # Todo : Delete RunnableWithMessageHistory and remove it (history handle by gradio)
    return ChatMessageHistory()


class RagClient:
    """ Rag client LangChain pipeline. """

    pipeline: Runnable

    db_vector: Chroma
    llm_model: BaseLLM
    prompt_rag: BasePromptTemplate
    embeddings_model: HuggingFaceEmbeddings

    def __init__(
            self,
            model_id: str,
            hf_token: str,

            id_prompt_rag: str = "athroniaeth/rag-prompt-mistral-custom-2",

            models_kwargs: dict = None,
            search_kwargs: dict = None,
    ):
        if models_kwargs is None:
            models_kwargs = {'max_length': 512}

        if search_kwargs is None:
            search_kwargs = {}

        self.llm_model = get_llm_model(
            model_id=model_id,
            hf_token=hf_token,
            max_new_tokens=512,
            **models_kwargs
        )

        self.embeddings_model = HuggingFaceEmbeddings()
        self.db_vector = Chroma(embedding_function=self.embeddings_model)

        self.prompt_rag = hub.pull(id_prompt_rag)  # "athroniaeth/rag-prompt")

        if search_kwargs is None:
            search_kwargs = {
                "k": 3,  # Amount of documents to return
                "score_threshold": 0.5,  # Minimum relevance threshold for similarity_score_threshold
                "fetch_k": 20,  # Amount of documents to pass to MMR algorithm
                "lambda_mult": 0.5,  # Diversity of results returned by MMR
                # "filter": {'metadata_key': 'metadata_value'}  # Filter by document metadata
            }

        self.retreiver = self.db_vector.as_retriever(search_kwargs=search_kwargs)  # noqa

        combine_docs_chain = create_stuff_documents_chain(
            self.llm_model,
            self.prompt_rag
        )

        # LangChain 'rag_chain' example "langchain\chains\retrieval.py"
        self.pipeline = create_retrieval_chain(
            self.retreiver,
            combine_docs_chain
        )

    def load_pdf(self, path: str):
        """ Load a PDF file and create a vector representation of the text. """
        loader = PyMuPDFLoader(path)

        documents = loader.load()
        length_documents = len(documents)
        logging.debug(f"Loaded {length_documents} pages from '{path}'")

        splitter = RecursiveCharacterTextSplitter(chunk_size=2048)
        chunks = splitter.split_documents(documents)
        length_chunks = len(chunks)

        logging.debug(f"Split {length_chunks} chunks (ratio: {length_chunks / length_documents:.2f} chunks/page)")
        self.db_vector = Chroma.from_documents(chunks, embedding=self.embeddings_model)

    def clean_pdf(self):
        if self.db_vector is not None:
            self.db_vector = Chroma(embedding_function=self.embeddings_model)

    def invoke(self, query: str) -> Tuple[str, List[Document]]:
        # It's just a joke
        if query == "42":
            return "The answer to the ultimate question of life, the universe, and everything is 42.", []

        # Todo : Add inference logging decorator
        pipeline_output = self.pipeline.invoke(
            input={"input": f"{query}"},
        )

        llm_output = pipeline_output["answer"]
        list_document_context = pipeline_output["context"]

        logging.debug(f"Result of llm model :\n\"\"\"\n{llm_output}\n\"\"\"")

        return llm_output, list_document_context
