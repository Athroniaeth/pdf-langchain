import logging
from typing import List

from fitz import Document
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src._typing import History
from src.models import get_llm_model


class RagClient:
    """ Rag client LangChain pipeline. """

    pipeline: Runnable

    db_vector: Chroma
    llm_model: BaseLLM
    prompt_rag: BasePromptTemplate
    embeddings_model: HuggingFaceEmbeddings

    list_document_context: List[Document]

    def __init__(
            self,
            model_id: str,
            hf_token: str,

            id_prompt_rag: str = "athroniaeth/rag-prompt-mistral-custom-2",

            models_kwargs: dict = None,
            search_kwargs: dict = None,
    ):
        self.list_document_context = []

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

        splitter = RecursiveCharacterTextSplitter(chunk_size=1048)
        chunks = splitter.split_documents(documents)
        length_chunks = len(chunks)

        logging.debug(f"Split {length_chunks} chunks (ratio: {length_chunks / length_documents:.2f} chunks/page)")
        self.db_vector = Chroma.from_documents(chunks, embedding=self.embeddings_model)

    def clean_pdf(self):
        if self.db_vector is not None:
            # https://github.com/langchain-ai/langchain/discussions/9495#discussioncomment-7451042
            for collection in self.db_vector._client.list_collections():  # noqa
                ids = collection.get()['ids']
                if len(ids): collection.delete(ids)

    def invoke(
            self,
            message: str,
            history: History
    ) -> (
            str
    ):
        # Merge history, and add user message
        query = ""
        """
        for chat_message in history:
            role = chat_message.role
            content = chat_message.content

            if role == "user":
                query += f"User: {content}\n"

            elif role == "assistant":
                query += f"Assistant: {content}\n"

        query += f"User: {message}\nAssistant:"
        """
        query = message

        # It's just a joke
        if query == "42":
            return "The answer to the ultimate question of life, the universe, and everything is 42.", []

        # Todo : Add inference logging decorator
        pipeline_output = self.pipeline.invoke(
            input={"input": f"{query}"},
        )

        llm_output = pipeline_output["answer"]
        self.list_document_context = pipeline_output["context"]

        logging.debug(f"Result of llm model :\n\"\"\"\n{llm_output}\n\"\"\"")

        return llm_output
