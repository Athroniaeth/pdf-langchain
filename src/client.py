import logging
import uuid
from typing import Optional
from uuid import UUID

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.language_models import BaseLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src._typing import History
from src.models import get_llm_model


class RagClient:
    """ Rag client LangChain pipeline. """
    llm_model: BaseLLM
    embeddings_model: HuggingFaceEmbeddings

    def __init__(
            self,
            model_id: str,
            hf_token: str,
            id_prompt_rag: str = "athroniaeth/rag-prompt-mistral-custom-2",
            models_kwargs: dict = None,
    ):
        if models_kwargs is None:
            models_kwargs = {'max_length': 512}

        self.llm_model = get_llm_model(
            model_id=model_id,
            hf_token=hf_token,
            max_new_tokens=512,
            **models_kwargs
        )

        self.embeddings_model = HuggingFaceEmbeddings()
        self.prompt_rag = hub.pull(id_prompt_rag)

    def process_pdf(
            self,
            file_path: str,
            state_uuid: Optional[UUID] = None
    ) -> UUID:
        # Utiliser l'état utilisateur pour obtenir un identifiant unique
        state_uuid = self.get_unique_user_key(state_uuid)
        user_db = self.get_user_db(state_uuid)

        # Code pour traiter le fichier PDF et mettre à jour user_db
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1048)
        chunks = splitter.split_documents(documents)
        user_db.add_documents(chunks)
        # ...
        print(f"PDF processed for user with key {state_uuid}")
        return state_uuid

    @staticmethod
    def get_unique_user_key(state_uuid: Optional[UUID] = None) -> UUID:
        if state_uuid is None:
            return uuid.uuid4()
        return state_uuid

    def get_user_db(self, user_key: UUID) -> Chroma:
        user_chroma_client = Chroma(
            embedding_function=self.embeddings_model,
            collection_name=f"{user_key}",
            persist_directory="db/"
        )
        return user_chroma_client

    def invoke(
            self,
            message: str,
            history: History,
            state_uuid: Optional[UUID] = None,
    ):
        search_kwargs = {
            "k": 3,  # Amount of documents to return
            # "score_threshold": 0.5,  # Minimum relevance threshold for similarity_score_threshold
            # "fetch_k": 20,  # Amount of documents to pass to MMR algorithm
            # "lambda_mult": 0.5,  # Diversity of results returned by MMR
            # "filter": {'metadata_key': 'metadata_value'}  # Filter by document metadata
        }

        state_uuid = self.get_unique_user_key(state_uuid)
        db_vector = self.get_user_db(state_uuid)

        retreiver = db_vector.as_retriever(search_kwargs=search_kwargs)

        combine_docs_chain = create_stuff_documents_chain(
            self.llm_model,
            self.prompt_rag
        )

        # LangChain 'rag_chain' example "langchain\chains\retrieval.py"
        pipeline = create_retrieval_chain(
            retreiver,
            combine_docs_chain
        )

        query = message

        # Todo : Add inference logging decorator
        pipeline_output = pipeline.invoke(
            input={"input": f"{query}"},
        )

        llm_output = pipeline_output["answer"]
        list_document_context = pipeline_output["context"]

        logging.debug(f"Result of llm model :\n\"\"\"\n{llm_output}\n\"\"\"")

        return state_uuid, llm_output, list_document_context

    def clean_pdf(self, state_uuid: UUID) -> UUID:
        state_uuid = self.get_unique_user_key(state_uuid)
        db_vector = self.get_user_db(state_uuid)

        for collection in db_vector._client.list_collections():  # noqa
            ids = collection.get()['ids']
            if len(ids): collection.delete(ids)

        return state_uuid
