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

from src import DATABASE_PATH
from src._typing import History
from src.models import get_llm_model


def history_to_template(history: History, message: str) -> str:
    """
    Convert the history of chat messages to a template for LLM.

    Args:
        history (History): List of chat messages.
        message (str): Current message from user.

    Returns:
        str: Template for LLM.
    """
    query = ""
    for chat_message in history:
        role = chat_message.role
        content = chat_message.content

        if role == "user":
            query += f"User: {content}\n\n"

        elif role in ["assistant", "system"]:
            query += f"Assistant: {content}\n\n"

    query += f"User: {message}\n\nAssistant:"
    return query


class RagClient:
    """
    Rag client LangChain pipeline.

    Args:
        model_id (str): The model ID of Hugging Face LLM.
        hf_token (str): The Hugging Face token.
        id_prompt_rag (str): The ID of the prompt for RAG model.
        models_kwargs (dict): The additional keyword arguments for the model.

    Attributes:
        llm_model (BaseLLM): The language model.
        embeddings_model (HuggingFaceEmbeddings): The embeddings model.
        prompt_rag (hub): The prompt for RAG model.
    """

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
            models_kwargs = {"max_length": 512}

        self.llm_model = get_llm_model(model_id=model_id, hf_token=hf_token, max_new_tokens=512, **models_kwargs)

        self.embeddings_model = HuggingFaceEmbeddings()
        self.prompt_rag = hub.pull(id_prompt_rag)

    def process_pdf(self, file_path: str, state_uuid: Optional[UUID] = None) -> UUID:
        """
        Gradio pipeline, process the PDF file and update the user database.

        Args:
            file_path (str): The path to the PDF file.
            state_uuid (Optional[UUID]): The unique user key.

        Returns:
            UUID: The unique user key
        """

        # Utiliser l'état utilisateur pour obtenir un identifiant unique
        state_uuid = self.get_unique_user_key(state_uuid)
        user_db = self.get_user_db(state_uuid)

        # Code pour traiter le fichier PDF et mettre à jour user_db
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1048)
        chunks = splitter.split_documents(documents)
        user_db.add_documents(chunks)

        print(f"PDF processed for user with key {state_uuid}")
        return state_uuid

    @staticmethod
    def get_unique_user_key(state_uuid: Optional[UUID] = None) -> UUID:
        """Gradio pipeline, get the unique user key."""
        if state_uuid is None:
            return uuid.uuid4()
        return state_uuid

    def get_user_db(self, user_key: UUID) -> Chroma:
        """Get the user database from UUID."""
        user_chroma_client = Chroma(
            embedding_function=self.embeddings_model,
            collection_name=f"{user_key}",
            persist_directory=f"{DATABASE_PATH.absolute()}",
        )
        return user_chroma_client

    def invoke(
        self,
        message: str,
        history: History,
        state_uuid: Optional[UUID] = None,
    ):
        """
        Gradio pipeline, invoke the RAG model.

        Args:
            message (str): The current message from user.
            history (History): List of chat messages.
            state_uuid (Optional[UUID]): The unique user key.

        Returns:
            UUID: The unique user key
            str: The response from the model.
            list: The list of document context.
        """
        search_kwargs = {
            "k": 3,  # Amount of documents to return
            # "score_threshold": 0.5,  # Minimum relevance threshold for similarity_score_threshold
            # "fetch_k": 20,  # Amount of documents to pass to MMR algorithm
            # "lambda_mult": 0.5,  # Diversity of results returned by MMR
            # "filter": {'metadata_key': 'metadata_value'} # Filter by document metadata.
        }

        # Get PDF content from user database
        state_uuid = self.get_unique_user_key(state_uuid)
        db_vector = self.get_user_db(state_uuid)

        # Create a retriever from the user database
        retreiver = db_vector.as_retriever(search_kwargs=search_kwargs)

        # Create RAG pipeline, LangChain 'rag_chain' example "langchain\chains\retrieval.py"
        combine_docs_chain = create_stuff_documents_chain(self.llm_model, self.prompt_rag)

        pipeline = create_retrieval_chain(retreiver, combine_docs_chain)

        # Transform history to template for LLM
        query = history_to_template(history, message)

        # Todo : Add inference logging decorator
        pipeline_output = pipeline.invoke(
            input={"input": f"{query}"},
        )

        # Get the output from the pipeline
        llm_output = pipeline_output["answer"]
        list_document_context = pipeline_output["context"]

        logging.debug(f'Result of llm model :\n"""\n{llm_output}\n"""')

        return state_uuid, llm_output, list_document_context

    def clean_pdf(self, state_uuid: UUID) -> UUID:
        """Gradio pipeline, clean the user database."""
        state_uuid = self.get_unique_user_key(state_uuid)
        db_vector = self.get_user_db(state_uuid)

        for collection in db_vector._client.list_collections():  # noqa
            ids = collection.get()["ids"]
            if len(ids):
                collection.delete(ids)

        return state_uuid
