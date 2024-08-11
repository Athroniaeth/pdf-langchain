import functools
import uuid
from typing import Optional, List
from uuid import UUID

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_chroma import Chroma
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseLLM
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import Field, BaseModel
from transformers import AutoTokenizer

from src import DATABASE_PATH
from src.models import get_llm_model

store = {}


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


def get_by_session_id(state_uuid: UUID, session_id: str = "") -> BaseChatMessageHistory:
    """Get the chat history by session ID."""
    # session_id is forced by LangChain but useless in this case
    if state_uuid not in store:
        store[state_uuid] = InMemoryHistory()
    return store[state_uuid]


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
    tokenizer: AutoTokenizer
    embeddings_model: HuggingFaceEmbeddings

    def __init__(
            self,
            model_id: str,
            hf_token: str,
            id_prompt_rag: str = "athroniaeth/rag-prompt-mistral-custom-2",  # Todo : Permettre de modifier cela dans le CLI
            id_prompt_contextualize: str = "athroniaeth/contextualize-prompt",  # Todo : Permettre de modifier cela dans le CLI
            models_kwargs: dict = None,
    ):
        if models_kwargs is None:
            models_kwargs = {"max_length": 256}

        self.llm_model = get_llm_model(model_id=model_id, hf_token=hf_token, max_new_tokens=512, **models_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.embeddings_model = HuggingFaceEmbeddings()

        self.prompt_rag = hub.pull(id_prompt_rag)
        self.contextualize_q_prompt = hub.pull(id_prompt_contextualize)

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
        """compressor = FlashrankRerank()
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=db_vector.as_retriever(search_kwargs={'k': 3}),
        )"""
        retriever = db_vector.as_retriever(search_kwargs={'k': 3})

        # Create RAG pipeline, LangChain 'rag_chain' example "langchain\chains\retrieval.py"
        combine_docs_chain = create_stuff_documents_chain(
            self.llm_model,
            self.prompt_rag,
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm_model,
            retriever,
            self.contextualize_q_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            combine_docs_chain
        )

        pipeline = RunnableWithMessageHistory(
            rag_chain,
            get_by_session_id,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # Todo : Add inference logging decorator
        pipeline_output = pipeline.invoke(
            input={"input": f"{message}"},
            config={"configurable": {"session_id": state_uuid}}
        )

        # Get the output from the pipeline
        llm_output = pipeline_output["answer"]
        list_document_context = pipeline_output["context"]

        logger.debug(f'Result of llm model :\n"""\n{llm_output}\n"""')

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

    def clean_history(self, state_uuid: UUID) -> UUID:
        """Gradio pipeline, clean the user chat history."""
        state_uuid = self.get_unique_user_key(state_uuid)
        user_history = get_by_session_id(state_uuid)
        user_history.clear()
        return state_uuid

    def undo_history(self, state_uuid: UUID) -> UUID:
        """Gradio pipeline, undo the last message from the user chat history."""
        state_uuid = self.get_unique_user_key(state_uuid)
        user_history = get_by_session_id(state_uuid)

        if len(user_history.messages) > 0:
            user_history.messages.pop()
            user_history.messages.pop()

        return state_uuid
