import logging
import uuid
from typing import List, Tuple

from fitz import Document
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src._typing import History
from src.models import get_llm_model, log_inference

store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


class RagClient:
    def __init__(
            self,
            model_id: str,
            hf_token: str,

            id_prompt_rag: str = "athroniaeth/rag-prompt-mistral-custom-2",
            id_prompt_contextualize: str = "athroniaeth/contextualize-prompt",

            models_kwargs: dict = None,
            search_kwargs: dict = None,
    ):
        self.id = uuid.uuid4()

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
        self.prompt_contextualize = hub.pull(id_prompt_contextualize)  # "athroniaeth/contextualize-prompt")

        self.load_retriever(search_kwargs)
        self.load_pipeline()

    def load_retriever(self, search_kwargs: dict) -> None:
        if search_kwargs is None:
            search_kwargs = {
                "k": 3,  # Amount of documents to return
                "score_threshold": 0.5,  # Minimum relevance threshold for similarity_score_threshold
                "fetch_k": 20,  # Amount of documents to pass to MMR algorithm
                "lambda_mult": 0.5,  # Diversity of results returned by MMR
                # "filter": {'metadata_key': 'metadata_value'}  # Filter by document metadata
            }

        self.retreiver = self.db_vector.as_retriever(search_kwargs=search_kwargs)  # noqa

    def load_pipeline(self) -> None:
        history_aware_retriever = create_history_aware_retriever(
            self.llm_model,
            self.retreiver,
            self.prompt_contextualize
        )

        question_answer_chain = create_stuff_documents_chain(
            self.llm_model,
            self.prompt_rag
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        self.pipeline = RunnableWithMessageHistory(
            rag_chain,
            get_by_session_id,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def load_pdf(self, path: str):
        loader = PyMuPDFLoader(path)

        documents = loader.load()
        length_documents = len(documents)
        logging.debug(f"Loaded {length_documents} pages from '{path}'")

        splitter = RecursiveCharacterTextSplitter(chunk_size=512)
        chunks = splitter.split_documents(documents)
        length_chunks = len(chunks)

        logging.debug(f"Split {length_chunks} chunks (ratio: {length_chunks / length_documents:.2f} chunks/page)")
        self.db_vector = Chroma.from_documents(chunks, embedding=self.embeddings_model)

    def invoke(self, query: str) -> Tuple[str, List[Document]]:
        # Rafraichit le pipeline
        self.load_pipeline()

        with log_inference(self.id):
            pipeline_output = self.pipeline.invoke(
                input={"input": f"{query}"},
                config={"configurable": {"session_id": self.id}}
            )

            llm_output = pipeline_output["answer"]
            list_document_context = pipeline_output["context"]

        logging.debug(f"Result of llm model :\n\"\"\"\n{llm_output}\n\"\"\"")

        # Prend uniquement ce qui a été généré par le LLM
        generated_llm_output = llm_output  # """[length_prompt:]"""
        return generated_llm_output, list_document_context

    def respond(self, message: str, history: History) -> str:

        if message == "42":
            response = "The answer to the ultimate question of life, the universe and everything is 42."
            return response

        llm_output, list_document_context = self.invoke(message)
        return llm_output
