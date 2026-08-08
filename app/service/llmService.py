from langchain_core import chat_history
from langchain_openai import ChatOpenAI
from langchain_classic.chains import(
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain_core.messages import HumanMessage, AIMessage


from config import Config

class LLMService:
    def __init__(self, vectore_store):
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0.7,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.chat_history = []

        retriever= vector_store.vector_store.as_retriever()



        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        )

        history_aware_retriever=create_history_aware_retriever(
            self.llm,
            retriever,
            contextualize_q_prompt
        )

        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

          # Build the full answer generation prompt template
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),           # System instruction + context
                MessagesPlaceholder("chat_history"),     # Injected chat history
                ("human", "{input}"),                    # The user's question
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            self.llm,       # The LLM used for answer generation
            qa_prompt        # The prompt template with {context} placeholder
        )

        self.rag_chain = create_retrieval_chain(
            history_aware_retriever,    # Retriever with question reformulation
            question_answer_chain       # Chain that generates answers from context
        )

    def get_response(self, query):
        response = self.rag_chain.invoke(
            {
                    "input": query,                      # The user's question
                    "chat_history": self.chat_history     # Conversation history so far
                }
            )

        answer = response["answer"]

        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=answer))

        return answer
