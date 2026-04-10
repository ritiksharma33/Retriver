from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from app.services.embedder import get_vectorstore
from app.core.config import settings

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",           # Free tier model
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0.3,
    convert_system_message_to_human=True
)

# One memory instance per server session (extend to per-user with a dict keyed by session_id)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
    k=6  # keep last 6 exchanges
)

def get_chain():
    retriever = get_vectorstore().as_retriever(
        search_kwargs={"k": settings.TOP_K}
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )