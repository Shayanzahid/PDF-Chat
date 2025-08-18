from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

app = FastAPI()

# Global session state (like your Streamlit session_state)
conversation_chain = None
memory = None


def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def build_chain(vector_store):
    global memory
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the retrieved context to answer."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("system", "Context:\n{context}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=document_chain
    )

    def conversation(inputs):
        history = memory.load_memory_variables({})["chat_history"]

        response = retrieval_chain.invoke({
            "input": inputs["input"],
            "chat_history": history
        })

        memory.save_context(
            {"input": inputs["input"]},
            {"output": response["answer"]}
        )

        return {
            "answer": response["answer"],
            "chat_history": memory.load_memory_variables({})["chat_history"]
        }

    return conversation


@app.post("/process-pdf/")
async def process_pdf(file: UploadFile = File(...)):
    global conversation_chain

    raw_text = get_pdf_text(file.file)
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

    conversation_chain = build_chain(vector_store)
    return {"message": "PDF processed successfully. You can now ask questions."}


class Question(BaseModel):
    question: str


@app.post("/ask/")
async def ask_question(payload: Question):
    global conversation_chain
    if conversation_chain is None:
        return {"error": "No PDF processed yet."}

    response = conversation_chain({"input": payload.question})
    return {"answer": response["answer"]}
