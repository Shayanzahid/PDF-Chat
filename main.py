import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs): 
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()

    # Conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Prompt template with chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the retrieved context to answer."),
        ("placeholder", "{chat_history}"),   # put history before user
        ("human", "{input}"),
        ("system", "Context:\n{context}")
    ])

    # Stuff retrieved docs into the prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retrieval + QA chain
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=document_chain
    )

    # Conversation wrapper
    def conversation(inputs):
        # get past history
        history = memory.load_memory_variables({})["chat_history"]

        # run retrieval with history
        response = retrieval_chain.invoke({
            "input": inputs["input"],
            "chat_history": history
        })

        # save Q/A to memory
        memory.save_context(
            {"input": inputs["input"]},
            {"output": response["answer"]}
        )

        # return both answer and updated history
        return {
            "answer": response["answer"],
            "chat_history": memory.load_memory_variables({})["chat_history"]
        }

    return conversation


def handle_user_input(user_question):
    response = st.session_state.conversation_chain({'input': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF here and click on 'Process '", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vector_store = get_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation_chain = get_conversation_chain(vector_store)

if __name__ == '__main__':
    main()