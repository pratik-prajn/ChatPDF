import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_txt(docs):
    txt = ""
    for doc in docs:
        rdr = PdfReader(doc)
        for pg in rdr.pages:
            txt += pg.extract_text()
    return txt

def get_chks(txt):
    splt = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chks = splt.split_text(txt)
    return chks

def get_vs(chks):
    embd = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vs = FAISS.from_texts(texts=chks, embedding=embd)
    return vs

def get_chain(vs):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    mem = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vs.as_retriever(),
        memory=mem
    )
    return chain

def handle_qstn(qstn):
    rsp = st.session_state.chain({'question': qstn})
    st.session_state.hist = rsp['chat_history']
    for i, msg in enumerate(st.session_state.hist):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "hist" not in st.session_state:
        st.session_state.hist = None
    st.header("Chat with PDFs :books:")
    qstn = st.text_input("Ask a question about your documents:")
    if qstn:
        handle_qstn(qstn)
    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                txt = get_txt(docs)
                chks = get_chks(txt)
                vs = get_vs(chks)
                st.session_state.chain = get_chain(vs)

if __name__ == '__main__':
    main()
