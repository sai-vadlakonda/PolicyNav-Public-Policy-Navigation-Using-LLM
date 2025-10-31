from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="template")

PDF_PATH = "uploaded.pdf"
VECTOR_PATH = "vector_store.pkl"

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_pdf")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    with open(PDF_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Process PDF and save vector store
    loader = PDFPlumberLoader(PDF_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)
    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embedder)
    # Save vector store to disk
    with open(VECTOR_PATH, "wb") as f:
        pickle.dump(vector, f)
    return JSONResponse({"message": "PDF uploaded successfully!"})

@app.post("/ask")
def ask_question(request: Request, question: str = Form(...)):
    if not os.path.exists(PDF_PATH) or not os.path.exists(VECTOR_PATH):
        return JSONResponse({"error": "No PDF uploaded."}, status_code=400)
    # Load vector store from disk
    with open(VECTOR_PATH, "rb") as f:
        vector = pickle.load(f)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    llm = ChatOpenAI(
        model="tinyllama-1.1b-chat-v1.0",
        temperature=0.7,
        openai_api_base=os.environ.get("OPENAI_API_BASE", "http://localhost:1234/v1"),
        openai_api_key=os.environ.get("OPENAI_API_KEY", "lm-studio"),
        request_timeout=60,
        verbose=True
    )
    prompt = """
You are a domain expert assistant.
Use the provided context to answer the question clearly and accurately.
If the answer cannot be found in the context, say 'The information is not available in the provided context.'
Provide a well-structured answer in 3–4 sentences and keep it factual.

Context:
{context}

Question:
{question}

Answer:
"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None,
    )
    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )
    result = qa(question)
    answer = result["result"]
    # Deduplicate sources and format output
    sources = set()
    if "source_documents" in result:
        for doc in result["source_documents"]:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources.add(doc.metadata['source'])
            elif isinstance(doc, dict) and 'metadata' in doc and 'source' in doc['metadata']:
                sources.add(doc['metadata']['source'])
    sources_list = list(sources)
    sources_str = ", ".join(sources_list) if sources_list else "uploaded.pdf"
    return JSONResponse({
        "answer": answer,
        "sources": sources_str
    })
