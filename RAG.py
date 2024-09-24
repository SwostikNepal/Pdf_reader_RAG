import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from PyPDF2 import PdfReader
from langchain_cohere import CohereEmbeddings
from langchain_cohere.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Define API key for Cohere
api_key = os.getenv('API_KEY')

# Global variables for vector and retriever
vector = None
retriever = None

# Helper function to extract text from a PDF file
def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        pdf_reader = PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")
    finally:
        file.file.close()

# Endpoint to upload a PDF file and create the document store
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vector, retriever

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(file)

    # Create documents object from extracted text
    docs = [Document(page_content=pdf_text)]

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    text_chunks = text_splitter.split_documents(docs)

    # Define the embedding model
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=api_key)

    # Create the vector store
    vector = FAISS.from_documents(text_chunks, embeddings)

    # Define the retriever interface
    retriever = vector.as_retriever()

    return {"message": f"PDF '{file.filename}' uploaded and processed successfully."}


# Endpoint to query the uploaded PDF
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_document(query: QueryRequest):
    global vector, retriever

    if not vector or not retriever:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded yet.")

    question = query.question
    
    # Define the LLM model
    llm = Cohere(cohere_api_key=api_key)

    # Define the prompt template
    template = """"
```You are an AI assistant tasked with answering questions strictly based on the content provided in the PDF document above. Follow these instructions carefully:```

Thoroughly examine the provided context to determine whether the question is answered within the PDF content.

If the answer to the question is found within the context:

Provide a clear and concise answer based solely on the information in the context.
Do not include any information that is not present in the given context.

If the answer is not mentioned in the provided context:

State explicitly: "The information about [topic] is not mentioned in the provided context."
If partial information is available:

Provide the partial answer and specify: "This is partial information based on the provided context."
Additional Guidelines:
Only use the exact information from the context.
No assumptions or inference beyond the content should be made.
Context: {context}
Question: {input}
"""
    
    # Create the prompt from the template
    prompt = ChatPromptTemplate.from_template(template)

    # Create the document chain
    doc_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    # Query the document
    response = retrieval_chain.invoke({"input": question})

    return {"answer": response["answer"]}
