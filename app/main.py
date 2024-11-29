from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


VALID_EGOS = [
    "Ego del Poderoso", "Ego del Ganador", "Ego del Sabio", "Ego del Popular",
    "Ego del Amado", "Ego del Héroe", "Ego del Débil", "Ego del Perdedor",
    "Ego del Tonto", "Ego del Intimidado", "Ego del Engañado", "Ego de la Víctima"
]

# Initialize FastAPI app
app = FastAPI()

# Setup variables
qa_chain = None
vectorstore = None


def validate_response(response):
    for ego in VALID_EGOS:
        if ego.lower() in response.lower():
            return True
    return False


def search_scriptures(question, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    result = retriever.get_relevant_documents(question)
    if result:
        return result[0].page_content
    return None


def setup_rag():
    try:
        loader = TextLoader('data/raw/biblia.txt', encoding='utf-8')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        embeddings = OllamaEmbeddings(model="llama3.2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        llm = OllamaLLM(model="llama3.2", temperature=0.7, top_p=0.8, top_k=100,num_predict=100)

        prompt_template = """
        Eres un consejero bíblico cálido y empático basado en el texto 'La Auténtica Felicidad'.
        Tu tarea es identificar los egos dominantes en las preguntas y guiar con una respuesta breve y cálida. 

        ### Instrucciones:
        1. Responde en un máximo de 2-3 oraciones.
        2. Identifica claramente el ego dominante.
        3. Ofrece orientación breve, usando solo información relevante del texto.


        ### EJEMPLOS:
        Pregunta: "Odio a mis compañeros porque siempre tienen más éxito que yo."
        Respuesta: "Esto refleja el Ego del Ganador, que busca validación en la comparación con otros. Jesús nos invita a valorar nuestro esfuerzo y encontrar paz en nuestro propio camino."

        Pregunta: "Siempre me preocupa lo que los demás piensan de mí."
        Respuesta: "Esto es el Ego del Popular, que busca aceptación externa. La verdadera felicidad está en aceptar tu esencia sin depender de la aprobación de los demás."

        Pregunta: {question}

        Contexto: {context}

        Respuesta:
        """
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=False
        )
        return qa_chain, vectorstore
    except Exception as e:
        raise RuntimeError(f"Error configurando RAG: {str(e)}")


class Query(BaseModel):
    question: str


@app.on_event("startup")
async def startup_event():
    global qa_chain, vectorstore
    qa_chain, vectorstore = setup_rag()


@app.post("/chat")
async def chat_endpoint(query: Query):
    global qa_chain, vectorstore
    question = query.question

    if not question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    
    result = qa_chain({"query": question})
    response = result["result"]


    if validate_response(response):
        return {"reflection": response}
    else:
        verse = search_scriptures(question, vectorstore)
        if verse:
            return {"reflection": verse}
        else:
            return {"reflection": "No se encontró un versículo relacionado. Intenta con otra pregunta."}
