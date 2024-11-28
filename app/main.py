from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from rich.console import Console

console = Console()

VALID_EGOS = [
    "Ego del Poderoso", "Ego del Ganador", "Ego del Sabio", "Ego del Popular",
    "Ego del Amado", "Ego del Héroe", "Ego del Débil", "Ego del Perdedor",
    "Ego del Tonto", "Ego del Intimidado", "Ego del Engañado", "Ego de la Víctima"
]

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
        
        llm = OllamaLLM(model="llama3.2", temperature=0.7, top_p=0.9, top_k=100)

        prompt_template = """
        Eres un consejero bíblico cálido y empático basado en el texto 'La Auténtica Felicidad'.
        Tu tarea es identificar los egos dominantes en las preguntas y guiar con una respuesta breve y cálida. 
        Usa solo la sabiduría contenida en el texto y asegúrate de mencionar solo egos explícitamente listados en el contenido.

        ### INSTRUCCIONES:
        1. Identifica el ego dominante basado en la descripción en el texto.
        3. Responde con calidez, explicando el ego y ofreciendo orientación en máximo **2-3 oraciones.**
        4. Usa únicamente los egos mencionados en el texto proporcionado.
        5. No sugieras nada , solo enfoque en la reflexión y orientación.
        6. Responde de manera natural, no señales que identificaste el ego , ni coloques en tu respuesta respuesta con calidez, solo responde.

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
        console.print(f"[bold red]Error configurando RAG:[/bold red] {str(e)}")
        raise

def main():
    try:
        console.print("\nEspacio de Apoyo y Reflexión")
        console.print("Estoy aquí para escucharte y caminar juntos...\n")
        
        qa_chain, vectorstore = setup_rag()
        
        while True:
            question = console.input("\n¿Qué te preocupa hoy? (escribe 'salir' para terminar): ")
            
            if question.lower() == 'salir':
                break
                
            console.print(f"\nTú: {question}")
            result = qa_chain({"query": question})
            response = result['result']
   
            if "Respuesta:" in response:
                response = response.split("Respuesta:", 1)[1].strip()
            
            if validate_response(response):
                console.print(f"Reflexión: {response}")
            else:
                console.print("[bold yellow]No se detectó un ego válido en la respuesta.[/bold yellow]")
                verse = search_scriptures(question, vectorstore)
                if verse:
                    console.print(f"Reflexión basada en las Escrituras: {verse}")
                else:
                    console.print("No se encontró un versículo relacionado. Intenta con otra pregunta.")
            
            console.print("─" * 60 + "\n")
            
    except KeyboardInterrupt:
        console.print("\n\nGracias por compartir. ¡Hasta pronto!")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()

