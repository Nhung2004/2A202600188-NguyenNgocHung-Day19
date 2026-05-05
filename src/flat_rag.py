from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import DATA_PATH
from dotenv import load_dotenv

load_dotenv()

class FlatRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vector_store = None
        self.qa_chain = None

    def build_index(self):
        print("Reading corpus for Flat RAG...")
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read()

        print("Chunking text...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
        chunks = splitter.create_documents([text])
        
        print(f"Created {len(chunks)} chunks. Building VectorDB (Chroma)...")
        # Lưu index vào memory (có thể cấu hình persist_directory nếu muốn lưu xuống đĩa)
        self.vector_store = Chroma.from_documents(chunks, self.embeddings)
        
        # Setup Retrieval Chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        template = """Dựa vào ngữ cảnh sau đây, hãy trả lời câu hỏi. 
Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói "Tôi không biết".

Ngữ cảnh:
{context}

Câu hỏi: {question}
Trả lời:"""
        prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        self.qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("Flat RAG VectorDB built successfully.")

    def query(self, question: str) -> str:
        if not self.qa_chain:
            return "Lỗi: Vector DB chưa được khởi tạo."
        try:
            return self.qa_chain.invoke(question)
        except Exception as e:
            return f"Lỗi: {e}"
