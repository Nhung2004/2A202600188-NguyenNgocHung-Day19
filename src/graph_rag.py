import networkx as nx
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class EntityExtraction(BaseModel):
    entity: str = Field(description="Thực thể (Entity) chính được nhắc đến trong câu hỏi")

class GraphRAG:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def extract_entity_from_query(self, query: str) -> str:
        """Dùng LLM trích xuất entity chính từ câu hỏi."""
        structured_llm = self.llm.with_structured_output(EntityExtraction)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một hệ thống trích xuất thông tin. Hãy xác định MỘT thực thể (Entity) chính, quan trọng nhất trong câu hỏi sau để dùng làm từ khoá tìm kiếm trong Knowledge Graph. Trả về tên thực thể ngắn gọn."),
            ("human", "{question}")
        ])
        chain = prompt | structured_llm
        try:
            res = chain.invoke({"question": query})
            return res.entity
        except Exception as e:
            print(f"Lỗi extract entity: {e}")
            return ""

    def query(self, question: str) -> str:
        entity = self.extract_entity_from_query(question)
        print(f"  [GraphRAG] Từ khoá thực thể: '{entity}'")
        
        # Tìm node phù hợp nhất trong đồ thị
        matched_node = None
        if entity:
            for node in self.graph.nodes:
                if str(node).lower() == entity.lower() or entity.lower() in str(node).lower() or str(node).lower() in entity.lower():
                    matched_node = node
                    break

        if not matched_node:
            return f"Không tìm thấy thông tin về '{entity}' trong đồ thị tri thức."

        print(f"  [GraphRAG] Node khớp trong đồ thị: '{matched_node}'")
        
        # Duyệt đồ thị (2-hop BFS)
        context_parts = []
        neighbors_1hop = list(self.graph.neighbors(matched_node))
        
        for n1 in neighbors_1hop:
            rel1 = self.graph.edges[matched_node, n1].get('relation', 'liên quan tới')
            context_parts.append(f"{matched_node} {rel1} {n1}.")
            
            # Duyệt tiếp hop thứ 2
            neighbors_2hop = list(self.graph.neighbors(n1))
            for n2 in neighbors_2hop:
                if n2 != matched_node: # Tránh quay ngược lại node ban đầu
                    rel2 = self.graph.edges[n1, n2].get('relation', 'liên quan tới')
                    context_parts.append(f"{n1} {rel2} {n2}.")
        
        # Loại bỏ các quan hệ trùng lặp (Deduplication)
        context_parts = list(set(context_parts))
        context = " ".join(context_parts)
        
        # Gọi LLM sinh câu trả lời cuối cùng
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Dựa vào các kiến thức từ Knowledge Graph sau đây, hãy trả lời câu hỏi. Nếu thông tin không có trong Knowledge Graph, hãy trả lời 'Tôi không biết'."),
            ("human", "Knowledge Graph Context: {context}\n\nCâu hỏi: {question}")
        ])
        chain = prompt | self.llm
        try:
            res = chain.invoke({"context": context, "question": question})
            return res.content
        except Exception as e:
            return f"Lỗi sinh câu trả lời: {e}"
