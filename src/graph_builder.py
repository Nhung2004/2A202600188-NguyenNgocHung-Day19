import os
import networkx as nx
from src.config import DATA_PATH, GRAPH_PATH
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

class Triple(BaseModel):
    subject: str = Field(description="Chủ thể (Subject)")
    relation: str = Field(description="Mối quan hệ (Relation/Predicate)")
    object: str = Field(description="Đối tượng (Object)")

class Triples(BaseModel):
    triples: List[Triple]

def extract_triples(text: str) -> list:
    """
    Sử dụng OpenAI LLM để trích xuất triples (Subject, Relation, Object).
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(Triples)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Bạn là một chuyên gia xây dựng Knowledge Graph. Hãy trích xuất tất cả các mối quan hệ từ văn bản được cung cấp và trả về dưới dạng danh sách các bộ ba (subject, relation, object). Hãy giữ các thực thể ngắn gọn, chính xác."),
        ("human", "{text}")
    ])
    chain = prompt | structured_llm
    
    # Chia nhỏ văn bản để tránh quá giới hạn context nếu text dài
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    all_triples = []
    for i, chunk in enumerate(chunks):
        print(f"Đang trích xuất triples từ chunk {i+1}/{len(chunks)}...")
        try:
            result = chain.invoke({"text": chunk})
            if result and result.triples:
                for t in result.triples:
                    all_triples.append((t.subject, t.relation, t.object))
        except Exception as e:
            print(f"Lỗi khi trích xuất chunk {i+1}: {e}")
            
    return all_triples

def build_graph(triples: list) -> nx.Graph:
    """
    Xây dựng NetworkX Graph từ danh sách triples.
    """
    G = nx.Graph()
    for subject, relation, obj in triples:
        G.add_edge(subject, obj, relation=relation)
    return G

def save_and_visualize_graph(G: nx.Graph, save_path: str = "results/graph.html"):
    """
    Lưu và trực quan hoá đồ thị dưới dạng Interactive HTML sử dụng Pyvis.
    """
    try:
        from pyvis.network import Network
        
        # Tạo đối tượng mạng Pyvis
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", directed=True)
        
        # Thêm các nodes và edges từ NetworkX graph vào Pyvis
        net.from_nx(G)
        
        # Hiển thị Edge Labels (Quan hệ)
        for edge in net.edges:
            relation = G.edges[edge['from'], edge['to']].get('relation', '')
            edge['label'] = relation
            edge['title'] = relation
        
        # Cấu hình tuỳ chỉnh giao diện để dễ nhìn hơn
        net.set_options("""
        var options = {
          "nodes": {
            "shape": "dot",
            "size": 20,
            "font": {
              "size": 16
            }
          },
          "edges": {
            "color": {
              "inherit": true
            },
            "smooth": false,
            "font": {
              "size": 12,
              "align": "middle"
            }
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
        """)
        
        net.save_graph(save_path)
        print(f"Interactive Graph UI saved to {save_path}")
    except ImportError:
        print("Vui lòng cài đặt pyvis: pip install pyvis")

def run_graph_building():
    """Hàm chính để chạy quá trình xử lý văn bản -> đồ thị"""
    print("Reading corpus...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    print("Extracting triples using OpenAI...")
    triples = extract_triples(text)
    
    print(f"Extracted {len(triples)} triples. Building Graph...")
    G = build_graph(triples)
    
    os.makedirs("results", exist_ok=True)
    save_and_visualize_graph(G, GRAPH_PATH)
    
    print("Done building graph!")
    return G

if __name__ == "__main__":
    run_graph_building()
