from src.graph_builder import run_graph_building
from src.flat_rag import FlatRAG
from src.graph_rag import GraphRAG
from src.evaluation import evaluate_systems
import os
import sys

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=== BƯỚC 1: XÂY DỰNG GRAPHRAG (NETWORKX) ===")
    G = run_graph_building()
    graph_rag = GraphRAG(G)
    
    print("\n=== BƯỚC 2: XÂY DỰNG FLAT RAG ===")
    flat_rag = FlatRAG()
    flat_rag.build_index()
    
    print("\n=== BƯỚC 3: ĐÁNH GIÁ VÀ SO SÁNH ===")
    queries = [
        "Ai là người sáng lập OpenAI?",
        "Mối quan hệ giữa Microsoft và OpenAI là gì?",
        "Kể tên một số đối thủ cạnh tranh của OpenAI trong lĩnh vực AI.",
        "Sản phẩm nào của Google tương tự như ChatGPT?",
        "DeepMind nổi tiếng với sản phẩm nào và CEO của họ là ai?",
        "Meta trước đây được gọi là gì?",
        "Giám đốc AI của Meta là ai?",
        "NVIDIA cung cấp công nghệ gì để huấn luyện mô hình AI?",
        "Dịch vụ đám mây lớn nhất thế giới là gì và ai sáng lập công ty đó?",
        "Apple sản xuất những thiết bị nào?",
        "Google đã mua lại công ty nào vào năm 2014?",
        "Ai là CEO của NVIDIA?",
        "OpenAI được thành lập vào năm nào?",
        "Trụ sở chính của OpenAI nằm ở đâu?",
        "Ai sáng lập Google?",
        "Mark Zuckerberg đã sáng lập công ty nào?",
        "Mô hình AI mã nguồn mở của Meta có tên là gì?",
        "Bill Gates và Paul Allen sáng lập công ty nào?",
        "Steve Wozniak và Ronald Wayne đồng sáng lập công ty nào?",
        "Hệ điều hành nổi tiếng nhất của Microsoft là gì?"
    ]
    
    os.makedirs("results", exist_ok=True)
    evaluate_systems(flat_rag, graph_rag, queries)
    print("\n=== HOÀN THÀNH ===")

if __name__ == "__main__":
    main()
