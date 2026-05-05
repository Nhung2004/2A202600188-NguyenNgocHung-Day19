import pandas as pd
import time
from src.flat_rag import FlatRAG
from src.graph_rag import GraphRAG
from src.config import RESULTS_PATH

def evaluate_systems(flat_rag: FlatRAG, graph_rag: GraphRAG, queries: list):
    results = []
    
    for i, q in enumerate(queries):
        print(f"Evaluating query {i+1}: {q}")
        
        # Flat RAG eval
        start = time.time()
        ans_flat = flat_rag.query(q)
        time_flat = time.time() - start
        
        # Graph RAG eval
        start = time.time()
        ans_graph = graph_rag.query(q)
        time_graph = time.time() - start
        
        results.append({
            "Query": q,
            "FlatRAG_Answer": ans_flat,
            "FlatRAG_Time": round(time_flat, 4),
            "GraphRAG_Answer": ans_graph,
            "GraphRAG_Time": round(time_graph, 4)
        })
        
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_PATH, index=False, encoding="utf-8")
    print(f"Saved evaluation results to {RESULTS_PATH}")
    return df
