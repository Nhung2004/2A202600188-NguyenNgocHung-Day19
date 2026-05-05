import os
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-openai-api-key-here":
    print("CẢNH BÁO: Chưa cấu hình OPENAI_API_KEY trong file .env")

# Cấu hình đường dẫn chung
DATA_PATH = "data/tech_corpus.txt"
GRAPH_PATH = "results/graph.html"
RESULTS_PATH = "results/comparison_report.csv"
