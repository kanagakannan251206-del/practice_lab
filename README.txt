23AD22L - Agentic AI Tools and Frameworks - Model Exam
Each qNN_*.py file is ONE complete program: Flask backend + interactive HTML frontend + agent + vector memory.

1. python -m venv venv && venv\Scripts\activate   (Windows)   |   source venv/bin/activate  (Linux/Mac)
2. pip install -r requirements.txt
3. Copy .env.example to .env and paste your OpenAI API key
4. python q01_shopping_cost.py   -> open http://127.0.0.1:5001   (port = 5000 + question number)
Memory persists in faiss_memory_qNN/ or chroma_memory_qNN/ folders (use the Clear button on the page to reset).
