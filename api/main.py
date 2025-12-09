# api/main.py
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from agent.agent import HotelRAGAgent

app = FastAPI(title="酒店评论分析智能体（LangChain + Ollama）", version="1.0")

# 添加 CORS 支持，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 Agent（启动时加载一次）
agent = HotelRAGAgent()

@app.get("/")
async def read_root():
    """返回前端页面"""
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return {"message": "前端页面未找到，请访问 /docs 查看 API 文档"}

@app.get("/ask")
async def ask(query: str):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    answer = agent.ask(query.strip())
    return {"question": query, "answer": answer}
from fastapi import Request

# @app.post("/ask")
# async def ask_post(request: Request):
#     """
#     支持 POST 请求，接收 JSON 格式：{"query": "你的问题"}
#     """
#     data = await request.json()
#     query = data.get("query", "")
#     if not query or not query.strip():
#         raise HTTPException(status_code=400, detail="问题不能为空")
#     answer = agent.ask(query.strip())
#     return {"question": query, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    print("启动酒店评论 Agent 服务...")
    print("访问 http://localhost:8000 查看前端页面")
    print("访问 http://localhost:8000/docs 查看交互式 API 文档")
    uvicorn.run(app, host="0.0.0.0", port=8000)
