from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.quick_silver_route import router as quick_silver_route
from routes.brain_nova_route import brain_nova_router
from routes.user_route import user_router
from routes.voice_route import voice_router

app = FastAPI(
    title="Vetox API",
    description="Chat with Vetox AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(quick_silver_route)
app.include_router(brain_nova_router)
app.include_router(user_router)
app.include_router(voice_router)

@app.get("/")
async def root():
    pass