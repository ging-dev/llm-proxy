from contextlib import asynccontextmanager
from typing import AsyncIterator, TypedDict
from fastapi import FastAPI
import httpx
from .routers import duckduckgo

class State(TypedDict):
    http_client: httpx.AsyncClient

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    async with httpx.AsyncClient() as http_client:
        yield {'http_client': http_client}

app = FastAPI(title='Freedom LLM', description='Free AI for everyone', lifespan=lifespan)

app.include_router(duckduckgo.router)

@app.get('/')
async def root():
    return {'message': 'Hello, my name is Ging'}
