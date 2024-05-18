import httpx

from fastapi import APIRouter, HTTPException, Header, Request, Response
from httpx_sse import EventSource
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask
from app.internal.constants import DUCKDUCKGO_CHAT_ENDPOINT, DUCKDUCKGO_STATUS_ENDPOINT, SESSION_KEY
from typing import Annotated, AsyncIterator, List, Literal
from pydantic import BaseModel

DONE = '[DONE]'


class Message(BaseModel):
    role: Literal['assistant', 'user']
    content: str


class Chat(BaseModel):
    model: Literal[
        'gpt-3.5-turbo-0125',
        'claude-3-haiku-20240307'
    ] = 'gpt-3.5-turbo-0125'
    messages: list[Message]
    stream: bool = False

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'model': 'claude-3-haiku-20240307',
                    'messages': [{
                        'role': 'user',
                        'content': 'Hello',
                    }]
                }
            ]
        }
    }


class Choice(BaseModel):
    message: Message


class CompletionsResult(BaseModel):
    choices: List[Choice]


router = APIRouter()


@router.post('/ddg/chat/completions',
             response_model=CompletionsResult,
             responses={
                 200: {
                     'content': {'text/event-stream': {}},
                     'description': 'Return the JSON completions result or an event stream.',
                 }
             })
async def chat(input: Chat, request: Request, response: Response, x_session_id: Annotated[str | None, Header()] = None):
    http_client: httpx.AsyncClient = request.state.http_client
    session_id = x_session_id or (await http_client.get(DUCKDUCKGO_STATUS_ENDPOINT, headers={'x-vqd-accept': '1'})).headers.get(SESSION_KEY)

    req = http_client.build_request('POST', DUCKDUCKGO_CHAT_ENDPOINT,
                                    json=input.model_dump(exclude={'stream'}),
                                    headers={SESSION_KEY: session_id})
    resp = await http_client.send(req, stream=True)

    if resp.status_code != 200:
        raise HTTPException(status_code=400)

    async def agenerator() -> AsyncIterator[str]:
        async for event in EventSource(resp).aiter_sse():
            if event.data == DONE:
                return

            if 'message' in (decoded := event.json()):
                yield decoded['message']

    async def event_generator():
        async for chunk in agenerator():
            yield {
                'data': {
                    'choices': [{
                        'delta': chunk
                    }]
                }
            }

        yield DONE

    response.headers['x-session-id'] = resp.headers.get(SESSION_KEY)

    if input.stream:
        return EventSourceResponse(event_generator(), background=BackgroundTask(resp.aclose), headers=response.headers)

    content = ''
    async for chunk in agenerator():
        content += chunk

    return CompletionsResult(choices=[Choice(message=Message(role='assistant', content=content))])
