import httpx

from fastapi import APIRouter, HTTPException, Header, Request, Response
from httpx_sse import EventSource
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask
from app.internal.constants import DEFAULT_USER_AGENT, DUCKDUCKGO_CHAT_ENDPOINT, DUCKDUCKGO_STATUS_ENDPOINT
from typing import Annotated, List, Literal, cast
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
    session_id = x_session_id or (await http_client.get(DUCKDUCKGO_STATUS_ENDPOINT, headers={
        'x-vqd-accept': '1',
        'user-agent': DEFAULT_USER_AGENT,
    })).headers.get('x-vqd-4')

    req = http_client.build_request('POST', DUCKDUCKGO_CHAT_ENDPOINT,
                                    json=input.model_dump(exclude={'stream'}),
                                    headers={
                                        'x-vqd-4': session_id,
                                        'user-agent': DEFAULT_USER_AGENT
                                    })
    resp = await http_client.send(req, stream=input.stream)

    if resp.status_code != 200:
        raise HTTPException(status_code=400)

    async def agenerator():
        async for event in EventSource(resp).aiter_sse():
            if event.data == DONE:
                return

            content = cast(dict[str, str], event.json()).get('message')
            if content:
                yield content

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

    response.headers['x-session-id'] = resp.headers.get('x-vqd-4')

    if input.stream:
        return EventSourceResponse(event_generator(), background=BackgroundTask(resp.aclose), headers=response.headers)

    content = ''
    async for chunk in agenerator():
        content += chunk

    return CompletionsResult(choices=[Choice(message=Message(role='assistant', content=content))])
