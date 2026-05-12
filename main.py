import os
import json
import re
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware

from utils import is_url, is_base64, prepare_redis_key, get_from_env_or_config
from env_manager import storage_class as storage
from io_processing import *
from query_with_langchain import *
from telemetry_middleware import TelemetryMiddleware
from constants import _VIDEO_URLS


app = FastAPI(
    title="Sakhi API Service",
    #   docs_url=None,  # Swagger UI: disable it by setting docs_url=None
    redoc_url=None,  # ReDoc : disable it by setting docs_url=None
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    description='',
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info('Invoking startup_event')
    load_dotenv()
    logger.info('startup_event : Engine created')


@app.on_event("shutdown")
async def shutdown_event():
    logger.info('Invoking shutdown_event')
    logger.info('shutdown_event : Engine closed')

Context = Enum("Context", {type: type for type in get_from_env_or_config('request', 'supported_context', None).split(',')})
DropdownOutputFormat = Enum("DropdownOutputFormat", {type: type for type in get_from_env_or_config('request', 'supported_response_format', None).split(',')})
DropDownInputLanguage = Enum("DropDownInputLanguage", {type: type for type in get_from_env_or_config('request', 'supported_lang_codes', None).split(',')})
ResponseType = Enum("ResponseType", {type: type for type in get_from_env_or_config('request', 'supported_response_type', None).split(',')})

class OutputResponse(BaseModel):
    text: str
    audio: str = None
    language: DropDownInputLanguage # type: ignore
    format: DropdownOutputFormat # type: ignore
    response_type: ResponseType # type: ignore
    number_of_input_tokens: int
    number_of_output_tokens: int
    number_of_total_tokens: int

class ResponseForQuery(BaseModel):
    output: OutputResponse


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""
    status: str = "OK"


class QueryInputModel(BaseModel):
    language: DropDownInputLanguage # type: ignore
    text: str = ""
    audio: str = ""
    context: Context # type: ignore


class QueryOuputModel(BaseModel):
    format: DropdownOutputFormat # type: ignore

class QueryModel(BaseModel):
    input: QueryInputModel
    output: QueryOuputModel

# Telemetry API logs middleware
app.add_middleware(TelemetryMiddleware)


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to Sakhi API Service"}

def is_valid_video(url: str) -> bool:
    """Check if the url (ignoring query params) matches any entry in _VIDEO_URLS"""
    if not url:
        return False
    return any(url.split('?')[0] == video_url.split('?')[0] for video_url in _VIDEO_URLS)


@app.get(
    "/health",
    tags=["Health Check"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
    include_in_schema=True
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


@app.post("/v1/query", tags=["Q&A over Document Store"], include_in_schema=True)
async def query(request: QueryModel, x_request_id: str = Header(None, alias="X-Request-ID")) -> ResponseForQuery:
    load_dotenv()
    indices = json.loads(get_from_env_or_config('database', 'indices', None))
    language = request.input.language.name
    context = request.input.context.name
    output_format = request.output.format.name
    index_id = indices.get(context.lower())
    audio_url = request.input.audio
    query_text = request.input.text
    is_audio = False
    text = None
    regional_answer = None
    audio_output_url = None
    logger.info({"label": "query", "query_text": query_text, "index_id": index_id, "context": context, "input_language": language, "output_format": output_format, "audio_url": audio_url})
    if not query_text and not audio_url:
        raise HTTPException(status_code=422, detail="Either 'text' or 'audio' should be present!")

    if query_text:
        text, error_message = process_incoming_text(query_text, language)
        if output_format == "audio":
            is_audio = True
    else:
        if not is_url(audio_url) and not is_base64(audio_url):
            logger.error({"index_id": index_id, "query": query_text, "input_language": language, "output_format": output_format, "audio_url": audio_url, "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY, "error_message": "Invalid audio input!"})
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid audio input!")
        query_text, text, error_message = process_incoming_voice(audio_url, language)
        is_audio = True
    
    if text is not None:
        answer, error_message, status_code, input_tokens, output_tokens, total_tokens, response_type = querying_with_langchain_gpt3(index_id, text, context)
        # Find and extract the full URL
        video_url = None
        if 'https://' in answer:
            answer, video_url = extract_video_url_and_clean(answer)
        if len(answer) != 0:
            regional_answer, error_message = process_outgoing_text(answer, language)
            logger.info({"regional_answer": regional_answer})
            if regional_answer is not None:
                if is_audio:
                    mp3_data_buffer, mp3_filename, error_message = process_outgoing_voice(regional_answer, language)
                    if mp3_data_buffer is not None:
                        upload_success = storage.upload_to_storage(object_name=mp3_filename, file_content=mp3_data_buffer)
                        if upload_success:
                            audio_output_url, error_message = storage.generate_public_url(mp3_filename)
                            logger.debug(f"Audio Ouput URL ===> {audio_output_url}")
                        else:
                            status_code = 503
                    else:
                        status_code = 503
                else:
                    audio_output_url = ""
            else:
                status_code = 503
    else:
        status_code = 503

    if status_code != 200:
        logger.error({"index_id": index_id, "query": query_text, "input_language": language, "output_format": output_format, "audio_url": audio_url, "status_code": status_code, "error_message": error_message})
        raise HTTPException(status_code=status_code, detail=error_message)
    

    if video_url:
        regional_answer = f"यह वीडियो सुझाव के तौर पर साझा किया गया है। \n{video_url} \n{regional_answer}"
    response = ResponseForQuery(output=OutputResponse(text=regional_answer, audio=audio_output_url, language=language, format=output_format, response_type=response_type, number_of_input_tokens=input_tokens, number_of_output_tokens=output_tokens, number_of_total_tokens=total_tokens))
    return response

@app.post("/v1/chat", tags=["Conversation chat over Document Store"], include_in_schema=True)
async def chat(request: QueryModel, x_request_id: str = Header(None, alias="X-Request-ID"),
                x_source: str = Header(None, alias="x-source"),
                x_consumer_id: str = Header(None, alias="x-consumer-id")) -> ResponseForQuery:
    load_dotenv()
    indices = json.loads(get_from_env_or_config('database', 'indices', None))
    language = request.input.language.name
    context = request.input.context.name
    output_format = request.output.format.name
    index_id = indices.get(context.lower())
    audio_url = request.input.audio
    query_text = request.input.text
    is_audio = False
    text = None
    regional_answer = None
    audio_output_url = None
    logger.info({"label": "query", "query_text": query_text, "index_id": index_id, "context": context, "input_language": language, "output_format": output_format, "audio_url": audio_url})
    redis_session_id  = prepare_redis_key(x_source, x_consumer_id, context)
    if not query_text and not audio_url:
        raise HTTPException(status_code=422, detail="Either 'text' or 'audio' should be present!")

    if query_text:
        text, error_message = process_incoming_text(query_text, language)
        if output_format == "audio":
            is_audio = True
    else:
        if not is_url(audio_url) and not is_base64(audio_url):
            logger.error({"index_id": index_id, "query": query_text, "input_language": language, "output_format": output_format, "audio_url": audio_url, "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY, "error_message": "Invalid audio input!"})
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid audio input!")
        query_text, text, error_message = process_incoming_voice(audio_url, language)
        is_audio = True
    
    if text is not None:
        answer, error_message, status_code, input_tokens, output_tokens, total_tokens, response_type = conversation_retrieval_chain(index_id, text, redis_session_id, context)
        # Find and extract the full URL
        video_url = None
        if 'https://' in answer:
            answer, video_url = extract_video_url_and_clean(answer)
        if len(answer) != 0:
            regional_answer, error_message = process_outgoing_text(answer, language)
            logger.info({"regional_answer": regional_answer})
            if regional_answer is not None:
                if is_audio:
                    prefix = f"{x_consumer_id}-{query_text}"
                    mp3_data_buffer, mp3_filename, error_message = process_outgoing_voice(regional_answer, language, prefix)
                    if mp3_data_buffer is not None:
                        upload_success = storage.upload_to_storage(object_name=mp3_filename, file_content=mp3_data_buffer)
                        if upload_success:
                            audio_output_url, error_message = storage.generate_public_url(mp3_filename)
                            logger.debug(f"Audio Ouput URL ===> {audio_output_url}")
                        else:
                            status_code = 503
                    else:
                        status_code = 503
                else:
                    audio_output_url = ""
            else:
                status_code = 503
    else:
        status_code = 503

    if status_code != 200:
        logger.error({"index_id": index_id, "query": query_text, "input_language": language, "output_format": output_format, "audio_url": audio_url, "status_code": status_code, "error_message": error_message})
        raise HTTPException(status_code=status_code, detail=error_message)

    if video_url and is_valid_video(video_url):
        regional_answer = f"यह वीडियो सुझाव के तौर पर साझा किया गया है। \n{video_url} \n{regional_answer}"  # Add video URL at the beginning
    response = ResponseForQuery(output=OutputResponse(text=regional_answer, audio=audio_output_url, language=language, format=output_format, response_type=response_type, number_of_input_tokens=input_tokens, number_of_output_tokens=output_tokens, number_of_total_tokens=total_tokens))
    return response

def extract_video_url_and_clean(answer: str):
    if not answer:
        return answer, None

    # find first occurrence of http (the split point)
    http_index = answer.find('http')
    if http_index == -1:
        return answer.strip(), None

    # extract URL by taking the first whitespace-delimited token starting at http
    tail = answer[http_index:]
    video_url = tail.split()[0]  # do NOT strip other punctuation here (per your request)

    # everything before the URL
    before = answer[:http_index]

    # find last '.' or '?' in the text before the URL
    last_punct_index = max(before.rfind('.'), before.rfind('?'), before.rfind('।'), before.rfind('!'))

    if last_punct_index != -1:
        # keep up to and including that punctuation
        cleaned_text = before[:last_punct_index + 1].strip()
    else:
        # no '.' or '?' found before the URL: keep all text before URL
        cleaned_text = before.strip()

    return cleaned_text, video_url

