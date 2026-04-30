import os
import ast
import json
import random
from typing import Any, List, Tuple

import tiktoken
from dotenv import load_dotenv
from langchain.docstore.document import Document

from env_manager import llm_class, vectorstore_class
from logger import logger
from redis_util import read_messages_from_redis, store_messages_in_redis
from utils import convert_chat_messages, get_from_env_or_config

load_dotenv()

temperature = float(get_from_env_or_config("llm", "temperature"))
chatClient  = llm_class.get_client(temperature=temperature)
max_messages = int(get_from_env_or_config("llm", "max_messages")) # Maximum number of messages to include in conversation history
llm_type = os.getenv("LLM_TYPE")

def querying_with_langchain_gpt3(index_id, query, context):
    intent_response, response_type = check_bot_intent(query, context)
    if intent_response:
        return intent_response, None, 200, 0, 0, 0, response_type
    
    try:
        system_rules = ""
        activity_prompt_config = get_from_env_or_config("llm", "activity_prompt", None)
        if activity_prompt_config:
            activity_prompt_dict = ast.literal_eval(activity_prompt_config)
            system_rules = activity_prompt_dict.get(context)

        top_docs_to_fetch = get_from_env_or_config("database", "top_docs_to_fetch", None)
        documents = vectorstore_class.similarity_search_with_score(query, index_id, k=20)
        logger.debug(f"Marqo documents : {str(documents)} \n")
        min_score = get_from_env_or_config("database", "docs_min_score", None)
        filtered_document = get_score_filtered_documents(documents, float(min_score))
        filtered_document = filtered_document[:int(top_docs_to_fetch)]
        contexts = get_formatted_documents(filtered_document)
        if not documents or not contexts or not filtered_document:
            return "शायद मैं सही से नहीं समझ पाई 🙂 क्या आप इसे एक और तरीके से पूछ सकते हैं?", None, 200, 0, 0, 0, "fallback"

        system_rules = system_rules.format(contexts=contexts)
        answer = call_chat_model(
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": query}
            ]
        )
        logger.info({"label": "llm_response", "response": answer.content})
        
        response = answer.content

        if llm_type == "bedrock":
            token_usage = answer.response_metadata["usage"]
            input_tokens = token_usage["prompt_tokens"]
            output_tokens = token_usage["completion_tokens"]
            total_tokens = token_usage["total_tokens"]
        elif llm_type == "openai":
            token_usage = answer.response_metadata["token_usage"]
            input_tokens = token_usage["prompt_tokens"]
            output_tokens = token_usage["completion_tokens"]
            total_tokens = token_usage["total_tokens"]
        else:
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

        
        return response.strip(";"), None, 200, input_tokens, output_tokens, total_tokens, "ai_generated"
    except Exception as e:
        error_message = str(e.__context__) + " and " + e.__str__()
        status_code = 500

    return "", error_message, status_code, 0, 0, 0, "failed"

def conversation_retrieval_chain(index_id, query, session_id, context):
    # intent_response, response_type = check_bot_intent(query, context)
    # if intent_response:
    #     print(f'>>> intent_response: {intent_response}')
    #     return intent_response, None, 200, 0, 0, 0, response_type
    
    try:
        system_rules = ""
        activity_prompt_config = get_from_env_or_config("llm", "activity_prompt", None)
        activity_prompt_dict = ast.literal_eval(activity_prompt_config)
        system_rules = activity_prompt_dict.get(context)
        previous_messages  = read_messages_from_redis(session_id)
        formatted_messages = format_previous_messages(previous_messages)
        user_message = {"role":"user","content": query}
        intent_system_prompt = get_chat_intent_prompt()
        intent_payload = create_payload_by_message_count(user_message, intent_system_prompt, messages=formatted_messages, max_messages=max_messages)
        search_intent = get_intent_query(intent_payload)

        intent_response, response_type = check_bot_intent(search_intent, context)
        if intent_response:
            print(f'>>> intent_response: {intent_response}')
            return intent_response, None, 200, 0, 0, 0, response_type

        documents = vectorstore_class.similarity_search_with_score(search_intent, index_id, k=20)
        logger.debug(f"Marqo documents : {str(documents)}")
        min_score = get_from_env_or_config("database", "docs_min_score", None)
        filtered_document = get_score_filtered_documents(documents, float(min_score))
        top_docs_to_fetch = get_from_env_or_config("database", "top_docs_to_fetch", None)
        filtered_document = filtered_document[:int(top_docs_to_fetch)]
        contexts = get_formatted_documents(filtered_document)
        if not documents or not contexts or not filtered_document:
            return "शायद मैं सही से नहीं समझ पाई 🙂 क्या आप इसे एक और तरीके से पूछ सकते हैं?", None, 200, 0, 0, 0, "fallback"

        system_rules = system_rules.format(contexts=contexts)
        system_rules = {"role": "system", "content": system_rules}
        message_payload  = create_payload_by_message_count(user_message,system_rules,formatted_messages,max_messages=max_messages)
        answer = call_chat_model(message_payload)
        logger.info({"label": "llm_response", "response": answer.content})
        
        response = answer.content


        if llm_type == "bedrock":
            token_usage = answer.response_metadata["usage"]
            input_tokens = token_usage["prompt_tokens"]
            output_tokens = token_usage["completion_tokens"]
            total_tokens = token_usage["total_tokens"]
        elif llm_type == "openai":
            token_usage = answer.response_metadata["token_usage"]
            input_tokens = token_usage["prompt_tokens"]
            output_tokens = token_usage["completion_tokens"]
            total_tokens = token_usage["total_tokens"]
        else:
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

        assistant_message = format_assistant_message(response.strip(";"))
        messages = read_messages_from_redis(session_id)
        messages.extend([user_message,assistant_message])
        final_response = response.strip(";").replace("**","*")
        store_messages_in_redis(session_id, messages)
        print(f"Response: >>>>>>>>>>>> {final_response}\n\n")
        
        formatting_prompt_config = get_from_env_or_config("llm", "formatting_prompt", None)
        if formatting_prompt_config:
            formatting_prompt_dict = ast.literal_eval(formatting_prompt_config)
            formatting_rules = formatting_prompt_dict.get(context)
            if formatting_rules:
                
                formatting_prompt = formatting_rules.format(
                    query=user_message,
                    response=final_response
                )
                formatting_response = llm_class.get_client(temperature=0.2).invoke(formatting_prompt)
                print(f"Formatted Response: >>>>>>>>>>> {formatting_response.content}")
                final_response = formatting_response.content

        return final_response, None, 200, input_tokens, output_tokens, total_tokens, "ai_generated"
    except Exception as e:
        error_message = str(e.__context__) + " and " + e.__str__()
        status_code = 500

    return "", error_message, status_code, 0, 0, 0, "failed"

def call_chat_model(messages: List[dict]) -> str:
    converted_messsages = convert_chat_messages(messages)
    response = chatClient.invoke(input=converted_messsages)
    return response

def format_assistant_message(a):
    """Formats the assistant message
    Args:
        a (str, optional): assistant's reply.
    Returns:
        dict: formatted assistant message
    """
    return {'role': 'assistant', 'content': a.strip()}

def get_chat_intent_prompt():
    intent_prompt = get_from_env_or_config("llm", "chat_intent_prompt")
    return {'role': "system", 'content': intent_prompt }

def get_intent_query(messages=[]):
    """Reformulates the conversation into a focused search query.

    Args:
        messages (list): list of messages representing the conversation history
    Returns:
        str: clean reformulated search query
    """
    json_format_message = {
        "role": "user",
        "content": 'Respond only with a JSON object in this exact format: {"thoughts": "<brief reasoning>", "query": "<reformulated search query>"}'
    }
    clientIntent = llm_class.get_client(temperature=0.1)
    converted_messsages = convert_chat_messages(messages + [json_format_message])
    response = clientIntent.invoke(input=converted_messsages)

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:-1])
        result = json.loads(content)
        logger.debug(f"Intent query thoughts: {result.get('thoughts', '')}")
        return result["query"]
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"Failed to parse intent query JSON, using raw response: {response.content}")
        return response.content



def count_tokens_str(doc, model="gpt-4"):
    """Count tokens in a string.

    Args:
        doc (str): String to count tokens for.
    Returns:
        int: number of tokens in the string

    """
    encoder = tiktoken.encoding_for_model(model)  # BPE encoder # type: ignore
    return len(encoder.encode(doc, disallowed_special=()))

def count_tokens(messages):
    """
    Counts tokens in a list of messages.
    Source: https://platform.openai.com/docs/guides/chat/introduction

    Args:
        messages (list): list of messages to count tokens for
    Returns:
        int: number of tokens in the list of messages
    """
    num_tokens = 0
    for message in messages:
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4
        for key, value in message.items():
            num_tokens += count_tokens_str(value)
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def create_payload_by_message_count(user_message, system_message, messages=[], max_messages=4):  # IMPORTANT
    """Get the message history for the conversation, limited by message count.

    Args:
        user_message (str): User message to add to the history.
        system_message (str): System message to add to the history.
        messages (list, optional): List of previous messages. Defaults to [].
        max_messages (int, optional): Maximum number of messages to include. Defaults to 4.

    Returns:
        list: Message history
    """
    message_history = [system_message]
    total_count =  max_messages * 2
    message_history.extend(messages[-total_count:])
    message_history.append(user_message)
    return message_history

def create_message_payload(user_message, system_message, messages=[], max_tokens=3000):  # IMPORTANT
    """Get the message history for the conversation.
    # NOTE: Include user message {role=user,content=user_q} in the message history

    Args:
        message_payload (dict, optional): Formatted RAG prompt to add (temporarily) to the conversation. Defaults to {}.
        max_tokens (int, optional): Maximum number of tokens to limit the message history to. Defaults to 3000.

    Returns:
        list: message history

    NOTE: 
        - System-Prompt is always added to the beginning of the message history
        - message_payload is added to the end of the message history (if provided)

    """
    message_history = []
    total_tokens = 0
    system_token_count = count_tokens([system_message])
    max_tokens -= system_token_count  # subtract the system prompt tokens
    if len(user_message) > 0:
        messages = messages + [user_message]
    else:
        messages = messages

    for message in reversed(messages):
        message_tokens = count_tokens([message])
        if total_tokens + message_tokens <= max_tokens:
            total_tokens += message_tokens
            # This inserts the message at the beginning of the list
            message_history.insert(0, message)
        else:
            break
    message_history.insert(0, system_message)
    return message_history

def format_previous_messages(messages):
    """
    Format previous messages for display
    """
    formatted_messages = []
    for message in messages:
        if message['role'] == 'user':
            formatted_messages.append({"role":"user", "content":f"Question: {message['content']}"})
        elif message['role'] == 'assistant':
            formatted_messages.append({"role":"assistant", "content":message['content']})
    return formatted_messages


def check_bot_intent(query: str, context: str):

    enable_bot_intent = get_from_env_or_config("llm", "enable_bot_intent", None)
    logger.debug(f"enable_bot_intent: {enable_bot_intent}")
    if enable_bot_intent.lower() == "false":
        return None, None

    intent_prompt = get_from_env_or_config("llm", "intent_prompt")
    if context == "swadhaar_agent_dev":
        intent_prompt = get_from_env_or_config("llm", "intent_prompt_dev")

    intent_response = call_chat_model(
        messages=[{"role": "system", "content": intent_prompt}, {"role": "user", "content": query}]
    )
    logger.info({"label": "intent_response", "intent_response": intent_response})
    
    # Extract the content from AIMessage object and process it to get just the intent
    intent_content = intent_response.content.lower()
    print(f'intent_content: {intent_content}')
    # Extract just the intent by looking for keywords
    intent_type = "out_of_scope"  # default intent
    if "out_of_scope" in intent_content:
        intent_type = "out_of_scope"
    elif "bot_query" in intent_content:
        intent_type = "bot_query"
    elif "finance_query" in intent_content:
        intent_type = "finance_query"

    if intent_type == "bot_query":
        bot_prompt_config = get_from_env_or_config("llm", "bot_prompt", "")
        bot_prompt_dict = ast.literal_eval(bot_prompt_config)
        system_rules = bot_prompt_dict.get(context)
        response = call_chat_model(
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": query}
            ]
        )
        logger.info({"label": "llm_bot_response", "bot_response": response})
        return response.content, "ai_generated"
    elif intent_type == "out_of_scope":
        out_of_scope_responses = [
            "💡 हम्म… मैं पैसे की बचत, फ़्रॉड से बचाव, और डिजिटल सेवाओं के सुरक्षित इस्तेमाल जैसे विषयों में मदद कर सकती हूँ 🔐। क्या आप इन विषयों के बारे में और जानना चाहेंगी? 😊",
            "❗मैं इस विषय में मदद नहीं कर सकती, लेकिन डिजिटल और पैसों से जुड़े ज़रूरी विषयों में मदद कर सकती हूँ। क्या आप इन विषयों के बारे में और जानना चाहेंगी? 😊"
        ]
        return random.choice(out_of_scope_responses), "out_of_scope"
    else:
        return None, None
            
 

def get_score_filtered_documents(documents: List[Tuple[Document, Any]], min_score=0.0):
    return [(document, search_score) for document, search_score in documents if search_score > min_score]


def get_formatted_documents(documents: List[Tuple[Document, Any]]):
    sources = ""
    for document, _ in documents:
        metadata_str = "; ".join(f"{key}: {value}" for key, value in document.metadata.items())
        sources += f"""\n> {document.page_content} \n Metadata: {metadata_str}\n\n"""
    return sources


def generate_source_format(documents: List[Tuple[Document, Any]]) -> str:
    """Generates an answer format based on the given data.

    Args:
    data: A list of tuples, where each tuple contains a Document object and a
        score.

    Returns:
    A string containing the formatted answer, listing the source documents
    and their corresponding pages.
    """
    try:
        sources = {}
        for doc, _ in documents:
            file_name = doc.metadata['file_name']
            page_label = doc.metadata['page_label']
            sources.setdefault(file_name, []).append(page_label)

        answer_format = "\nSources:\n"
        counter = 1
        for file_name, pages in sources.items():
            answer_format += f"{counter}. {file_name} - (Pages: {', '.join(pages)})\n"
            counter += 1
        return answer_format
    except Exception as e:
        error_message = "Error while preparing source markdown"
        logger.error(f"{error_message}: {e}", exc_info=True)
        return ""

def concatenate_elements(arr):
    # Concatenate elements from index 1 to n
    separator = ': '
    result = separator.join(arr[1:])