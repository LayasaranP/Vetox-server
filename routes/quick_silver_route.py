from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse
import asyncio
from quicksilver.model import get_quicksilver_response
from database.save_response import save_chats_to_db

from database.get_response import get_user_chat_history, get_chat_session, delete_chat_session

router = APIRouter(
    prefix="/chat",
    tags=["quicksilver"]
)


@router.post("/quickSilver")
async def quicksilver_chat(request: Request):
    try:
        user_message = await request.json()
        message = user_message.get("message")
        user_id = user_message.get("user_id", "anonymous")
        chat_id = user_message.get("chat_id")  # Optional
        is_temporary = user_message.get("is_temporary", False)

        if not message or not message.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty.")

        chat_history = []
        if chat_id and chat_id != "temporary":
            session = get_chat_session(chat_id)
            if session:
                for chat in session.get("chats", []):
                    chat_history.append(f"User: {chat['prompt']}")
                    chat_history.append(f"Chatbot: {chat['response']}")

        async def stream_generator():
            full_response = ""
            try:
                for chunk in get_quicksilver_response(message, chat_history):
                    full_response += chunk
                    yield chunk
                    await asyncio.sleep(0.01)

                if not is_temporary:
                    new_chat_id = save_chats_to_db(user_id, message, full_response, chat_id)
                    if new_chat_id:
                        yield f"\n\n[CHAT_ID]:{new_chat_id}"
            except Exception as e:
                yield f"\n\n[ERROR]:Internal server error: {str(e)}"

        return StreamingResponse(stream_generator(), media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error.{e}")


@router.get("/history/{user_id}")
async def fetch_history(user_id: str):
    history = get_user_chat_history(user_id)
    return history


@router.get("/session/{chat_id}")
async def fetch_session(chat_id: str):
    session = get_chat_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/session/{chat_id}")
async def delete_session(chat_id: str):
    success = delete_chat_session(chat_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or already deleted")
    return {"message": "Session deleted successfully"}
