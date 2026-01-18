from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse
import asyncio
from brain_nova.vetox import get_response_brain_nova
from database.save_response import save_chats_to_db

brain_nova_router = APIRouter(
    prefix="/chat",
    tags=["brain_nova"]
)


@brain_nova_router.post("/brainNova")
async def brainNova_chat(request: Request):
    try:
        user_message = await request.json()
        message = user_message.get("message")
        user_id = user_message.get("user_id", "anonymous")
        chat_id = user_message.get("chat_id")
        is_temporary = user_message.get("is_temporary", False)
        is_temporary = user_message.get("is_temporary", False)

        if not message or not message.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty.")

        async def stream_generator():
            full_response = ""
            try:
                for chunk in get_response_brain_nova(message, chat_id):
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
