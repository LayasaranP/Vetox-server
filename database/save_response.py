from database.config import chat_collection
from bson import ObjectId


from datetime import datetime

def save_chats_to_db(user_id: str, message: str, response: str, chat_id: str = None):
    try:
        chat_document = {
            "prompt": message,
            "response": response,
            "timestamp": datetime.now()
        }

        if chat_id:
            # Update existing chat session
            chat_collection.update_one(
                {"_id": ObjectId(chat_id), "user_id": user_id},
                {"$push": {"chats": chat_document}}
            )
            return chat_id
        else:
            # Create new chat session
            new_session = {
                "user_id": user_id,
                "created_at": datetime.now(),
                "chats": [chat_document]
            }
            result = chat_collection.insert_one(new_session)
            return str(result.inserted_id)

    except Exception as e:
        print(f"An error occurred while saving the chat: {e}")
        return None
