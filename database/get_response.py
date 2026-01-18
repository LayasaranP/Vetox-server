from database.config import chat_collection
from bson import ObjectId
from datetime import datetime

def get_user_chat_history(user_id: str):
    try:
        # Fetch sessions for the user, returning only _id and a preview
        sessions = chat_collection.find({"user_id": user_id}, {"_id": 1, "created_at": 1, "chats": {"$slice": 1}})
        history = []
        for s in sessions:
            title = "New Chat"
            if s.get("chats") and len(s["chats"]) > 0:
                title = s["chats"][0]["prompt"][:30] + "..."
            
            history.append({
                "chat_id": str(s["_id"]),
                "title": title,
                "created_at": s.get("created_at")
            })
        
        # Sort by created_at descending, handle missing date correctly
        history.sort(key=lambda x: x.get("created_at") or datetime.min, reverse=True)
        return history
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return []

def get_chat_session(chat_id: str):
    try:
        session = chat_collection.find_one({"_id": ObjectId(chat_id)})
        if session:
            # Convert ObjectId to string for JSON serialization
            session["_id"] = str(session["_id"])
            return session
        return None
    except Exception as e:
        print(f"Error fetching chat session: {e}")
        return None

def delete_chat_session(chat_id: str):
    try:
        result = chat_collection.delete_one({"_id": ObjectId(chat_id)})
        return result.deleted_count > 0
    except Exception as e:
        print(f"Error deleting chat session: {e}")
        return False
