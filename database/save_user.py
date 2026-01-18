from database.config import users_collection


def save_user_to_db(user_id: str, name: str, email: str):
    try:
        user_document = {
            "user_id": user_id,
            "name": name,
            "email": email
        }

        users_collection.insert_one(user_document)

        return {
            "message": "User saved successfully.",
        }

    except Exception as e:
        return f"An error occurred while saving the user: {e}"
