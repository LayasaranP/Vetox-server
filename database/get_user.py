from database.config import users_collection


def get_user_from_db(user_id: str):
    users = users_collection.find_one({"user_id": user_id})

    if users is None:
        return "No user found"

    return {
        "name": users["name"],
        "email": users["email"]
    }