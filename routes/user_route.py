from fastapi import APIRouter, Request
from database.save_user import save_user_to_db
from database.get_user import get_user_from_db

user_router = APIRouter(
    prefix="/user",
    tags=["user"]
)


@user_router.post("/register")
async def register_user(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id")
        name = data.get("name")
        email = data.get("email")

        if not email or not name:
            return {"error": "Name and email and id are required"}

        existing_user = get_user_from_db(user_id)

        if existing_user != "No user found":
            return get_user_from_db(user_id)

        result = save_user_to_db(user_id, name, email)
        return result

    except Exception as e:
        return {"error": str(e)}


@user_router.get("/{user_id}")
def fetch_user(user_id: str):
    try:
        user = get_user_from_db(user_id)
        return user
    except Exception as e:
        print(e)
