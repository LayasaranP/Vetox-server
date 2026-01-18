from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://layasaranp_db_user:so3nBAzrDOEro9WE@vetoxai.sf7b8rt.mongodb.net/?appName=VetoxAI"

client = MongoClient(uri, server_api=ServerApi('1'))

db = client['vetoxAI_db']

users_collection = db["users"]

chat_collection = db["user_chat"]

try:
    client.admin.command('ping')
except Exception as e:
    print(e)