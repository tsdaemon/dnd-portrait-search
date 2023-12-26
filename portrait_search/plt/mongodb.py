from motor.motor_asyncio import AsyncIOMotorClient

from portrait_search.plt.config import Config


_mongodb_client = None


def get_mongodb_client():
    global _mongodb_client

    if _mongodb_client is None:
        mongodb_uri = Config().mongodb_uri
        _mongodb_client = AsyncIOMotorClient(mongodb_uri)

    return _mongodb_client
