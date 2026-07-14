import redis.asyncio as aioredis
import json
from typing import Optional

class RedisStorage:
    def __init__(self, redis_url, password):
        self.pool = aioredis.ConnectionPool.from_url(redis_url, password=password)
    
    def get_redis_client(self):
        return aioredis.Redis(connection_pool=self.pool)
    
    async def close(self):
        await self.pool.disconnect()
    
    def _get_key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"
    
    async def get_history(self, session_id: str) -> list:
        redis_client = self.get_redis_client()
        key = self._get_key(session_id)
        history = await redis_client.lrange(key, 0, -1)
        return [json.loads(item) for item in history]
    
    async def append_message(self, session_id: str, role: str, content: str, ttl: Optional[int] = 3600):
        redis_client = self.get_redis_client()
        key = self._get_key(session_id)
        data = {"role": role, "content": content}
        async with redis_client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, json.dumps(data))
            pipe.expire(key, ttl)
            await pipe.execute()