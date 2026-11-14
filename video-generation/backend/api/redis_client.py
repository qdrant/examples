import os
import redis

redis_url = os.environ.get("REDIS_URL")
r = redis.Redis.from_url(redis_url, decode_responses=True)