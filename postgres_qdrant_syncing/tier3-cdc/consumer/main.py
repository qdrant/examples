"""
Tier 3: Kafka/Redpanda Consumer → Qdrant Sync

Reads Debezium change events from Redpanda and applies them to Qdrant.

Event format (after Debezium ExtractNewRecordState transform):
  - CREATE / UPDATE: all product fields present, __op = 'c' or 'u'
  - DELETE: __op = 'd', __deleted = 'true', article_id present

Run standalone:
    PYTHONPATH=../.. python consumer/main.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from confluent_kafka import Consumer, KafkaError, KafkaException

from shared.qdrant_helpers import (
    delete_product as _async_delete,
    init_collection as _async_init_collection,
    upsert_product as _async_upsert,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("consumer")

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "pgserver.public.products")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "qdrant-sync")


# Debezium metadata fields injected by ExtractNewRecordState — strip before upsert
_DEBEZIUM_META = frozenset(
    ["__op", "__deleted", "__source_ts_ms", "__ts_ms", "__table", "__lsn"]
)


async def _handle_event(event: dict) -> None:
    op = event.get("__op")
    article_id = event.get("article_id")

    if not article_id:
        logger.warning("Event missing article_id, skipping: %s", event)
        return

    if op in ("c", "u"):
        logger.info("Upserting product %s (op=%s)", article_id, op)
        product = {k: v for k, v in event.items() if k not in _DEBEZIUM_META}
        await _async_upsert(product)
    elif op == "d":
        logger.info("Deleting product %s", article_id)
        await _async_delete(article_id)
    else:
        logger.warning("Unknown op '%s' for article_id %s", op, article_id)


async def _run_consumer_async() -> None:
    await _async_init_collection()

    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "group.id": GROUP_ID,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([TOPIC])
    logger.info("Subscribed to topic: %s (group: %s)", TOPIC, GROUP_ID)

    loop = asyncio.get_running_loop()

    try:
        while True:
            msg = await loop.run_in_executor(None, lambda: consumer.poll(1.0))

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug(
                        "Reached end of partition %s [%d] at offset %d",
                        msg.topic(),
                        msg.partition(),
                        msg.offset(),
                    )
                elif msg.error():
                    raise KafkaException(msg.error())
                continue

            ok = True
            if msg.value() is None:
                # tombstone message (delete) — key carries the article_id
                try:
                    key = json.loads(msg.key()) if msg.key() else {}
                    article_id = key.get("article_id")
                    if article_id:
                        logger.info("Tombstone delete for %s", article_id)
                        await _async_delete(article_id)
                except Exception as exc:
                    logger.error("Failed to process tombstone: %s", exc)
                    ok = False
            else:
                try:
                    event = json.loads(msg.value())
                    await _handle_event(event)
                except Exception as exc:
                    logger.error(
                        "Failed to process message at offset %d: %s",
                        msg.offset(),
                        exc,
                    )
                    ok = False

            # only commit on success (at-least-once delivery)
            if ok:
                await loop.run_in_executor(
                    None, lambda: consumer.commit(asynchronous=False)
                )

    except KeyboardInterrupt:
        logger.info("Consumer interrupted, shutting down...")
    finally:
        consumer.close()


def run_consumer() -> None:
    asyncio.run(_run_consumer_async())


if __name__ == "__main__":
    # retry loop — wait for Redpanda to be ready
    for attempt in range(10):
        try:
            run_consumer()
            break
        except KafkaException as exc:
            logger.warning("Kafka not ready (attempt %d/10): %s", attempt + 1, exc)
            time.sleep(5)