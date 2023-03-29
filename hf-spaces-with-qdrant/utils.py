import io
from functools import lru_cache

import requests
from PIL import Image


@lru_cache
def get_file(url):
    return Image.open(io.BytesIO(requests.get(url).content))


def get_images(results):
    seen_urls = set()
    unique_results = []
    for result in results:
        if not result.payload['url'] in seen_urls:
            seen_urls.add(result.payload['url'])
            unique_results.append(result)

    images = [(get_file(result.payload['url']), result.payload['caption'] +
               " - " + str(result.score)) for result in unique_results]

    return images
