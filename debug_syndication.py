"""Debug script to see what data the Twitter syndication API returns."""

import asyncio
import json
import math
import re
import sys
sys.path.insert(0, 'src')

import aiohttp

def extract_tweet_id(url):
    match = re.search(r"/status/(\d+)", url)
    return match.group(1) if match else None

def generate_syndication_token(tweet_id):
    tweet_id_num = int(tweet_id)
    value = (tweet_id_num / 1e15) * math.pi
    base36_str = ""
    int_part = int(value)
    while int_part > 0:
        digit = int_part % 36
        if digit < 10:
            base36_str = str(digit) + base36_str
        else:
            base36_str = chr(ord('a') + digit - 10) + base36_str
        int_part //= 36
    token = re.sub(r'(0+|\.)', '', base36_str)
    return token if token else "0"

async def debug_syndication():
    url = "https://x.com/hollingsfsdf/status/1982214423054852291"
    tweet_id = extract_tweet_id(url)

    if not tweet_id:
        print("Could not extract tweet ID")
        return

    token = generate_syndication_token(tweet_id)
    syndication_url = f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&token={token}"

    print(f"Tweet ID: {tweet_id}")
    print(f"Token: {token}")
    print(f"Syndication URL: {syndication_url}\n")

    async with aiohttp.ClientSession() as session:
        async with session.get(syndication_url) as response:
            data = await response.json()

    print("Full JSON response:")
    print("="*80)
    print(json.dumps(data, indent=2))
    print("="*80)

    print("\n\nExtracted fields:")
    print(f"User name: {data.get('user', {}).get('name')}")
    print(f"User screen_name (@): {data.get('user', {}).get('screen_name')}")
    print(f"User profile image: {data.get('user', {}).get('profile_image_url_https')}")
    print(f"Tweet text: {data.get('text')}")
    print(f"Created at: {data.get('created_at')}")
    print(f"Favorite count: {data.get('favorite_count')}")
    print(f"Reply count: {data.get('reply_count')}")
    print(f"Retweet count: {data.get('retweet_count')}")

    photos = data.get('photos', [])
    print(f"\nPhotos ({len(photos)}):")
    for i, photo in enumerate(photos, 1):
        print(f"  {i}. {photo.get('url')}")

if __name__ == "__main__":
    asyncio.run(debug_syndication())
