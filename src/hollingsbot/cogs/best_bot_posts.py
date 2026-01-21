"""Best Bot Posts - ELO-based tournament bracket for images and text posts."""

import asyncio
import concurrent.futures
import io
import logging
import os
import random
import re
import sqlite3
import textwrap
from pathlib import Path

import discord
from discord.ext import commands
from PIL import Image, ImageDraw, ImageFont

try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

__all__ = ["BestBotPosts"]

_LOG = logging.getLogger(__name__)

DB_PATH = os.getenv("PROMPT_DB_PATH", "/data/hollingsbot.db")
TOURNAMENT_CHANNEL_ID = os.getenv("TOURNAMENT_CHANNEL_ID")
GOOD_BOT_POSTS_CHANNEL_ID = os.getenv("GOOD_BOT_POSTS_CHANNEL_ID")
IMAGES_FOLDER = Path(os.getenv("BEST_BOT_POSTS_IMAGES", "generated/best_bot_posts"))
POLL_INTERVAL = 7
MATCHUP_DELAY = 1

# Text rendering settings
TEXT_CARD_WIDTH = 512
TEXT_CARD_HEIGHT = 512
TEXT_BG_COLOR = (54, 57, 63)
TEXT_FG_COLOR = (255, 255, 255)

# OCR settings - threshold for detecting white header
WHITE_THRESHOLD = 240
MIN_MARGIN_HEIGHT = 10


def _init_db():
    """Create the elo_posts and match_history tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS elo_posts (
            id INTEGER PRIMARY KEY,
            name TEXT,
            filename TEXT,
            post_type TEXT DEFAULT 'image',
            text_content TEXT,
            rating INTEGER DEFAULT 1000,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS match_history (
            id INTEGER PRIMARY KEY,
            winner_id INTEGER,
            loser_id INTEGER,
            winner_rating_before INTEGER,
            loser_rating_before INTEGER,
            winner_rating_after INTEGER,
            loser_rating_after INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (winner_id) REFERENCES elo_posts(id),
            FOREIGN KEY (loser_id) REFERENCES elo_posts(id)
        )
    """)
    conn.commit()
    conn.close()


def _get_random_pair():
    """Get two posts for a matchup using weighted strategy selection.

    Strategies (percentages configurable):
    - Fresh vs Fresh (30%): New posts compete against each other
    - Fresh vs Established (30%): Calibrate new posts against known quantities
    - High vs High (20%): Refine top tier rankings
    - Low vs Low (20%): Refine bottom tier rankings
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Get all posts with match counts
    all_posts = conn.execute("""
        SELECT *, (wins + losses) as total_matches
        FROM elo_posts
        ORDER BY total_matches ASC
    """).fetchall()
    conn.close()

    if len(all_posts) < 2:
        return all_posts

    total = len(all_posts)

    # Define percentile thresholds
    HIGH_RATING_PERCENTILE = 0.20  # Top 20% by rating
    LOW_RATING_PERCENTILE = 0.20   # Bottom 20% by rating

    # Separate rated (1+ matches) from unrated posts
    rated_posts = [p for p in all_posts if p['total_matches'] > 0]
    unrated_posts = [p for p in all_posts if p['total_matches'] == 0]

    # Fresh = unrated posts (0 matches)
    fresh_posts = unrated_posts

    # High/Low pools only from rated posts (skip unrated)
    if rated_posts:
        by_rating = sorted(rated_posts, key=lambda p: p['rating'])
        rating_cutoff = max(1, int(len(rated_posts) * HIGH_RATING_PERCENTILE))
        low_posts = by_rating[:rating_cutoff]
        high_posts = by_rating[-rating_cutoff:]
    else:
        low_posts = []
        high_posts = []

    # Build strategies only for pools that have enough posts
    strategies = []
    if len(fresh_posts) >= 2:
        strategies.append(('fresh_vs_fresh', 40))
    if len(high_posts) >= 2:
        strategies.append(('high_vs_high', 30))
    if len(low_posts) >= 2:
        strategies.append(('low_vs_low', 30))

    # Fallback if no strategies available
    if not strategies:
        return random.sample(list(all_posts), 2)

    # Pick strategy weighted randomly
    total_weight = sum(w for _, w in strategies)
    roll = random.randint(1, total_weight)
    cumulative = 0
    chosen_strategy = strategies[0][0]
    for name, weight in strategies:
        cumulative += weight
        if roll <= cumulative:
            chosen_strategy = name
            break

    # Select pair based on strategy
    def pick_two(pool):
        return random.sample(list(pool), 2)

    if chosen_strategy == 'fresh_vs_fresh':
        return pick_two(fresh_posts)
    elif chosen_strategy == 'high_vs_high':
        return pick_two(high_posts)
    elif chosen_strategy == 'low_vs_low':
        return pick_two(low_posts)

    return random.sample(list(all_posts), 2)


def _update_ratings(winner_id, winner_new_rating, loser_id, loser_new_rating,
                    winner_rating_before=None, loser_rating_before=None):
    """Update ratings after a match and record history."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE elo_posts SET rating = ?, wins = wins + 1 WHERE id = ?",
                 (winner_new_rating, winner_id))
    conn.execute("UPDATE elo_posts SET rating = ?, losses = losses + 1 WHERE id = ?",
                 (loser_new_rating, loser_id))
    conn.execute("""
        INSERT INTO match_history (winner_id, loser_id, winner_rating_before, loser_rating_before,
                                   winner_rating_after, loser_rating_after)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (winner_id, loser_id, winner_rating_before, loser_rating_before,
          winner_new_rating, loser_new_rating))
    conn.commit()
    conn.close()


def _elo_calc(winner_rating, loser_rating, k=50):
    """Calculate new ELO ratings. Returns (winner_new, loser_new)."""
    expected = 1.0 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
    winner_new = winner_rating + k * (1 - expected)
    loser_new = loser_rating + k * (0 - (1 - expected))
    return int(winner_new), int(loser_new)


def _filename_exists(filename: str) -> bool:
    """Check if a filename is already in the database."""
    conn = sqlite3.connect(DB_PATH)
    result = conn.execute("SELECT 1 FROM elo_posts WHERE filename = ?", (filename,)).fetchone()
    conn.close()
    return result is not None


def _insert_post(name: str, filename: str, post_type: str = "image", text_content: str = None) -> int:
    """Insert a new post and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO elo_posts (name, filename, post_type, text_content) VALUES (?, ?, ?, ?)",
        (name, filename, post_type, text_content)
    )
    post_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return post_id


# --- OCR Caption Parsing ---

def _get_header_height(img: Image.Image) -> int:
    """Find the height of a white/light header bar at the top of the image.

    Scans down looking for where the actual image content begins (not text).
    Text in the header has sparse dark pixels; image content has dense dark pixels.
    Returns 0 if no header is detected.
    """
    gray = img.convert('L')
    width, height = gray.size

    # Sample more positions across the row
    num_samples = 20
    sample_positions = [int(width * i / num_samples) for i in range(1, num_samples)]

    for y in range(min(height, 150)):  # Headers shouldn't be more than 150px
        # Count how many sample points are dark
        dark_count = sum(1 for x in sample_positions if gray.getpixel((x, y)) < WHITE_THRESHOLD)
        dark_ratio = dark_count / len(sample_positions)

        # Text = sparse dark pixels (< 30%), Image = dense dark pixels (> 50%)
        if dark_ratio > 0.5:
            return y if y >= MIN_MARGIN_HEIGHT else 0

    return 0


MAX_CAPTION_LENGTH = 100  # Max chars for OCR-extracted captions


def _extract_caption_ocr(img: Image.Image) -> str | None:
    """Extract caption text from white header area using OCR.

    Tries header OCR first, then falls back to full image OCR.
    Returns None only if OCR completely fails or finds no text.
    """
    if not HAS_OCR:
        return None

    def _clean_ocr_text(text: str) -> str:
        """Clean up OCR output."""
        text = text.replace('\n', ' ').strip()
        # Limit length to avoid garbage dumps
        if len(text) > MAX_CAPTION_LENGTH:
            text = text[:MAX_CAPTION_LENGTH].rsplit(' ', 1)[0] + "..."
        return text

    # Try header OCR first
    header_height = _get_header_height(img)
    if header_height > 0:
        header = img.crop((0, 0, img.width, header_height))
        try:
            text = pytesseract.image_to_string(header).strip()
            text = _clean_ocr_text(text)
            if text:
                return text
        except Exception as e:
            _LOG.warning(f"Header OCR failed: {e}")

    # Fall back to full image OCR
    try:
        text = pytesseract.image_to_string(img).strip()
        text = _clean_ocr_text(text)
        if text:
            _LOG.info(f"Used full image OCR: {text[:50]}...")
            return text
    except Exception as e:
        _LOG.warning(f"Full image OCR failed: {e}")

    return None


def _extract_caption_from_filename(filename: str) -> str:
    """Try to extract a readable caption from filename.

    Handles patterns like:
    - prompt_text_model-name_seed-12345.png
    - a_cat_sitting_on_a_chair.png
    - 1068005240764387398-514b9.png (Discord ID - returns "Untitled")
    """
    # Remove extension
    name = Path(filename).stem

    # Remove common suffixes (hash, seed, model info)
    # Pattern: text followed by 5-char alphanumeric hash
    parts = name.rsplit('-', 1)
    if len(parts) == 2 and len(parts[1]) == 5 and parts[1].isalnum():
        name = parts[0]

    # Check if result is just a numeric ID (Discord message/attachment ID)
    if name.isdigit() and len(name) > 10:
        return "Untitled"

    # Check if it's a hex hash
    if re.match(r'^[0-9a-f]{20,}$', name, re.IGNORECASE):
        return "Untitled"

    # Replace underscores with spaces
    name = name.replace('_', ' ')

    # Remove seed patterns like "seed 12345" or "seed-12345"
    name = re.sub(r'\s*seed[\s-]*\d+', '', name, flags=re.IGNORECASE)

    # Remove model name patterns (usually at the end)
    # Pattern: replicate-openai-gpt-image-1.5, flux-schnell, google-imagen-4, etc.
    name = re.sub(r'\s*(replicate|openai|flux|stable[-.]?diffusion|google[-.]?imagen?)[-.\w]*\s*$', '', name, flags=re.IGNORECASE)

    return name.strip() or "Untitled"


def ingest_image(img_path: Path, use_ocr: bool = True) -> int | None:
    """Ingest an image file into the ELO database.

    - Parses caption from header (OCR) or filename
    - Copies image to standard location with ID-based name
    - Creates database record

    Returns the new post ID, or None if already exists.
    """
    filename = img_path.name

    # Skip if already ingested
    if _filename_exists(filename):
        return None

    # Load image
    try:
        img = Image.open(img_path)
    except Exception as e:
        _LOG.warning(f"Failed to open {img_path}: {e}")
        return None

    # Try to extract caption
    caption = None
    if use_ocr:
        caption = _extract_caption_ocr(img)

    if not caption:
        caption = _extract_caption_from_filename(filename)

    # Insert into database
    post_id = _insert_post(name=caption, filename=filename, post_type="image")

    # Copy/convert to standard location
    dest_path = IMAGES_FOLDER / f"{post_id}.png"
    try:
        # Convert to RGB if needed (handles RGBA, P mode, etc)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        img.save(dest_path, 'PNG')
    except Exception as e:
        _LOG.warning(f"Failed to save {dest_path}: {e}")
        # Still keep the db record, image might already be in place

    _LOG.info(f"Ingested post {post_id}: {caption}")
    return post_id


def ingest_text_post(text: str, name: str = None) -> int:
    """Ingest a text post into the ELO database."""
    display_name = name or (text[:50] + "..." if len(text) > 50 else text)
    post_id = _insert_post(name=display_name, filename=None, post_type="text", text_content=text)
    _LOG.info(f"Ingested text post {post_id}: {display_name}")
    return post_id


# --- Image Rendering ---

def _render_text_card(text: str) -> Image.Image:
    """Render text content as an image card."""
    img = Image.new('RGB', (TEXT_CARD_WIDTH, TEXT_CARD_HEIGHT), TEXT_BG_COLOR)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except (OSError, IOError):
        font = ImageFont.load_default()

    wrapped = textwrap.fill(text, width=35)
    bbox = draw.textbbox((0, 0), wrapped, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (TEXT_CARD_WIDTH - text_w) // 2
    y = (TEXT_CARD_HEIGHT - text_h) // 2
    draw.text((x, y), wrapped, font=font, fill=TEXT_FG_COLOR)

    return img


def _get_post_image(post) -> Image.Image:
    """Get the image for a post (loads file for image posts, renders for text)."""
    if post['post_type'] == 'text':
        return _render_text_card(post['text_content'] or post['name'])
    else:
        return Image.open(IMAGES_FOLDER / f"{post['id']}.png")


def _build_matchup_image(post_a, post_b):
    """Combine two posts side by side."""
    img1 = _get_post_image(post_a)
    img2 = _get_post_image(post_b)
    combined = Image.new('RGB', (img1.width + img2.width + 50, max(img1.height, img2.height)), (0, 0, 0))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width + 50, 0))
    buf = io.BytesIO()
    combined.save(buf, 'PNG')
    buf.seek(0)
    return buf


# --- Cog ---

class BestBotPosts(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.running = False
        self.good_bot_channel_ids = set()

        if GOOD_BOT_POSTS_CHANNEL_ID:
            for cid in GOOD_BOT_POSTS_CHANNEL_ID.split(','):
                cid = cid.strip()
                if cid.isdigit():
                    self.good_bot_channel_ids.add(int(cid))

        _init_db()
        IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)

    async def cog_load(self):
        if not TOURNAMENT_CHANNEL_ID:
            _LOG.warning("TOURNAMENT_CHANNEL_ID not set, matchups disabled")
        else:
            self.running = True
            self.bot.loop.create_task(self._run_loop())

        if not self.good_bot_channel_ids:
            _LOG.warning("GOOD_BOT_POSTS_CHANNEL_ID not set, auto-ingestion disabled")

    async def cog_unload(self):
        self.running = False

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Auto-ingest images posted to good-bot-posts channel."""
        if message.channel.id not in self.good_bot_channel_ids:
            return
        if message.author.bot:
            return

        # Process attachments
        for attachment in message.attachments:
            if not attachment.content_type or not attachment.content_type.startswith('image/'):
                continue

            # Download and ingest
            try:
                img_data = await attachment.read()
                temp_path = IMAGES_FOLDER / f"temp_{attachment.filename}"
                temp_path.write_bytes(img_data)

                post_id = ingest_image(temp_path, use_ocr=HAS_OCR)
                if post_id:
                    await message.add_reaction("\u2705")  # checkmark

                # Clean up temp file (image was copied to id.png)
                if temp_path.exists():
                    temp_path.unlink()

            except Exception as e:
                _LOG.exception(f"Failed to ingest attachment: {e}")

        # Check for text-only posts (no attachments, has content)
        if not message.attachments and message.content.strip():
            text = message.content.strip()
            if len(text) > 10:  # Minimum length for text posts
                ingest_text_post(text)
                await message.add_reaction("\u2705")

    async def _run_loop(self):
        await self.bot.wait_until_ready()
        channel = self.bot.get_channel(int(TOURNAMENT_CHANNEL_ID))
        if not channel:
            _LOG.error(f"Tournament channel {TOURNAMENT_CHANNEL_ID} not found")
            return

        while self.running:
            try:
                await self._run_matchup(channel)
            except Exception as e:
                _LOG.exception(f"Matchup error: {e}")
            await asyncio.sleep(MATCHUP_DELAY)

    async def _run_matchup(self, channel):
        posts = _get_random_pair()
        if len(posts) < 2:
            _LOG.warning("Not enough posts for matchup")
            await asyncio.sleep(60)
            return

        post_a, post_b = posts[0], posts[1]

        try:
            img_buf = _build_matchup_image(post_a, post_b)
        except FileNotFoundError as e:
            _LOG.warning(f"Image not found: {e}")
            return

        msg_text = f"**A: {post_a['name']}** (||`{post_a['rating']:4d}`||)\n**B: {post_b['name']}** (||`{post_b['rating']:4d}`||)"
        msg = await channel.send(content=msg_text, file=discord.File(fp=img_buf, filename="matchup.png"))
        await msg.add_reaction("\U0001f170")
        await msg.add_reaction("\U0001f171")

        winner = await self._poll_winner(msg)
        if winner is None:
            return

        if winner == 0:
            w_new, l_new = _elo_calc(post_a['rating'], post_b['rating'])
            _update_ratings(post_a['id'], w_new, post_b['id'], l_new,
                           post_a['rating'], post_b['rating'])
            await channel.send(f"Winner: **{post_a['name']}** (+{w_new - post_a['rating']} pts)")
        else:
            w_new, l_new = _elo_calc(post_b['rating'], post_a['rating'])
            _update_ratings(post_b['id'], w_new, post_a['id'], l_new,
                           post_b['rating'], post_a['rating'])
            await channel.send(f"Winner: **{post_b['name']}** (+{w_new - post_b['rating']} pts)")

    async def _poll_winner(self, msg):
        """Poll until same winner twice in a row, requiring at least one human vote.

        Waits indefinitely until a human votes.
        """
        prev_winner = None
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            msg = await msg.channel.fetch_message(msg.id)

            a_count = b_count = 1
            for r in msg.reactions:
                if str(r.emoji) == "\U0001f170":
                    a_count = r.count
                elif str(r.emoji) == "\U0001f171":
                    b_count = r.count

            # Require at least one human vote (bot adds 1 to each, so need > 2 total)
            if a_count + b_count <= 2:
                prev_winner = None
                continue

            if a_count > b_count:
                winner = 0
            elif b_count > a_count:
                winner = 1
            else:
                winner = None
                prev_winner = None
                continue

            if winner == prev_winner:
                return winner
            prev_winner = winner

    @commands.command(name="score")
    async def score_cmd(self, ctx):
        """Show highest and lowest rated posts."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        best = conn.execute("SELECT * FROM elo_posts ORDER BY rating DESC LIMIT 1").fetchone()
        worst = conn.execute("SELECT * FROM elo_posts ORDER BY rating ASC LIMIT 1").fetchone()
        total = conn.execute("SELECT COUNT(*) FROM elo_posts").fetchone()[0]
        conn.close()

        if not best or not worst:
            await ctx.send("No posts in database")
            return

        await ctx.send(
            f"**{total} posts in bracket**\n"
            f"**Best:** {best['name']} - {best['rating']} pts ({best['wins']}W/{best['losses']}L)\n"
            f"**Worst:** {worst['name']} - {worst['rating']} pts ({worst['wins']}W/{worst['losses']}L)"
        )

    @commands.command(name="backfill")
    @commands.is_owner()
    async def backfill_cmd(self, ctx, use_ocr: bool = True):
        """Clear database and backfill 10 random images from the folder."""
        try:
            # Clear tables first
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM match_history")
            conn.execute("DELETE FROM elo_posts")
            conn.commit()
            conn.close()
            await ctx.send(f"Cleared database. Backfilling 10 random images from {IMAGES_FOLDER}...")

            # Gather eligible images
            candidates = []
            for img_path in IMAGES_FOLDER.glob("*"):
                if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp', '.gif'):
                    continue
                if img_path.name.startswith("temp_"):
                    continue
                if img_path.stem.isdigit():
                    continue
                candidates.append(img_path)

            await ctx.send(f"Found {len(candidates)} eligible images, ingesting in parallel...")

            # Use ThreadPoolExecutor for parallel OCR (8 workers)
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    loop.run_in_executor(executor, ingest_image, img_path, use_ocr)
                    for img_path in candidates
                ]
                results = await asyncio.gather(*futures)

            count = sum(1 for r in results if r is not None)
            await ctx.send(f"Backfill complete: {count} posts ingested")
        except Exception as e:
            _LOG.exception("Backfill failed")
            await ctx.send(f"Backfill failed: {e}")


async def setup(bot):
    await bot.add_cog(BestBotPosts(bot))
