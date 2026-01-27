"""Best Bot Posts - Single-elimination tournament bracket for images and text posts."""

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
CHAMPIONS_INTERVAL = 16  # Run champions tournament every N regular tournaments

# Text rendering settings
TEXT_CARD_WIDTH = 512
TEXT_CARD_HEIGHT = 512
TEXT_BG_COLOR = (54, 57, 63)
TEXT_FG_COLOR = (255, 255, 255)

# OCR settings
WHITE_THRESHOLD = 240
MIN_MARGIN_HEIGHT = 10

# Standard seeded bracket matchups for Round 1 (seed pairs, 1-indexed)
SEEDED_BRACKET = [
    (1, 16), (8, 9), (4, 13), (5, 12),
    (2, 15), (7, 10), (3, 14), (6, 11),
]

ROUND_NAMES = {
    1: "Round of 16",
    2: "Quarterfinals",
    3: "Semifinals",
    4: "Final",
}

# Placement for losers by round eliminated
PLACEMENT_BY_ROUND = {
    1: 9,   # Lost in Round of 16 -> 9th
    2: 5,   # Lost in Quarterfinals -> 5th
    3: 3,   # Lost in Semifinals -> 3rd
    4: 2,   # Lost in Final -> 2nd
}


def _init_db():
    """Create tables if they don't exist and add new columns."""
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

    # Legacy table - keep for historical data
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

    # New tournament tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tournaments (
            id INTEGER PRIMARY KEY,
            tournament_number INTEGER NOT NULL,
            is_champions INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active',
            winner_id INTEGER,
            started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME,
            FOREIGN KEY (winner_id) REFERENCES elo_posts(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tournament_entrants (
            tournament_id INTEGER,
            seed INTEGER,
            post_id INTEGER,
            placement INTEGER,
            PRIMARY KEY (tournament_id, seed),
            FOREIGN KEY (tournament_id) REFERENCES tournaments(id),
            FOREIGN KEY (post_id) REFERENCES elo_posts(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tournament_matches (
            id INTEGER PRIMARY KEY,
            tournament_id INTEGER NOT NULL,
            round INTEGER NOT NULL,
            match_index INTEGER NOT NULL,
            post_a_id INTEGER,
            post_b_id INTEGER,
            winner_id INTEGER,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (tournament_id) REFERENCES tournaments(id),
            FOREIGN KEY (post_a_id) REFERENCES elo_posts(id),
            FOREIGN KEY (post_b_id) REFERENCES elo_posts(id),
            FOREIGN KEY (winner_id) REFERENCES elo_posts(id)
        )
    """)

    # Add new columns to elo_posts (ignore errors if already exist)
    for col, default in [
        ("tournament_wins", "0"),
        ("tournaments_played", "0"),
        ("best_placement", "NULL"),
        ("total_placement_points", "0"),
    ]:
        try:
            conn.execute(
                f"ALTER TABLE elo_posts ADD COLUMN {col} INTEGER DEFAULT {default}"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists

    conn.commit()
    conn.close()


# --- Post Management ---

def _filename_exists(filename: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    result = conn.execute(
        "SELECT 1 FROM elo_posts WHERE filename = ?", (filename,)
    ).fetchone()
    conn.close()
    return result is not None


def _insert_post(
    name: str,
    filename: str,
    post_type: str = "image",
    text_content: str | None = None,
) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO elo_posts (name, filename, post_type, text_content) VALUES (?, ?, ?, ?)",
        (name, filename, post_type, text_content),
    )
    post_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return post_id


# --- OCR Caption Parsing ---

def _get_header_height(img: Image.Image) -> int:
    gray = img.convert("L")
    width, height = gray.size
    num_samples = 20
    sample_positions = [int(width * i / num_samples) for i in range(1, num_samples)]

    for y in range(min(height, 150)):
        dark_count = sum(
            1 for x in sample_positions if gray.getpixel((x, y)) < WHITE_THRESHOLD
        )
        dark_ratio = dark_count / len(sample_positions)
        if dark_ratio > 0.5:
            return y if y >= MIN_MARGIN_HEIGHT else 0

    return 0


MAX_CAPTION_LENGTH = 100


def _extract_caption_ocr(img: Image.Image) -> str | None:
    if not HAS_OCR:
        return None

    def _clean_ocr_text(text: str) -> str:
        text = text.replace("\n", " ").strip()
        if len(text) > MAX_CAPTION_LENGTH:
            text = text[:MAX_CAPTION_LENGTH].rsplit(" ", 1)[0] + "..."
        return text

    header_height = _get_header_height(img)
    if header_height > 0:
        header = img.crop((0, 0, img.width, header_height))
        try:
            text = pytesseract.image_to_string(header).strip()
            text = _clean_ocr_text(text)
            if text:
                return text
        except Exception as e:
            _LOG.warning("Header OCR failed: %s", e)

    try:
        text = pytesseract.image_to_string(img).strip()
        text = _clean_ocr_text(text)
        if text:
            _LOG.info("Used full image OCR: %s...", text[:50])
            return text
    except Exception as e:
        _LOG.warning("Full image OCR failed: %s", e)

    return None


def _extract_caption_from_filename(filename: str) -> str:
    name = Path(filename).stem
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) == 5 and parts[1].isalnum():
        name = parts[0]

    if name.isdigit() and len(name) > 10:
        return "Untitled"

    if re.match(r"^[0-9a-f]{20,}$", name, re.IGNORECASE):
        return "Untitled"

    name = name.replace("_", " ")
    name = re.sub(r"\s*seed[\s-]*\d+", "", name, flags=re.IGNORECASE)
    name = re.sub(
        r"\s*(replicate|openai|flux|stable[-.]?diffusion|google[-.]?imagen?)[-.\w]*\s*$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    return name.strip() or "Untitled"


def ingest_image(img_path: Path, use_ocr: bool = True) -> int | None:
    """Ingest an image file into the database.

    Returns the new post ID, or None if already exists.
    """
    filename = img_path.name
    if _filename_exists(filename):
        return None

    try:
        img = Image.open(img_path)
    except Exception as e:
        _LOG.warning("Failed to open %s: %s", img_path, e)
        return None

    caption = None
    if use_ocr:
        caption = _extract_caption_ocr(img)

    if not caption:
        caption = _extract_caption_from_filename(filename)

    post_id = _insert_post(name=caption, filename=filename, post_type="image")

    dest_path = IMAGES_FOLDER / f"{post_id}.png"
    try:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(dest_path, "PNG")
    except Exception as e:
        _LOG.warning("Failed to save %s: %s", dest_path, e)

    _LOG.info("Ingested post %d: %s", post_id, caption)
    return post_id


def ingest_text_post(text: str, name: str | None = None) -> int:
    """Ingest a text post into the database."""
    display_name = name or (text[:50] + "..." if len(text) > 50 else text)
    post_id = _insert_post(
        name=display_name, filename=None, post_type="text", text_content=text
    )
    _LOG.info("Ingested text post %d: %s", post_id, display_name)
    return post_id


# --- Image Rendering ---

def _render_text_card(text: str) -> Image.Image:
    """Render text content as an image card."""
    img = Image.new("RGB", (TEXT_CARD_WIDTH, TEXT_CARD_HEIGHT), TEXT_BG_COLOR)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
        )
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
    if post["post_type"] == "text":
        return _render_text_card(post["text_content"] or post["name"])
    return Image.open(IMAGES_FOLDER / f"{post['id']}.png")


def _build_matchup_image(post_a, post_b):
    """Combine two posts side by side."""
    img1 = _get_post_image(post_a)
    img2 = _get_post_image(post_b)
    combined = Image.new(
        "RGB",
        (img1.width + img2.width + 50, max(img1.height, img2.height)),
        (0, 0, 0),
    )
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width + 50, 0))
    buf = io.BytesIO()
    combined.save(buf, "PNG")
    buf.seek(0)
    return buf


# --- Tournament Logic ---

def _get_post_by_id(conn, post_id):
    """Fetch a post row by ID."""
    return conn.execute(
        "SELECT * FROM elo_posts WHERE id = ?", (post_id,)
    ).fetchone()


def _get_active_tournament(conn):
    """Return the active tournament row, or None."""
    return conn.execute(
        "SELECT * FROM tournaments WHERE status = 'active' ORDER BY id DESC LIMIT 1"
    ).fetchone()


def _count_completed_regular(conn):
    """Count completed non-champions tournaments."""
    row = conn.execute(
        "SELECT COUNT(*) FROM tournaments WHERE status = 'completed' AND is_champions = 0"
    ).fetchone()
    return row[0]


def _last_completed_tournament(conn):
    """Return the most recently completed tournament, or None."""
    return conn.execute(
        "SELECT * FROM tournaments WHERE status = 'completed' ORDER BY id DESC LIMIT 1"
    ).fetchone()


def _should_run_champions(conn):
    """Check if it's time for a champions tournament."""
    completed_regular = _count_completed_regular(conn)
    if completed_regular == 0 or completed_regular % CHAMPIONS_INTERVAL != 0:
        return False
    # Don't run back-to-back champions
    last = _last_completed_tournament(conn)
    if last and last["is_champions"]:
        return False
    return True


def _select_regular_entrants(conn):
    """Select 16 entrants for a regular tournament, prioritizing least-seen posts."""
    rows = conn.execute(
        "SELECT id FROM elo_posts ORDER BY tournaments_played ASC, RANDOM() LIMIT 16"
    ).fetchall()
    return [r[0] for r in rows]


def _select_champions_entrants(conn):
    """Select 16 entrants for a champions tournament from recent regular winners."""
    recent = conn.execute(
        """SELECT id, winner_id FROM tournaments
           WHERE status = 'completed' AND is_champions = 0
           ORDER BY id DESC LIMIT ?""",
        (CHAMPIONS_INTERVAL,),
    ).fetchall()

    # Collect unique winners
    seen = set()
    entrants = []
    for t in recent:
        wid = t["winner_id"]
        if wid and wid not in seen:
            seen.add(wid)
            entrants.append(wid)

    # If we need more (duplicate winners), fill with runner-ups
    if len(entrants) < 16:
        tournament_ids = [t["id"] for t in recent]
        if tournament_ids:
            placeholders = ",".join("?" * len(tournament_ids))
            runners = conn.execute(
                f"""SELECT post_id FROM tournament_entrants
                    WHERE tournament_id IN ({placeholders}) AND placement = 2
                    ORDER BY tournament_id DESC""",
                tournament_ids,
            ).fetchall()
            for r in runners:
                if r["post_id"] not in seen and len(entrants) < 16:
                    seen.add(r["post_id"])
                    entrants.append(r["post_id"])

    # If still not enough, fill with top tournament winners overall
    if len(entrants) < 16:
        fillers = conn.execute(
            "SELECT id FROM elo_posts WHERE tournament_wins > 0 "
            "ORDER BY tournament_wins DESC, RANDOM()"
        ).fetchall()
        for f in fillers:
            if f["id"] not in seen and len(entrants) < 16:
                seen.add(f["id"])
                entrants.append(f["id"])

    return entrants[:16]


def _create_tournament(conn, is_champions=False):
    """Create a new tournament with 16 entrants and 15 matches. Returns tournament id."""
    row = conn.execute(
        "SELECT COALESCE(MAX(tournament_number), 0) FROM tournaments"
    ).fetchone()
    tournament_number = row[0] + 1

    if is_champions:
        entrant_ids = _select_champions_entrants(conn)
    else:
        entrant_ids = _select_regular_entrants(conn)

    if len(entrant_ids) < 16:
        _LOG.warning(
            "Only %d posts available, need 16 for tournament", len(entrant_ids)
        )
        return None

    cursor = conn.execute(
        "INSERT INTO tournaments (tournament_number, is_champions) VALUES (?, ?)",
        (tournament_number, 1 if is_champions else 0),
    )
    tid = cursor.lastrowid

    # Assign seeds
    if is_champions:
        post_stats = []
        for pid in entrant_ids:
            row = conn.execute(
                "SELECT tournament_wins FROM elo_posts WHERE id = ?", (pid,)
            ).fetchone()
            post_stats.append((pid, row[0] if row else 0))
        post_stats.sort(key=lambda x: x[1], reverse=True)
        seeded_ids = [pid for pid, _ in post_stats]
    else:
        seeded_ids = list(entrant_ids)
        random.shuffle(seeded_ids)

    # Insert entrants (seed is 1-indexed)
    for seed_num, post_id in enumerate(seeded_ids, 1):
        conn.execute(
            "INSERT INTO tournament_entrants (tournament_id, seed, post_id) VALUES (?, ?, ?)",
            (tid, seed_num, post_id),
        )

    seed_to_post = {i + 1: pid for i, pid in enumerate(seeded_ids)}

    # Create Round 1 matches (8 matches) with pre-populated entrants
    for match_idx, (seed_a, seed_b) in enumerate(SEEDED_BRACKET):
        conn.execute(
            """INSERT INTO tournament_matches
               (tournament_id, round, match_index, post_a_id, post_b_id, status)
               VALUES (?, 1, ?, ?, ?, 'pending')""",
            (tid, match_idx, seed_to_post[seed_a], seed_to_post[seed_b]),
        )

    # Create Rounds 2-4 matches (empty, filled as winners advance)
    for rnd in range(2, 5):
        num_matches = 8 >> (rnd - 1)  # round 2: 4, round 3: 2, round 4: 1
        for match_idx in range(num_matches):
            conn.execute(
                """INSERT INTO tournament_matches
                   (tournament_id, round, match_index, status)
                   VALUES (?, ?, ?, 'pending')""",
                (tid, rnd, match_idx),
            )

    conn.commit()
    label = "Champions " if is_champions else ""
    _LOG.info(
        "Created %sTournament #%d (id=%d) with 16 entrants",
        label, tournament_number, tid,
    )
    return tid


def _get_next_pending_match(conn, tournament_id):
    """Find the next playable match: lowest round, lowest match_index, both entrants set."""
    return conn.execute(
        """SELECT * FROM tournament_matches
           WHERE tournament_id = ? AND status = 'pending'
             AND post_a_id IS NOT NULL AND post_b_id IS NOT NULL
           ORDER BY round ASC, match_index ASC
           LIMIT 1""",
        (tournament_id,),
    ).fetchone()


def _advance_winner(conn, tournament_id, rnd, match_index, winner_id):
    """Place the winner into the next round's match slot."""
    if rnd >= 4:
        return  # Final has no next round

    next_round = rnd + 1
    next_match_index = match_index // 2
    slot = "post_a_id" if match_index % 2 == 0 else "post_b_id"

    conn.execute(
        f"""UPDATE tournament_matches
            SET {slot} = ?
            WHERE tournament_id = ? AND round = ? AND match_index = ?""",
        (winner_id, tournament_id, next_round, next_match_index),
    )


def _record_placement(conn, tournament_id, post_id, placement):
    """Set the placement for an entrant."""
    conn.execute(
        "UPDATE tournament_entrants SET placement = ? WHERE tournament_id = ? AND post_id = ?",
        (placement, tournament_id, post_id),
    )


def _finalize_tournament(conn, tournament_id):
    """Mark tournament completed, update all entrant stats."""
    winner_id = None

    # Get the final match to find the winner
    final = conn.execute(
        "SELECT * FROM tournament_matches WHERE tournament_id = ? AND round = 4",
        (tournament_id,),
    ).fetchone()
    if final and final["winner_id"]:
        winner_id = final["winner_id"]
        _record_placement(conn, tournament_id, winner_id, 1)
        conn.execute(
            "UPDATE elo_posts SET tournament_wins = tournament_wins + 1 WHERE id = ?",
            (winner_id,),
        )

    # Update all entrants' stats
    entrants = conn.execute(
        "SELECT * FROM tournament_entrants WHERE tournament_id = ?",
        (tournament_id,),
    ).fetchall()
    for e in entrants:
        pid = e["post_id"]
        placement = e["placement"]
        conn.execute(
            "UPDATE elo_posts SET tournaments_played = tournaments_played + 1 WHERE id = ?",
            (pid,),
        )
        if placement is not None:
            conn.execute(
                "UPDATE elo_posts SET total_placement_points = total_placement_points + ? WHERE id = ?",
                (placement, pid),
            )
            conn.execute(
                """UPDATE elo_posts SET best_placement = ?
                   WHERE id = ? AND (best_placement IS NULL OR best_placement > ?)""",
                (placement, pid, placement),
            )

    conn.execute(
        """UPDATE tournaments
           SET status = 'completed', winner_id = ?, completed_at = CURRENT_TIMESTAMP
           WHERE id = ?""",
        (winner_id, tournament_id),
    )
    conn.commit()


# --- Cog ---

class BestBotPosts(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.running = False
        self.good_bot_channel_ids = set()

        if GOOD_BOT_POSTS_CHANNEL_ID:
            for cid in GOOD_BOT_POSTS_CHANNEL_ID.split(","):
                cid = cid.strip()
                if cid.isdigit():
                    self.good_bot_channel_ids.add(int(cid))

        _init_db()
        IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)

    async def cog_load(self):
        if not TOURNAMENT_CHANNEL_ID:
            _LOG.warning("TOURNAMENT_CHANNEL_ID not set, tournaments disabled")
        else:
            # Recovery: reset any mid-vote matches back to pending
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            conn.execute(
                "UPDATE tournament_matches SET status = 'pending' WHERE status = 'active'"
            )
            conn.commit()
            conn.close()

            self.running = True
            self.bot.loop.create_task(self._run_loop())

        if not self.good_bot_channel_ids:
            _LOG.warning(
                "GOOD_BOT_POSTS_CHANNEL_ID not set, auto-ingestion disabled"
            )

    async def cog_unload(self):
        self.running = False

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Auto-ingest images posted to good-bot-posts channel."""
        if message.channel.id not in self.good_bot_channel_ids:
            return
        if message.author.bot:
            return

        for attachment in message.attachments:
            if (
                not attachment.content_type
                or not attachment.content_type.startswith("image/")
            ):
                continue
            try:
                img_data = await attachment.read()
                temp_path = IMAGES_FOLDER / f"temp_{attachment.filename}"
                temp_path.write_bytes(img_data)

                post_id = ingest_image(temp_path, use_ocr=HAS_OCR)
                if post_id:
                    await message.add_reaction("\u2705")

                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                _LOG.exception("Failed to ingest attachment")

        if not message.attachments and message.content.strip():
            text = message.content.strip()
            if len(text) > 10:
                ingest_text_post(text)
                await message.add_reaction("\u2705")

    # --- Tournament Main Loop ---

    async def _run_loop(self):
        await self.bot.wait_until_ready()
        channel = self.bot.get_channel(int(TOURNAMENT_CHANNEL_ID))
        if not channel:
            _LOG.error("Tournament channel %s not found", TOURNAMENT_CHANNEL_ID)
            return

        while self.running:
            try:
                await self._run_tournament_step(channel)
            except Exception:
                _LOG.exception("Tournament error")
            await asyncio.sleep(MATCHUP_DELAY)

    async def _run_tournament_step(self, channel):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        try:
            tournament = _get_active_tournament(conn)
            if tournament is None:
                is_champ = _should_run_champions(conn)
                tid = _create_tournament(conn, is_champions=is_champ)
                if tid is None:
                    await asyncio.sleep(60)
                    return
                tournament = conn.execute(
                    "SELECT * FROM tournaments WHERE id = ?", (tid,)
                ).fetchone()

                label = (
                    "CHAMPIONS TOURNAMENT"
                    if tournament["is_champions"]
                    else "Tournament"
                )
                await channel.send(
                    f"**{label} #{tournament['tournament_number']} has begun!**"
                )

            match = _get_next_pending_match(conn, tournament["id"])
            if match is None:
                pending = conn.execute(
                    "SELECT COUNT(*) FROM tournament_matches "
                    "WHERE tournament_id = ? AND status != 'completed'",
                    (tournament["id"],),
                ).fetchone()[0]
                if pending == 0:
                    _finalize_tournament(conn, tournament["id"])
                    await self._announce_tournament_complete(
                        channel, tournament["id"], conn
                    )
                    return
                await asyncio.sleep(5)
                return

            await self._run_match(channel, match, tournament, conn)

        finally:
            conn.close()

    async def _run_match(self, channel, match, tournament, conn):
        """Run a single tournament match: display, vote, record, advance."""
        match_id = match["id"]
        rnd = match["round"]
        match_idx = match["match_index"]
        tid = tournament["id"]

        conn.execute(
            "UPDATE tournament_matches SET status = 'active' WHERE id = ?",
            (match_id,),
        )
        conn.commit()

        post_a = _get_post_by_id(conn, match["post_a_id"])
        post_b = _get_post_by_id(conn, match["post_b_id"])

        if not post_a or not post_b:
            _LOG.error("Missing post for match %d", match_id)
            conn.execute(
                "UPDATE tournament_matches SET status = 'pending' WHERE id = ?",
                (match_id,),
            )
            conn.commit()
            return

        try:
            img_buf = _build_matchup_image(post_a, post_b)
        except FileNotFoundError as e:
            _LOG.warning("Image not found for match %d: %s", match_id, e)
            conn.execute(
                "UPDATE tournament_matches SET status = 'pending' WHERE id = ?",
                (match_id,),
            )
            conn.commit()
            return

        total_in_round = 8 >> (rnd - 1)  # 8, 4, 2, 1
        round_name = ROUND_NAMES.get(rnd, f"Round {rnd}")

        label = (
            "CHAMPIONS TOURNAMENT"
            if tournament["is_champions"]
            else "Tournament"
        )
        header = (
            f"**{label} #{tournament['tournament_number']} - "
            f"{round_name} (Match {match_idx + 1}/{total_in_round})**"
        )

        if tournament["is_champions"]:
            a_wins = post_a["tournament_wins"] or 0
            b_wins = post_b["tournament_wins"] or 0
            msg_text = (
                f"{header}\n"
                f"**A: {post_a['name']}** ({a_wins} tournament wins)\n"
                f"**B: {post_b['name']}** ({b_wins} tournament wins)"
            )
        else:
            msg_text = (
                f"{header}\n"
                f"**A: {post_a['name']}**\n"
                f"**B: {post_b['name']}**"
            )

        msg = await channel.send(
            content=msg_text,
            file=discord.File(fp=img_buf, filename="matchup.png"),
        )
        await msg.add_reaction("\U0001f170")  # A button
        await msg.add_reaction("\U0001f171")  # B button

        winner_idx = await self._poll_winner(msg)
        if winner_idx is None:
            conn.execute(
                "UPDATE tournament_matches SET status = 'pending' WHERE id = ?",
                (match_id,),
            )
            conn.commit()
            return

        if winner_idx == 0:
            winner_post = post_a
            loser_post = post_b
        else:
            winner_post = post_b
            loser_post = post_a

        conn.execute(
            "UPDATE tournament_matches SET winner_id = ?, status = 'completed' WHERE id = ?",
            (winner_post["id"], match_id),
        )

        loser_placement = PLACEMENT_BY_ROUND[rnd]
        _record_placement(conn, tid, loser_post["id"], loser_placement)

        _advance_winner(conn, tid, rnd, match_idx, winner_post["id"])
        conn.commit()

        if rnd < 4:
            next_round_name = ROUND_NAMES.get(rnd + 1, f"Round {rnd + 1}")
            await channel.send(
                f"**{winner_post['name']}** wins! Advances to {next_round_name}."
            )
        else:
            await channel.send(
                f"**{winner_post['name']}** wins the tournament!"
            )

    async def _announce_tournament_complete(self, channel, tournament_id, conn):
        """Announce the full tournament results."""
        tournament = conn.execute(
            "SELECT * FROM tournaments WHERE id = ?", (tournament_id,)
        ).fetchone()
        label = (
            "CHAMPIONS TOURNAMENT"
            if tournament["is_champions"]
            else "Tournament"
        )

        entrants = conn.execute(
            """SELECT te.placement, ep.name
               FROM tournament_entrants te
               JOIN elo_posts ep ON te.post_id = ep.id
               WHERE te.tournament_id = ?
               ORDER BY te.placement ASC""",
            (tournament_id,),
        ).fetchall()

        lines = [f"**{label} #{tournament['tournament_number']} Complete!**"]

        placement_groups = {}
        for e in entrants:
            p = e["placement"]
            if p not in placement_groups:
                placement_groups[p] = []
            placement_groups[p].append(e["name"])

        for p in sorted(placement_groups.keys()):
            names = " / ".join(placement_groups[p])
            suffix = (
                "st" if p == 1
                else "nd" if p == 2
                else "rd" if p == 3
                else "th"
            )
            lines.append(f"{p}{suffix}: {names}")

        await channel.send("\n".join(lines))

    async def _poll_winner(self, msg):
        """Poll until same winner twice in a row, requiring at least one human vote."""
        prev_winner = None
        while self.running:
            await asyncio.sleep(POLL_INTERVAL)
            if not self.running:
                return None
            msg = await msg.channel.fetch_message(msg.id)

            a_count = b_count = 1
            for r in msg.reactions:
                if str(r.emoji) == "\U0001f170":
                    a_count = r.count
                elif str(r.emoji) == "\U0001f171":
                    b_count = r.count

            if a_count + b_count <= 2:
                prev_winner = None
                continue

            if a_count > b_count:
                winner = 0
            elif b_count > a_count:
                winner = 1
            else:
                prev_winner = None
                continue

            if winner == prev_winner:
                return winner
            prev_winner = winner

        return None

    # --- Commands ---

    @commands.command(name="score")
    async def score_cmd(self, ctx):
        """Show tournament leaderboard."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        total_tournaments = conn.execute(
            "SELECT COUNT(*) FROM tournaments WHERE status = 'completed'"
        ).fetchone()[0]

        leaders = conn.execute(
            """SELECT name, tournament_wins, tournaments_played, total_placement_points
               FROM elo_posts
               WHERE tournament_wins > 0
               ORDER BY tournament_wins DESC, total_placement_points ASC
               LIMIT 10"""
        ).fetchall()

        total_posts = conn.execute("SELECT COUNT(*) FROM elo_posts").fetchone()[0]

        active = _get_active_tournament(conn)
        conn.close()

        if not leaders:
            await ctx.send(
                f"**{total_posts} posts in pool** - No tournaments completed yet."
            )
            return

        lines = [
            f"**Tournament Leaderboard ({total_tournaments} tournaments completed, {total_posts} posts)**"
        ]
        for i, row in enumerate(leaders, 1):
            avg = (
                row["total_placement_points"] / row["tournaments_played"]
                if row["tournaments_played"] > 0
                else 0
            )
            lines.append(
                f"{i}. {row['name']} - {row['tournament_wins']} wins "
                f"(avg placement: {avg:.1f})"
            )

        if active:
            c2 = sqlite3.connect(DB_PATH)
            c2.row_factory = sqlite3.Row
            current_match = c2.execute(
                """SELECT round FROM tournament_matches
                   WHERE tournament_id = ? AND status IN ('pending', 'active')
                     AND post_a_id IS NOT NULL AND post_b_id IS NOT NULL
                   ORDER BY round ASC LIMIT 1""",
                (active["id"],),
            ).fetchone()
            c2.close()
            if current_match:
                round_name = ROUND_NAMES.get(
                    current_match["round"], f"Round {current_match['round']}"
                )
                t_label = (
                    "Champions Tournament"
                    if active["is_champions"]
                    else "Tournament"
                )
                lines.append(
                    f"**Current: {t_label} #{active['tournament_number']} - {round_name}**"
                )

        await ctx.send("\n".join(lines))

    @commands.command(name="bracket")
    async def bracket_cmd(self, ctx):
        """Show current tournament bracket status."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        tournament = _get_active_tournament(conn)
        if not tournament:
            tournament = _last_completed_tournament(conn)
            if not tournament:
                conn.close()
                await ctx.send("No tournaments yet.")
                return

        label = (
            "CHAMPIONS TOURNAMENT"
            if tournament["is_champions"]
            else "Tournament"
        )
        lines = [
            f"**{label} #{tournament['tournament_number']}** ({tournament['status']})"
        ]

        matches = conn.execute(
            """SELECT tm.*, ea.name as name_a, eb.name as name_b, ew.name as winner_name
               FROM tournament_matches tm
               LEFT JOIN elo_posts ea ON tm.post_a_id = ea.id
               LEFT JOIN elo_posts eb ON tm.post_b_id = eb.id
               LEFT JOIN elo_posts ew ON tm.winner_id = ew.id
               WHERE tm.tournament_id = ?
               ORDER BY tm.round ASC, tm.match_index ASC""",
            (tournament["id"],),
        ).fetchall()

        conn.close()

        current_round = None
        for m in matches:
            if m["round"] != current_round:
                current_round = m["round"]
                round_name = ROUND_NAMES.get(
                    current_round, f"Round {current_round}"
                )
                lines.append(f"\n**{round_name}:**")

            a_name = m["name_a"] or "TBD"
            b_name = m["name_b"] or "TBD"

            if m["status"] == "completed":
                if m["winner_id"] == m["post_a_id"]:
                    lines.append(f"  **{a_name}** vs ~~{b_name}~~")
                else:
                    lines.append(f"  ~~{a_name}~~ vs **{b_name}**")
            elif m["status"] == "active":
                lines.append(f"  {a_name} vs {b_name} (VOTING)")
            else:
                lines.append(f"  {a_name} vs {b_name}")

        await ctx.send("\n".join(lines))

    @commands.command(name="backfill")
    @commands.is_owner()
    async def backfill_cmd(self, ctx, use_ocr: bool = True):
        """Clear database and backfill images from the folder."""
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM tournament_matches")
            conn.execute("DELETE FROM tournament_entrants")
            conn.execute("DELETE FROM tournaments")
            conn.execute("DELETE FROM match_history")
            conn.execute("DELETE FROM elo_posts")
            conn.commit()
            conn.close()
            await ctx.send(
                f"Cleared database. Backfilling images from {IMAGES_FOLDER}..."
            )

            candidates = []
            for img_path in IMAGES_FOLDER.glob("*"):
                if img_path.suffix.lower() not in (
                    ".png", ".jpg", ".jpeg", ".webp", ".gif",
                ):
                    continue
                if img_path.name.startswith("temp_"):
                    continue
                if img_path.stem.isdigit():
                    continue
                candidates.append(img_path)

            await ctx.send(
                f"Found {len(candidates)} eligible images, ingesting in parallel..."
            )

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    loop.run_in_executor(executor, ingest_image, img_path, use_ocr)
                    for img_path in candidates
                ]
                results = await asyncio.gather(*futures)

            count = sum(1 for r in results if r is not None)
            await ctx.send(f"Backfill complete: {count} posts ingested")
        except Exception:
            _LOG.exception("Backfill failed")
            await ctx.send("Backfill failed - check logs")


async def setup(bot):
    await bot.add_cog(BestBotPosts(bot))
