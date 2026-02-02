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
    (1, 16),
    (8, 9),
    (4, 13),
    (5, 12),
    (2, 15),
    (7, 10),
    (3, 14),
    (6, 11),
]

ROUND_NAMES = {
    1: "Round of 16",
    2: "Quarterfinals",
    3: "Semifinals",
    4: "Final",
}

# Bracket image layout constants
BRACKET_THUMB = 48
BRACKET_SLOT_H = 55
BRACKET_MARGIN_TOP = 60
BRACKET_MARGIN_LEFT = 60
BRACKET_ROUND_SPACING = 300
BRACKET_CANVAS_W = 1500
BRACKET_CANVAS_H = BRACKET_MARGIN_TOP + 16 * BRACKET_SLOT_H + 60
BRACKET_BG = (47, 49, 54)  # #2f3136
BRACKET_LINE_COLOR = (85, 85, 85)  # #555
BRACKET_WIN_COLOR = (255, 255, 80)  # Yellow for winners
BRACKET_TEXT_COLOR = (255, 255, 255)  # White for non-decided entrants
BRACKET_LOSE_COLOR = (102, 102, 102)  # #666
BRACKET_TBD_COLOR = (120, 120, 120)
BRACKET_CHAMPION_THUMB = 160

# Placement for losers by round eliminated
PLACEMENT_BY_ROUND = {
    1: 9,  # Lost in Round of 16 -> 9th
    2: 5,  # Lost in Quarterfinals -> 5th
    3: 3,  # Lost in Semifinals -> 3rd
    4: 2,  # Lost in Final -> 2nd
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
            conn.execute(f"ALTER TABLE elo_posts ADD COLUMN {col} INTEGER DEFAULT {default}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    conn.commit()
    conn.close()


# --- Post Management ---


def _filename_exists(filename: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    result = conn.execute("SELECT 1 FROM elo_posts WHERE filename = ?", (filename,)).fetchone()
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
        dark_count = sum(1 for x in sample_positions if gray.getpixel((x, y)) < WHITE_THRESHOLD)
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
    post_id = _insert_post(name=display_name, filename=None, post_type="text", text_content=text)
    _LOG.info("Ingested text post %d: %s", post_id, display_name)
    return post_id


# --- Image Rendering ---


def _render_text_card(text: str) -> Image.Image:
    """Render text content as an image card."""
    img = Image.new("RGB", (TEXT_CARD_WIDTH, TEXT_CARD_HEIGHT), TEXT_BG_COLOR)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
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


MATCHUP_MAX_DIM = 1024  # Max width/height per image in matchup


def _cap_image_size(img: Image.Image, max_dim: int = MATCHUP_MAX_DIM) -> Image.Image:
    """Downscale an image so neither dimension exceeds max_dim."""
    if img.width <= max_dim and img.height <= max_dim:
        return img
    ratio = min(max_dim / img.width, max_dim / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    return img.resize(new_size, Image.LANCZOS)


def _build_matchup_image(post_a, post_b):
    """Combine two posts side by side, capping each to MATCHUP_MAX_DIM."""
    img1 = _cap_image_size(_get_post_image(post_a))
    img2 = _cap_image_size(_get_post_image(post_b))
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


def _get_thumbnail(post, size=BRACKET_THUMB) -> Image.Image:
    """Load a post image, center-crop to square, resize to size x size."""
    try:
        img = _get_post_image(post)
    except Exception:
        # Fallback: gray square
        return Image.new("RGB", (size, size), (80, 80, 80))

    # Center-crop to square
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def _dim_thumbnail(thumb: Image.Image) -> Image.Image:
    """Apply a dark overlay to indicate a loser."""
    dimmed = thumb.copy()
    overlay = Image.new("RGBA", dimmed.size, (0, 0, 0, 160))
    dimmed = dimmed.convert("RGBA")
    dimmed = Image.alpha_composite(dimmed, overlay)
    return dimmed.convert("RGB")


def _load_bracket_font(size=13):
    """Load DejaVuSans at the given size, falling back to default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _load_bracket_font_bold(size=16):
    """Load DejaVuSans-Bold at the given size, falling back to regular."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return _load_bracket_font(size)


def _truncate_name(name: str, max_len=20) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


def _build_bracket_image(tournament_id, db_path=None) -> io.BytesIO:
    """Render a full tournament bracket as a PNG image.

    Returns a BytesIO buffer containing the PNG data.
    """
    _db = db_path or DB_PATH
    conn = sqlite3.connect(_db)
    conn.row_factory = sqlite3.Row

    tournament = conn.execute("SELECT * FROM tournaments WHERE id = ?", (tournament_id,)).fetchone()

    matches = conn.execute(
        """SELECT tm.*, ea.name as name_a, eb.name as name_b,
                  ea.post_type as type_a, eb.post_type as type_b,
                  ea.text_content as text_a, eb.text_content as text_b,
                  ea.filename as file_a, eb.filename as file_b
           FROM tournament_matches tm
           LEFT JOIN elo_posts ea ON tm.post_a_id = ea.id
           LEFT JOIN elo_posts eb ON tm.post_b_id = eb.id
           WHERE tm.tournament_id = ?
           ORDER BY tm.round ASC, tm.match_index ASC""",
        (tournament_id,),
    ).fetchall()

    conn.close()

    # Organize matches by round
    rounds = {}
    for m in matches:
        rnd = m["round"]
        if rnd not in rounds:
            rounds[rnd] = []
        rounds[rnd].append(m)

    # Create canvas
    canvas = Image.new("RGB", (BRACKET_CANVAS_W, BRACKET_CANVAS_H), BRACKET_BG)
    draw = ImageDraw.Draw(canvas)
    font = _load_bracket_font(13)
    font_bold = _load_bracket_font_bold(16)
    title_font = _load_bracket_font_bold(22)

    # Title
    label = "CHAMPIONS TOURNAMENT" if tournament and tournament["is_champions"] else "Tournament"
    t_num = tournament["tournament_number"] if tournament else "?"
    title = f"{label} #{t_num}"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text(
        ((BRACKET_CANVAS_W - title_w) // 2, 15),
        title,
        font=title_font,
        fill=BRACKET_WIN_COLOR,
    )

    # Round headers
    for rnd_num in range(1, 5):
        rx = BRACKET_MARGIN_LEFT + (rnd_num - 1) * BRACKET_ROUND_SPACING
        rnd_name = ROUND_NAMES.get(rnd_num, f"Round {rnd_num}")
        draw.text((rx, 40), rnd_name, font=font, fill=BRACKET_TBD_COLOR)

    # Champion header
    champ_x = BRACKET_MARGIN_LEFT + 4 * BRACKET_ROUND_SPACING
    draw.text((champ_x, 40), "Champion", font=font, fill=BRACKET_TBD_COLOR)

    # Compute Y positions for each slot in each round
    # Round 1: 16 slots (8 matches * 2 entrants), evenly spaced
    slot_positions = {}  # (round, match_index, 'a'|'b') -> y_center
    match_centers = {}  # (round, match_index) -> y_center (midpoint of the two slots)

    for match_idx in range(8):
        slot_a = match_idx * 2
        slot_b = match_idx * 2 + 1
        ya = BRACKET_MARGIN_TOP + slot_a * BRACKET_SLOT_H + BRACKET_SLOT_H // 2
        yb = BRACKET_MARGIN_TOP + slot_b * BRACKET_SLOT_H + BRACKET_SLOT_H // 2
        slot_positions[(1, match_idx, "a")] = ya
        slot_positions[(1, match_idx, "b")] = yb
        match_centers[(1, match_idx)] = (ya + yb) // 2

    # Later rounds: centered between the two feeder matches
    for rnd_num in range(2, 5):
        num_matches = 8 >> (rnd_num - 1)
        for mi in range(num_matches):
            feeder_a = mi * 2
            feeder_b = mi * 2 + 1
            y_a = match_centers[(rnd_num - 1, feeder_a)]
            y_b = match_centers[(rnd_num - 1, feeder_b)]
            mid = (y_a + y_b) // 2
            slot_positions[(rnd_num, mi, "a")] = y_a
            slot_positions[(rnd_num, mi, "b")] = y_b
            match_centers[(rnd_num, mi)] = mid

    # Helper to build a fake post dict for thumbnail loading
    def _make_post_dict(m, side):
        """Build a minimal post dict from a match row for a given side ('a' or 'b')."""
        pid = m[f"post_{side}_id"]
        if pid is None:
            return None
        return {
            "id": pid,
            "name": m[f"name_{side}"] or "TBD",
            "post_type": m[f"type_{side}"] or "image",
            "text_content": m[f"text_{side}"],
            "filename": m[f"file_{side}"],
        }

    # Draw each round
    champion_post = None
    for rnd_num in range(1, 5):
        rx = BRACKET_MARGIN_LEFT + (rnd_num - 1) * BRACKET_ROUND_SPACING
        rnd_matches = rounds.get(rnd_num, [])

        for m in rnd_matches:
            mi = m["match_index"]
            is_completed = m["status"] == "completed"
            winner_id = m["winner_id"]

            for side in ("a", "b"):
                key = (rnd_num, mi, side)
                if key not in slot_positions:
                    continue
                y = slot_positions[key]
                post_dict = _make_post_dict(m, side)
                pid = m[f"post_{side}_id"]
                name = m[f"name_{side}"] or "TBD"

                # Determine if this entrant is winner/loser/pending
                is_winner = is_completed and pid == winner_id
                is_loser = is_completed and pid is not None and pid != winner_id
                is_tbd = pid is None

                # Draw thumbnail
                thumb_y = y - BRACKET_THUMB // 2
                if is_tbd:
                    # Empty slot with dashed border
                    draw.rectangle(
                        [rx, thumb_y, rx + BRACKET_THUMB, thumb_y + BRACKET_THUMB],
                        outline=BRACKET_TBD_COLOR,
                    )
                    # Dashed effect: draw small dashes
                    for dy in range(0, BRACKET_THUMB, 8):
                        draw.line(
                            [(rx, thumb_y + dy), (rx, thumb_y + min(dy + 4, BRACKET_THUMB))],
                            fill=BRACKET_TBD_COLOR,
                        )
                elif post_dict:
                    try:
                        thumb = _get_thumbnail(post_dict, BRACKET_THUMB)
                        if is_loser:
                            thumb = _dim_thumbnail(thumb)
                        canvas.paste(thumb, (rx, thumb_y))
                    except Exception:
                        draw.rectangle(
                            [rx, thumb_y, rx + BRACKET_THUMB, thumb_y + BRACKET_THUMB],
                            fill=(80, 80, 80),
                        )

                # Draw name label
                label_x = rx + BRACKET_THUMB + 6
                truncated = _truncate_name(name)
                if is_tbd:
                    text_color = BRACKET_TBD_COLOR
                    truncated = "TBD"
                elif is_loser:
                    text_color = BRACKET_LOSE_COLOR
                elif is_winner:
                    text_color = BRACKET_WIN_COLOR
                else:
                    text_color = BRACKET_TEXT_COLOR

                draw.text(
                    (label_x, y - 7),
                    truncated,
                    font=font,
                    fill=text_color,
                )

                # Track champion
                if rnd_num == 4 and is_winner and post_dict:
                    champion_post = post_dict

            # Draw connector lines to next round
            if rnd_num < 4:
                ya = slot_positions.get((rnd_num, mi, "a"))
                yb = slot_positions.get((rnd_num, mi, "b"))
                if ya is not None and yb is not None:
                    # Horizontal lines from each slot to a merge point
                    merge_x = rx + BRACKET_ROUND_SPACING - 40
                    mid_y = (ya + yb) // 2

                    # Line from slot A
                    draw.line(
                        [(rx + BRACKET_THUMB + 80, ya), (merge_x, ya)],
                        fill=BRACKET_LINE_COLOR,
                        width=1,
                    )
                    # Line from slot B
                    draw.line(
                        [(rx + BRACKET_THUMB + 80, yb), (merge_x, yb)],
                        fill=BRACKET_LINE_COLOR,
                        width=1,
                    )
                    # Vertical connector
                    draw.line(
                        [(merge_x, ya), (merge_x, yb)],
                        fill=BRACKET_LINE_COLOR,
                        width=1,
                    )
                    # Horizontal line to next round
                    next_rx = BRACKET_MARGIN_LEFT + rnd_num * BRACKET_ROUND_SPACING
                    draw.line(
                        [(merge_x, mid_y), (next_rx - 4, mid_y)],
                        fill=BRACKET_LINE_COLOR,
                        width=1,
                    )

    # Draw champion section
    if champion_post:
        cy = match_centers.get((4, 0), BRACKET_CANVAS_H // 2)
        cx = BRACKET_MARGIN_LEFT + 4 * BRACKET_ROUND_SPACING

        # Draw connector from final to champion
        final_rx = BRACKET_MARGIN_LEFT + 3 * BRACKET_ROUND_SPACING
        final_label_end = final_rx + BRACKET_THUMB + 160
        draw.line(
            [(final_label_end, cy), (cx - 4, cy)],
            fill=BRACKET_LINE_COLOR,
            width=2,
        )

        # Champion name above thumbnail
        champ_name = _truncate_name(champion_post["name"], 25)
        name_bbox = draw.textbbox((0, 0), champ_name, font=font_bold)
        name_w = name_bbox[2] - name_bbox[0]
        # Center the name above the thumbnail
        name_x = cx + (BRACKET_CHAMPION_THUMB - name_w) // 2
        name_y = cy - BRACKET_CHAMPION_THUMB // 2 - 24
        draw.text(
            (name_x, name_y),
            champ_name,
            font=font_bold,
            fill=(255, 215, 0),  # Gold
        )

        # Large thumbnail
        thumb_y = cy - BRACKET_CHAMPION_THUMB // 2
        try:
            big_thumb = _get_thumbnail(champion_post, BRACKET_CHAMPION_THUMB)
            canvas.paste(big_thumb, (cx, thumb_y))
        except Exception:
            draw.rectangle(
                [cx, thumb_y, cx + BRACKET_CHAMPION_THUMB, thumb_y + BRACKET_CHAMPION_THUMB],
                fill=(80, 80, 80),
            )

    buf = io.BytesIO()
    canvas.save(buf, "PNG")
    buf.seek(0)
    return buf


# --- Tournament Logic ---


def _get_post_by_id(conn, post_id):
    """Fetch a post row by ID."""
    return conn.execute("SELECT * FROM elo_posts WHERE id = ?", (post_id,)).fetchone()


def _get_active_tournament(conn):
    """Return the active tournament row, or None."""
    return conn.execute("SELECT * FROM tournaments WHERE status = 'active' ORDER BY id DESC LIMIT 1").fetchone()


def _count_completed_regular(conn):
    """Count completed non-champions tournaments."""
    row = conn.execute("SELECT COUNT(*) FROM tournaments WHERE status = 'completed' AND is_champions = 0").fetchone()
    return row[0]


def _last_completed_tournament(conn):
    """Return the most recently completed tournament, or None."""
    return conn.execute("SELECT * FROM tournaments WHERE status = 'completed' ORDER BY id DESC LIMIT 1").fetchone()


def _should_run_champions(conn):
    """Check if it's time for a champions tournament."""
    completed_regular = _count_completed_regular(conn)
    if completed_regular == 0 or completed_regular % CHAMPIONS_INTERVAL != 0:
        return False
    # Don't run back-to-back champions
    last = _last_completed_tournament(conn)
    return not (last and last["is_champions"])


def _select_regular_entrants(conn):
    """Select 16 entrants for a regular tournament, prioritizing least-seen posts."""
    rows = conn.execute("SELECT id FROM elo_posts ORDER BY tournaments_played ASC, RANDOM() LIMIT 16").fetchall()
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
            "SELECT id FROM elo_posts WHERE tournament_wins > 0 " "ORDER BY tournament_wins DESC, RANDOM()"
        ).fetchall()
        for f in fillers:
            if f["id"] not in seen and len(entrants) < 16:
                seen.add(f["id"])
                entrants.append(f["id"])

    return entrants[:16]


def _create_tournament(conn, is_champions=False):
    """Create a new tournament with 16 entrants and 15 matches. Returns tournament id."""
    row = conn.execute("SELECT COALESCE(MAX(tournament_number), 0) FROM tournaments").fetchone()
    tournament_number = row[0] + 1

    if is_champions:
        entrant_ids = _select_champions_entrants(conn)
    else:
        entrant_ids = _select_regular_entrants(conn)

    if len(entrant_ids) < 16:
        _LOG.warning("Only %d posts available, need 16 for tournament", len(entrant_ids))
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
            row = conn.execute("SELECT tournament_wins FROM elo_posts WHERE id = ?", (pid,)).fetchone()
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
        label,
        tournament_number,
        tid,
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
            conn.execute("UPDATE tournament_matches SET status = 'pending' WHERE status = 'active'")
            conn.commit()
            conn.close()

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

        for attachment in message.attachments:
            if not attachment.content_type or not attachment.content_type.startswith("image/"):
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
                tournament = conn.execute("SELECT * FROM tournaments WHERE id = ?", (tid,)).fetchone()

                label = "CHAMPIONS TOURNAMENT" if tournament["is_champions"] else "Tournament"
                await channel.send(f"**{label} #{tournament['tournament_number']} has begun!**")

            match = _get_next_pending_match(conn, tournament["id"])
            if match is None:
                pending = conn.execute(
                    "SELECT COUNT(*) FROM tournament_matches " "WHERE tournament_id = ? AND status != 'completed'",
                    (tournament["id"],),
                ).fetchone()[0]
                if pending == 0:
                    _finalize_tournament(conn, tournament["id"])
                    await self._announce_tournament_complete(channel, tournament["id"], conn)
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

        label = "CHAMPIONS TOURNAMENT" if tournament["is_champions"] else "Tournament"
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
            msg_text = f"{header}\n" f"**A: {post_a['name']}**\n" f"**B: {post_b['name']}**"

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
            await channel.send(f"**{winner_post['name']}** wins! Advances to {next_round_name}.")
        else:
            await channel.send(f"**{winner_post['name']}** wins the tournament!")

    async def _announce_tournament_complete(self, channel, tournament_id, conn):
        """Announce the full tournament results."""
        tournament = conn.execute("SELECT * FROM tournaments WHERE id = ?", (tournament_id,)).fetchone()
        label = "CHAMPIONS TOURNAMENT" if tournament["is_champions"] else "Tournament"

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
            suffix = "st" if p == 1 else "nd" if p == 2 else "rd" if p == 3 else "th"
            lines.append(f"{p}{suffix}: {names}")

        await channel.send("\n".join(lines))

        # Send bracket image
        try:
            buf = _build_bracket_image(tournament_id)
            await channel.send(
                file=discord.File(fp=buf, filename="bracket.png"),
            )
        except Exception:
            _LOG.warning("Failed to build bracket image for completed tournament")

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

        total_tournaments = conn.execute("SELECT COUNT(*) FROM tournaments WHERE status = 'completed'").fetchone()[0]

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
            await ctx.send(f"**{total_posts} posts in pool** - No tournaments completed yet.")
            return

        lines = [f"**Tournament Leaderboard ({total_tournaments} tournaments completed, {total_posts} posts)**"]
        for i, row in enumerate(leaders, 1):
            avg = row["total_placement_points"] / row["tournaments_played"] if row["tournaments_played"] > 0 else 0
            lines.append(f"{i}. {row['name']} - {row['tournament_wins']} wins " f"(avg placement: {avg:.1f})")

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
                round_name = ROUND_NAMES.get(current_match["round"], f"Round {current_match['round']}")
                t_label = "Champions Tournament" if active["is_champions"] else "Tournament"
                lines.append(f"**Current: {t_label} #{active['tournament_number']} - {round_name}**")

        await ctx.send("\n".join(lines))

    @commands.command(name="bracket")
    async def bracket_cmd(self, ctx):
        """Show current tournament bracket as an image."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        tournament = _get_active_tournament(conn)
        if not tournament:
            tournament = _last_completed_tournament(conn)
            if not tournament:
                conn.close()
                await ctx.send("No tournaments yet.")
                return

        tid = tournament["id"]
        conn.close()

        try:
            buf = _build_bracket_image(tid)
            await ctx.send(
                file=discord.File(fp=buf, filename="bracket.png"),
            )
        except Exception:
            _LOG.exception("Failed to build bracket image, falling back to text")
            await self._bracket_text_fallback(ctx, tid)

    async def _bracket_text_fallback(self, ctx, tournament_id):
        """Text-only bracket fallback if image generation fails."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        tournament = conn.execute("SELECT * FROM tournaments WHERE id = ?", (tournament_id,)).fetchone()

        label = "CHAMPIONS TOURNAMENT" if tournament["is_champions"] else "Tournament"
        lines = [f"**{label} #{tournament['tournament_number']}** ({tournament['status']})"]

        matches = conn.execute(
            """SELECT tm.*, ea.name as name_a, eb.name as name_b
               FROM tournament_matches tm
               LEFT JOIN elo_posts ea ON tm.post_a_id = ea.id
               LEFT JOIN elo_posts eb ON tm.post_b_id = eb.id
               WHERE tm.tournament_id = ?
               ORDER BY tm.round ASC, tm.match_index ASC""",
            (tournament_id,),
        ).fetchall()

        conn.close()

        current_round = None
        for m in matches:
            if m["round"] != current_round:
                current_round = m["round"]
                round_name = ROUND_NAMES.get(current_round, f"Round {current_round}")
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
            await ctx.send(f"Cleared database. Backfilling images from {IMAGES_FOLDER}...")

            candidates = []
            for img_path in IMAGES_FOLDER.glob("*"):
                if img_path.suffix.lower() not in (
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".gif",
                ):
                    continue
                if img_path.name.startswith("temp_"):
                    continue
                if img_path.stem.isdigit():
                    continue
                candidates.append(img_path)

            await ctx.send(f"Found {len(candidates)} eligible images, ingesting in parallel...")

            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [loop.run_in_executor(executor, ingest_image, img_path, use_ocr) for img_path in candidates]
                results = await asyncio.gather(*futures)

            count = sum(1 for r in results if r is not None)
            await ctx.send(f"Backfill complete: {count} posts ingested")
        except Exception:
            _LOG.exception("Backfill failed")
            await ctx.send("Backfill failed - check logs")


async def setup(bot):
    await bot.add_cog(BestBotPosts(bot))
