from __future__ import annotations
import io, os, logging
from typing import List, Set
import discord
from discord.ext import commands
from PIL import Image

__all__ = ["GifReplyChainCog"]
_log = logging.getLogger(__name__)
THINKING = "\N{THINKING FACE}"

def _is_image_attachment(att: discord.Attachment) -> bool:
    ct = (att.content_type or "").lower()
    if ct.startswith("image/"): return True
    n = att.filename.lower()
    return n.endswith((".png",".jpg",".jpeg",".gif",".webp",".bmp",".tiff"))

class GifReplyChainCog(commands.Cog):
    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        env_ids = os.getenv("STABLE_DIFFUSION_CHANNEL_IDS", "")
        self._allowed_channel_ids: Set[int] = {int(x.strip()) for x in env_ids.split(",") if x.strip().isdigit()}
        self._max_frames = int(os.getenv("GIF_CHAIN_MAX_FRAMES","30"))
        self._max_side_px = int(os.getenv("GIF_CHAIN_MAX_SIDE","768"))
        self._frame_ms = int(os.getenv("GIF_CHAIN_FRAME_MS","700"))
        # Discord often enforces ~8 MiB per file on many servers; allow override
        self._max_bytes = int(os.getenv("GIF_CHAIN_MAX_BYTES", str(8 * 1024 * 1024)))
        self._scan_limit = int(os.getenv("GIF_CHAIN_SCAN_LIMIT","400"))

    def _in_allowed_channel(self, msg: discord.Message) -> bool:
        if msg.guild is None: return False
        if not self._allowed_channel_ids: return True
        return msg.channel.id in self._allowed_channel_ids

    async def _react(self, msg: discord.Message, emoji: str, *, remove: bool = False) -> None:
        try:
            if remove:
                if self.bot.user is not None:
                    await msg.remove_reaction(emoji, self.bot.user)
                else:
                    await msg.clear_reaction(emoji)
            else:
                await msg.add_reaction(emoji)
        except Exception:
            pass

    async def _find_direct_reply(self, base: discord.Message) -> discord.Message | None:
        """Find the earliest message that directly replies to `base`.

        Scans messages after `base` up to `_scan_limit` and returns the first
        whose `reference.message_id` equals `base.id`. Returns None if not found.
        """
        try:
            async for msg in base.channel.history(limit=self._scan_limit, after=base.created_at, oldest_first=True):  # type: ignore[attr-defined]
                ref = getattr(msg, "reference", None)
                if ref and getattr(ref, "message_id", None) == base.id:
                    return msg
        except Exception:
            pass
        return None

    async def _collect_chain_images(self, start: discord.Message) -> List[bytes]:
        # Backward to root (ancestors)
        ancestors, seen, cur = [], set(), start
        while cur and cur.id not in seen:
            seen.add(cur.id)
            ancestors.append(cur)
            ref = getattr(cur, "reference", None)
            if not ref:
                break
            res = getattr(ref, "resolved", None)
            if isinstance(res, discord.Message):
                cur = res
                continue
            mid = getattr(ref, "message_id", None)
            if mid:
                try:
                    cur = await start.channel.fetch_message(mid)
                except Exception:
                    break
            else:
                break
        ancestors.reverse()  # root -> start

        # Forward to leaf (descendants)
        descendants: list[discord.Message] = []
        cur = start
        while True:
            nxt = await self._find_direct_reply(cur)
            if not nxt or nxt.id in seen:
                break
            descendants.append(nxt)
            seen.add(nxt.id)
            cur = nxt

        chain = ancestors + descendants  # full chain root -> leaf

        imgs: List[bytes] = []
        for m in chain:
            for a in m.attachments:
                if len(imgs) >= self._max_frames:
                    break
                if _is_image_attachment(a):
                    try:
                        imgs.append(await a.read())
                    except Exception:
                        _log.debug("Could not download attachment %s on %s", a.id, m.id)
            if len(imgs) >= self._max_frames:
                break
        return imgs

    def _prepare_frames(self, blobs: List[bytes]) -> List[Image.Image]:
        dec = []
        for b in blobs:
            try:
                im = Image.open(io.BytesIO(b)).convert("RGBA"); dec.append(im)
            except Exception: pass
        if not dec: return []
        scaled, mw, mh = [], 1, 1
        for im in dec:
            w,h = im.size; s = 1.0
            if max(w,h) > self._max_side_px: s = self._max_side_px/float(max(w,h))
            nw,nh = max(1,int(round(w*s))), max(1,int(round(h*s)))
            sim = im.resize((nw,nh), Image.LANCZOS); scaled.append(sim)
            mw, mh = max(mw,nw), max(mh,nh)
        out = []
        for sim in scaled:
            c = Image.new("RGBA", (mw,mh), (0,0,0,255))
            x,y = (mw-sim.width)//2, (mh-sim.height)//2
            c.paste(sim,(x,y),sim)
            # Keep as RGB; let Pillow handle palette quantization during save
            out.append(c.convert("RGB"))
        return out

    def _resize_frames(self, frames: List[Image.Image], scale: float) -> List[Image.Image]:
        if scale >= 0.999:
            return frames
        w, h = frames[0].size
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        return [im.resize((nw, nh), Image.LANCZOS) for im in frames]

    def _encode_gif(self, frames: List[Image.Image], colors: int) -> bytes:
        # Quantize each frame to reduce size, and save without optimize for stability
        q = [im.convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT) for im in frames]
        buf = io.BytesIO()
        q[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=q[1:],
            duration=self._frame_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        buf.seek(0)
        return buf.getvalue()

    async def _build_gif_bytes(self, frames: List[Image.Image]) -> bytes:
        if not frames:
            return b""
        # Try multiple compression strategies to fit under size limit
        size_limit = self._max_bytes
        scales = [1.0, 0.85, 0.72, 0.6, 0.5, 0.4, 0.33]
        drops = [1, 2, 3]
        palettes = [128, 64, 32]

        base = frames
        for s in scales:
            cur_frames = self._resize_frames(base, s) if s < 0.999 else base
            for d in drops:
                seq = cur_frames[::d] if d > 1 else cur_frames
                if not seq:
                    seq = cur_frames[-1:]
                for colors in palettes:
                    try:
                        data = self._encode_gif(seq, colors)
                    except Exception:
                        continue
                    if len(data) <= size_limit:
                        return data
        # Last resort: single frame at smallest scale/colors
        try:
            smallest = self._resize_frames(base, scales[-1])
            sf = smallest[-1:]
            return self._encode_gif(sf, palettes[-1])
        except Exception:
            return b""

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild: return
        if not self._in_allowed_channel(message): return
        if (message.content or "").strip().lower() != "gif": return
        ref = message.reference
        if not ref: return
        tgt = None
        res = getattr(ref, "resolved", None)
        if isinstance(res, discord.Message): tgt = res
        else:
            mid = getattr(ref, "message_id", None)
            if mid:
                try: tgt = await message.channel.fetch_message(mid)
                except Exception: tgt = None
        if tgt is None: return
        await self._react(message, THINKING)
        try:
            blobs = await self._collect_chain_images(tgt)
            if not blobs:
                await message.reply("No images found in the reply chain.", mention_author=False); return
            frames = self._prepare_frames(blobs)
            if not frames:
                await message.reply("Could not decode images from the reply chain.", mention_author=False); return
            data = await self._build_gif_bytes(frames)
            if not data:
                await message.reply("Failed to build GIF.", mention_author=False); return
            await message.reply(file=discord.File(io.BytesIO(data), filename="reply_chain.gif"), mention_author=False)
        except Exception as exc:
            _log.exception("gif_chain failed: %s", exc)
            try: await message.reply("Error creating GIF from reply chain.", mention_author=False)
            except Exception: pass
        finally:
            await self._react(message, THINKING, remove=True)

async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(GifReplyChainCog(bot))
