from discord.ext import commands

MAX_HELP_LEN = 1900  # keep a little margin below Discord 2000 limit


class General(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.command()
    async def ping(self, ctx: commands.Context):
        """Responds with Pong!"""
        await ctx.send("Pong!")

    @commands.command(name="help")
    async def help_cmd(self, ctx: commands.Context):
        """Show available features and commands (<=2000 chars)."""
        text = (
            "**Hollingsbot Help**\n"
            "Mention the bot to run commands anywhere (e.g., `@Bot help`).\n\n"
            "Image generation\n"
            "- `! prompt` quick image.\n"
            "- `$ prompt` higher quality; `$$ prompt` premium.\n"
            "- `^ prompt` SVG generator.\n"
            "- `edit: ...` reply to a message with an image (or attach one) to edit; the bot replies to your prompt message.\n"
            "- Tips: `{123}` sets seed; `<a, b, c>` expands to multiple prompts.\n"
            "- `!models` list available image generators.\n\n"
            "GIF from reply chain\n"
            "- Reply `gif` to any message to build a GIF from all images across the whole reply chain (root → leaf). Shows a thinking emoji while working.\n\n"
            "Chat with LLMs\n"
            "- Type normally; the bot replies with context and supports images.\n"
            "- `!models` list available chat models.\n"
            "- `!model <provider/model>` set your preferred model.\n"
            "- `!system` show; `!system <text>` set; `!system reset` clear your system prompt.\n"
            "- Long replies auto-split; large code blocks may be attached; SVG blocks are rendered.\n\n"
            "GPT‑2 channel\n"
            "- In the GPT‑2 channel, the bot replies to any message with a lightweight model.\n\n"
            "Admin\n"
            "- `!reset` restart the project containers (the bot may go offline briefly).\n\n"
            "Other\n"
            "- `ping` returns `Pong!`.\n"
            "- If a starboard is enabled, reacting to a bot message can repost it there.\n"
        )
        await ctx.send(text[:MAX_HELP_LEN])


async def setup(bot: commands.Bot):
    await bot.add_cog(General(bot))
