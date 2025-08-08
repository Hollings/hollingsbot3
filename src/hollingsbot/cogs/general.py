from discord.ext import commands

class General(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.command()
    async def ping(self, ctx: commands.Context):
        """Responds with Pong!"""
        await ctx.send('Pong!')

async def setup(bot: commands.Bot):
    await bot.add_cog(General(bot))
