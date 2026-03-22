"""Tamagotchi - A text-based virtual pet with an absurd number of interacting systems."""

from .cog import TamagotchiCog

__all__ = ["TamagotchiCog"]


async def setup(bot):
    await bot.add_cog(TamagotchiCog(bot))
