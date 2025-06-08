import sys, types
sys.modules.setdefault('transformers', types.SimpleNamespace(pipeline=lambda *a, **k: lambda prompt, max_new_tokens=500: [{'generated_text': prompt + ' output'}]))

import pytest
import asyncio

from text_generators import HuggingFaceTextGenerator

@pytest.mark.asyncio
async def test_hf_generator():
    gen = HuggingFaceTextGenerator('model')
    result = await gen.generate('hello')
    assert 'hello' in result
