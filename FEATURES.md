Goal: Extract each feature from Discord cogs into self-contained service classes with clean, Discord-agnostic interfaces for easy use from other code (e.g., LLM tool-calling).

> Feature Checklist (simplest → complex)
  
  - [ ] LLM help — src/hollingsbot/cogs/llm_chat.py::LLMAPIChat.help_command: !h shows chat help.
  - [ ] LLM list models — src/hollingsbot/cogs/llm_chat.py::LLMAPIChat.list_models: !models lists AVAILABLE_MODELS.
  - [ ] List image models — src/hollingsbot/cogs/image_gen_cog.py::ImageGenCog.on_message: !models shows prefixes/providers and scopes.
  - [ ] LLM set model — src/hollingsbot/cogs/llm_chat.py::LLMAPIChat.set_or_list_model: !model api/name sets per-user provider/model.
  - [ ] LLM system prompt — src/hollingsbot/cogs/llm_chat.py::LLMAPIChat.set_system_prompt: !system show/set/reset system prompt.
  - [ ] Starboard — src/hollingsbot/cogs/starboard.py::Starboard.on_reaction_add: reposts reacted bot messages to STARBOARD_CHANNEL_ID; skips ignored channels.
  - [ ] GPT-2 chat — src/hollingsbot/cogs/gpt2_chat.py::GPT2Chat.on_message: auto-replies in GPT2_CHANNEL_ID via Hugging Face.
  - [ ] Image generate — src/hollingsbot/cogs/image_gen_cog.py::ImageGenCog.on_message: prefix-based image gen; supports {seed}, <a,b,c>.
  - [ ] Image edit — src/hollingsbot/cogs/image_gen_cog.py::ImageGenCog.on_message: edit: edits attached/replied images; allowed in channels from STABLE_DIFFUSION_CHANNEL_IDS plus EDIT_CHANNEL_IDS.
  - [ ] Enhance reply — src/hollingsbot/cogs/enhance_cog.py::EnhanceCog.on_message: reply “enhance” to improve text (Anthropic) and attach image.
  - [ ] Experimental code-edit — src/hollingsbot/cogs/image_edit.py::ImageEditCog.handle_edit_request: reply to an image; gets Python edit_image(img) from OpenAI and runs it.
  - [ ] LLM auto-chat — src/hollingsbot/cogs/llm_chat.py::LLMAPIChat.on_message: auto-replies in LLM_WHITELIST_CHANNELS; ignores !, -, and edit:; handles images/text files; renders SVGs.
