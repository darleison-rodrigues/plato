import asyncio
import logging
from plato.ollama.client import OllamaClient

logging.basicConfig(level=logging.INFO)

async def test_client():
    async with OllamaClient() as ollama:
        print("Checking health...")
        health = await ollama.health_check()
        print(f"Ollama healthy: {health}")
        
        if health:
            print("\nListing models:")
            models = await ollama.list_models()
            for m in models[:3]:
                print(f"- {m.name} ({m.size / 1e9:.2f} GB)")
            
            print("\nTesting streaming (brief):")
            # We'll just take the first token and cancel
            try:
                async for chunk in ollama.generate_stream("qwen2.5-coder:1.5b", "Say hello in one word."):
                    if isinstance(chunk, str):
                        print(f"Token: {chunk}")
                        break
            except Exception as e:
                print(f"Stream test failed (expected if model missing): {e}")

if __name__ == "__main__":
    asyncio.run(test_client())
