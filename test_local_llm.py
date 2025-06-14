import os
import asyncio
import time
import psutil
from dotenv import load_dotenv
from categorize import LocalLLMProvider, MODEL_MEMORY_REQUIREMENTS

def print_system_memory():
    """Print current system memory status"""
    mem = psutil.virtual_memory()
    print("\nSystem Memory Status:")
    print(f"Total: {mem.total / (1024*1024*1024):.1f}GB")
    print(f"Available: {mem.available / (1024*1024*1024):.1f}GB")
    print(f"Used: {mem.used / (1024*1024*1024):.1f}GB")
    print(f"Percent: {mem.percent}%")

def print_model_options():
    """Print available models and their memory requirements"""
    print("\nAvailable Models and Memory Requirements:")
    for model, memory in MODEL_MEMORY_REQUIREMENTS.items():
        print(f"- {model}: {memory}GB")

async def test_local_llm():
    print("Testing Ollama connection...")
    print_system_memory()
    print_model_options()
    
    try:
        # Try to use the default model (mistral:7b-instruct-v0.2)
        llm = LocalLLMProvider()
        
        # Test prompt
        test_prompt = """Analyze this shoe description and extract structured information:
        "Vintage leather boot, size 9.5 US, made in Italy, 1980s. Brown color with brass buckles.
        Part of the Heritage Collection, item #H-123. Excellent condition, displayed in main hall."
        
        Return a JSON object with standard categories."""

        print("\nTesting model with sample prompt...")
        start_time = time.time()
        response = await llm.generate(test_prompt)
        end_time = time.time()
        
        print("\nModel response:")
        print(response)
        
        print(f"\nResponse time: {end_time - start_time:.2f} seconds")
        print("\nTest completed successfully!")
        
    except MemoryError as e:
        print(f"\nMemory Error: {e}")
        print("\nSuggestions:")
        print("1. Close other memory-intensive applications")
        print("2. Try a smaller model like 'mistral:7b-instruct-v0.2'")
        print("3. If using a large model, consider switching to a smaller one")
    except Exception as e:
        print(f"\nError testing Ollama: {str(e)}")
        print("\nMake sure Ollama is running with:")
        print("1. 'ollama serve' in one terminal")
        print("2. 'ollama pull mistral:7b-instruct-v0.2' to download the model")

if __name__ == "__main__":
    asyncio.run(test_local_llm()) 