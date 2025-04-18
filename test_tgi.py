import requests

# TGI endpoint (default port 8080)
TGI_URL = "http://localhost:8080/generate"

# Prompt to test
prompt = "You are a helpful assistant. What is the capital of France?"

# Payload for TGI
payload = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.02
    }
}

def test_tgi():
    print("🔁 Sending prompt to TGI...")
    try:
        response = requests.post(TGI_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        generated = result.get("generated_text", "[No 'generated_text' in response]")
        print("\n🧠 TGI Response:")
        print(generated)

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Failed to reach TGI endpoint: {e}")
        if response is not None:
            print(f"Response content: {response.text}")

if __name__ == "__main__":
    test_tgi()
