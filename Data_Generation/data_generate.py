import ollama
import requests # requests.exceptions.ConnectionError 를 잡기 위함

# Ollama 서버 설정
ollama_host = "http://sg039:11434"
client = ollama.Client(host=ollama_host)  # 클라이언트 인스턴스 생성

try:
    response = requests.get(ollama_host)
    print("Server connected")
    print(response.text) # Should print "Ollama is running" or similar
    
except requests.ConnectionError:
    print("Not connected")
    exit() # Exit if not connected, as the rest will fail
    
# Corrected prompt assignment (removed trailing comma)
prompt = "이 prompt 한국어로 번역해줘. Prompt: Beginners BBQ Class Taking Place in Missoula!"

print(f"Sending prompt: {prompt}") # Good to print what you're sending

try:
    response = client.chat(
        model="deepseek-r1:671b",                
        messages=[{
            'role': 'user',
            'content': prompt, # Now 'prompt' is a string
        }]
    )

    # Corrected way to access the response content
    print("Full response:", response) # Optional: print the whole response to see its structure
    print("Assistant's message:", response['message']['content'])

except Exception as e:
    print(f"An error occurred during client.chat: {e}")
    # If you want to see the full traceback for other errors:
    # import traceback
    # traceback.print_exc()