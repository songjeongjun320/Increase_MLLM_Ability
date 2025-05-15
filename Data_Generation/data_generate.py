import ollama
import requests # requests.exceptions.ConnectionError 를 잡기 위함

# Ollama 서버 설정
ollama_host = "http://sg023:11434"
client = ollama.Client(host=ollama_host)  # 클라이언트 인스턴스 생성

try:
    response = requests.get(ollama_host)
    print("Server connected")
    print(response.text)
    
except requests.ConnectionError:
    print("Not connected")
    
prompt = "Hello I am Jun"

response = client.chat(
    model="llama3.2:latest",                
    messages=[{
        'role': 'user',
        'content': prompt,
    }]
)

print(response['messages']['content'])