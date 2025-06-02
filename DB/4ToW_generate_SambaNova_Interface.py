# -*- coding: utf-8 -*-
from huggingface_hub import InferenceClient
import os

class DeepSeekHFChat:
    def __init__(self, api_key=None):
        """
        DeepSeek 채팅 클래스 초기화
        
        Args:
            api_key (str): Hugging Face API 키. None이면 환경변수에서 읽기 시도
        """
        if api_key is None:
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if api_key is None:
                raise ValueError("API 키가 필요합니다. 파라미터로 전달하거나 HUGGINGFACE_API_KEY 환경변수를 설정하세요.")
        
        self.client = InferenceClient(
            provider="sambanova",
            api_key=api_key,
        )
        
        # 기본 생성 설정
        self.settings = {
            'temperature': 0.9,
            'max_tokens': 2048,
            'top_p': 0.9
        }
        
        # 고정된 프롬프트 템플릿 (JSON 이스케이프 처리)
        self.prompt_template = """Task Instruction: Given certain text, you need to predict the next word of it. Moreover, before your output, you could first give short thoughts about how you infer the next word based on the provided context.

Here are five examples for the task:

Example 0: 우리는 가끔 온라인 쿠폰과 기타 특별 혜택을 제공합니다. <hCoT> Customers can explore additional ways to find deals beyond online coupons, like subscribing. </hCoT> 또는 제품 연구에 참여하고 싶다면 '홈 제품 배치'를 체크하고 몇 가지 질문에 답해주세요. 무엇을 기다리고 있나요?

Example 1: 방정식 2x + 5 = 17을 풀어보세요. 먼저 양변에서 5를 <hCoT> The context presents an equation 2x + 5 = 17 and mentions subtracting 5 from both sides, so the next word should be '빼면' to describe the subtraction operation. </hCoT> 빼면 2x = 12가 됩니다. 그 다음 양변을 2로 <hCoT> The context shows 2x = 12 and mentions dividing both sides by 2, so the next word should be '나누면' to complete the division step. </hCoT> 나누면 x = 6이 답입니다.

Example 2: Unity에서 2D 객체를 드래그할 때 다른 객체와의 최소 거리는 1.5f입니다. 두 객체가 <hCoT> The context describes distance constraints for 2D objects in Unity, so the next word should be '연결되면' to describe what happens when objects connect. </hCoT> 연결되면 드래그가 더 제한됩니다.

Example 3: 대수학에서 대체는 문자를 숫자로 바꾸는 것입니다. 숫자와 <hCoT> The context explains algebraic substitution involving numbers, so the next word should be '문자' as algebra deals with both numbers and variables. </hCoT> 문자 사이에는 곱셈 기호가 숨겨져 있습니다.

Example 4: 랜달즈빌 이사 회사 Movers MAX 디렉토리는 <hCoT> The context introduces a moving company directory called Movers MAX, so the next word should be '이사' to specify what kind of resources this directory provides. </hCoT> 이사 자원을 위한 원스톱 소스입니다.

Now please give me your prediction for the thought and next word based on the following context:

{user_input_context}

Thought:
Next Word:"""

    def ask_deepseek(self, question):
        """
        DeepSeek 모델에게 질문하고 답변 받기
        
        Args:
            question (str): 사용자 질문
            
        Returns:
            str: DeepSeek의 답변
        """
        try:
            # 사용자 입력을 프롬프트 템플릿에 치환
            formatted_content = self.prompt_template.format(user_input_context=question)
            
            print("답변 생성 중...")
            
            completion = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1",
                messages=[
                    {
                        "role": "user",
                        "content": formatted_content
                    }
                ],
                temperature=self.settings['temperature'],
                max_tokens=self.settings['max_tokens'],
                top_p=self.settings['top_p']
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            return f"❌ 오류 발생: {str(e)}"

    def update_settings(self):
        """생성 설정 업데이트"""
        print(f"\n⚙️ 현재 설정:")
        print(f"  - max_tokens: {self.settings['max_tokens']}")
        print(f"  - temperature: {self.settings['temperature']}")
        print(f"  - top_p: {self.settings['top_p']}")
        
        try:
            new_max_tokens = input(f"새로운 max_tokens (현재: {self.settings['max_tokens']}): ").strip()
            if new_max_tokens:
                self.settings['max_tokens'] = int(new_max_tokens)
            
            new_temp = input(f"새로운 temperature (현재: {self.settings['temperature']}): ").strip()
            if new_temp:
                self.settings['temperature'] = float(new_temp)
            
            new_top_p = input(f"새로운 top_p (현재: {self.settings['top_p']}): ").strip()
            if new_top_p:
                self.settings['top_p'] = float(new_top_p)
            
            print("✅ 설정이 업데이트되었습니다!")
        except ValueError:
            print("❌ 잘못된 값입니다. 설정이 변경되지 않았습니다.")

def interactive_chat():
    """대화형 채팅 시스템"""
    print("=" * 60)
    print("🤖 DeepSeek 대화형 채팅 시스템 (Hugging Face Hub)")
    print("=" * 60)
    
    # API 키 입력 받기
    api_key = input("Hugging Face API 키를 입력하세요 (환경변수 HUGGINGFACE_API_KEY로도 설정 가능): ").strip()
    
    try:
        # DeepSeek 채팅 클래스 초기화
        if not api_key:
            chat = DeepSeekHFChat()  # 환경변수에서 API 키 읽기 시도
        else:
            chat = DeepSeekHFChat(api_key)
        
        print("\n✅ DeepSeek 연결 완료! 이제 질문해보세요.")
        
    except ValueError as e:
        print(f"❌ {e}")
        return
    except Exception as e:
        print(f"❌ 초기화 오류: {e}")
        return
    
    print("💡 명령어:")
    print("  - 'quit', 'exit', '종료' : 프로그램 종료")
    print("  - 'clear', '클리어' : 화면 정리")
    print("  - 'settings', '설정' : 생성 설정 변경")
    print("-" * 60)
    
    conversation_count = 0
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input(f"\n[{conversation_count + 1}] 당신: ").strip()
            
            # 종료 명령어 체크
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n👋 채팅을 종료합니다. 감사합니다!")
                break
            
            # 화면 정리 명령어
            elif user_input.lower() in ['clear', '클리어']:
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            
            # 설정 변경
            elif user_input.lower() in ['settings', '설정']:
                chat.update_settings()
                continue
            
            # 빈 입력 체크
            elif not user_input:
                print("질문을 입력해주세요.")
                continue
            
            # DeepSeek에게 질문
            print(f"\n🤖 DeepSeek 답변 생성 중...")
            
            answer = chat.ask_deepseek(user_input)
            
            print(f"\n🤖 DeepSeek: {answer}")
            conversation_count += 1
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Ctrl+C가 감지되었습니다.")
            user_choice = input("정말 종료하시겠습니까? (y/n): ").strip().lower()
            if user_choice in ['y', 'yes', '예']:
                break
            else:
                print("계속 진행합니다...")
                continue
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            print("다시 시도해주세요.")
            continue

def main():
    """메인 함수"""
    # 사용 예시
    print("🔧 사용 방법:")
    print("1. 직접 API 키 전달:")
    print("   chat = DeepSeekHFChat('your_api_key_here')")
    print("   answer = chat.ask_deepseek('질문 내용')")
    print("\n2. 환경변수 사용:")
    print("   export HUGGINGFACE_API_KEY='your_api_key_here'")
    print("   chat = DeepSeekHFChat()")
    print("\n3. 대화형 모드:")
    print("   python script.py")
    print("\n" + "="*60)
    
    interactive_chat()

if __name__ == "__main__":
    main()