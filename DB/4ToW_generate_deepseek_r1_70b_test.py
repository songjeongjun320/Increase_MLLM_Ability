# -*- coding: utf-8 -*-
import torch
import gc
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# 환경 변수 설정 (성능 최적화)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 오프라인 모드 강제

# 멀티 GPU 설정 - A100 80GB x2 최적화
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # GPU 0, 1 사용

# A100 80GB 최적화를 위한 추가 환경 변수
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048,expandable_segments:True,roundup_power2_divisions:8"
os.environ["NCCL_DEBUG"] = "WARN"  # 멀티 GPU 통신 최적화

# bitsandbytes 가져오기 시도
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
    print("✅ BitsAndBytesConfig successfully imported")
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    print("⚠️ BitsAndBytesConfig import failed - using FP16 quantization")

# FlashAttention 사용 가능 여부 확인
try:
    import flash_attn
    FLASH_ATTENTION_AVAILABLE = True
    print("✅ FlashAttention available for maximum performance")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("⚠️ FlashAttention not available - using optimized attention")

def setup_torch_optimizations():
    """PyTorch 최적화 설정 - A100 80GB 특화"""
    # 메모리 관리 최적화
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.enable_flash_sdp(True)  # Scaled Dot Product Attention 최적화
    
    # CUDA 캐시 최적화 - A100 80GB 특화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # A100 80GB를 위한 더 큰 메모리 풀 설정
        torch.cuda.set_per_process_memory_fraction(0.95)  # 76GB 사용 가능

def clear_memory():
    """효율적인 메모리 정리 함수 (A100 80GB x2)"""
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()  # IPC 메모리 정리

def get_multi_gpu_device_map(num_gpus: int = 2):
    """A100 80GB x2 최적 디바이스 맵 생성"""
    if num_gpus == 2:
        # A100 80GB x2 최적 분배 - 70GB 활용
        device_map = {
            "model.embed_tokens": 0,
            "model.norm": 1,
            "lm_head": 1,
        }
        
        # 레이어를 두 GPU에 균등 분배 (70B 모델 기준)
        num_layers = 80  # DeepSeek-R1 Distill Llama 70B 레이어 수
        layers_per_gpu = num_layers // 2
        
        for i in range(num_layers):
            if i < layers_per_gpu:
                device_map[f"model.layers.{i}"] = 0
            else:
                device_map[f"model.layers.{i}"] = 1
        
        return device_map
    else:
        return "auto"

class OptimizedDeepSeekChat:
    def __init__(self, model_path: str, num_gpus: int = 2):
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.model = None
        self.tokenizer = None
        self.device = None
        self.generation_config = None
        
        # 고성능 캐시 시스템
        self.cached_tokens = {}
        self.kv_cache = None
        
        # 커스텀 프롬프트 템플릿 정의
        self.prompt_template = """
Task Instruction: Given certain text, you need to predict the next word of it. Moreover, before your output, you could first give short thoughts about how you infer the next word based on the provided context.\n
Here are five examples for the task:\n

Example 0: 위성 하나, 행성 둘 그리고 움직임 2013년 8월 5일 Rapidrain 저는 두 개의 큰 행성 근처에서 위성의 비행을 보여주는 프로그램을 작성하려고 합니다. 이 모든 것에서 위성의 질량은 무시할 수 있습니다. 저는 행성1로부터의 위치 에너지 = pe1을 가지고 있고, 그리고 <hCoT> The context involves celestial mechanics, likely leading to potential energy from the second planet. </hCoT>행성2로부터의 위치 에너지 = pe2를 가지고 있으며, 그리고 <hCoT> unpredictable </hCoT>위성의 <hCoT> The context involves simulating a satellite's energy interactions with two planets, focusing on movement. </hCoT>운동 에너지 = ke를 가지고 있습니다. 두 행성의 가속도 벡터 합을 사용하여 !! 단일 !! 가속도 벡터를 만들고, 현재 위치, 속도 벡터 및 !! 단일 !! 가속도 벡터로 인한 움직임을 사용하여 다음 위치를 계산할 수 있습니다. 이것은 좋습니다; (단일 행성 및 위성 모델에서 잘 작동합니다). 새로운 속도 벡터도 유사하게 원래 속도 벡터에 가속도 벡터에서 유도된 속도를 더하여 추론할 수 있습니다. 이것 또한 좋습니다; (단일 행성 및 위성 모델에서도 잘 작동합니다). 그러나 총 에너지가 약간 벗어납니다. 제 모델을 짧은 시간 동안 사용하면 총 에너지가 6.5 * 10**-4만큼 감소합니다. 정말 큰 숫자는 아니지만, 이것을 <hCoT> The context addresses energy conservation in a satellite simulation near two planets, aiming for improvement. </hCoT>0.0으로 줄이는 방법을 찾고 싶습니다. 총 에너지 변화(TE) = 0.0에 도달하기 위해 모델을 조정할 세 가지 가능성이 있습니다:<hCoT> Three methods exist to adjust the model and eliminate the total energy decrease. </hCoT> <hCoT> The context involves adjusting a physics model to achieve zero change in total energy. </hCoT>1. 속도만 증가시켜 운동 에너지를 높입니다. 두 행성으로부터의 거리만 증가시켜 위치 에너지를 높입니다. 속도와 거리 모두 (특정 비율로) 증가시켜 운동 에너지(KE)와 위치 에너지(PE)를 모두 높입니다. 물리학, 자연, 수학 또는 논리가 이 세 가지 경로 중 어느 것을 탐색해야 하는지 정의합니까? 2013년 8월 5일 voko 이것은 오일러의 삼체 문제로 알려져 있습니다. 그것을 찾아보고 정말로 당신이 하고 있는 일을 해야 하나요? 설명해주세요. 2013년 8월 5일 Rapidrain 죄송합니다, voko님. 하지만 "loop that up"이 무슨 뜻인지 이해하지 못하겠습니다. 그리고 정말로 제가 하고 있는 일을 해야 하나요? 설명해주세요. 2013년 8월 5일 voko 오일러의 삼체 문제에 대한 정보를 찾아보세요. 위키피디아에 해당 페이지가 있습니다. 영어가 모국어가 아니라면, 모국어로 정보를 검색해볼 수 있습니다. 2013년 8월 5일 Rapidrain 다시 한번 Voko님, 'loop that up'이 무슨 뜻인가요? 이것이 오일러의 삼체 문제를 푸는 방법을 지칭하는 것인가요? 2013년 8월 5일 voko "Look that up" = "그 정보를 찾아보라"는 뜻입니다. 바퀴를 다시 발명하지 마세요. 2013년 8월 5일<hCoT> The dialogue shows voko clarifying "look that up" about Euler's three body problem. </hCoT> <hCoT> unpredictable </hCoT>두 개의 고정된 중심점 문제"라고도 알려져 있습니다. 하지만 그것이 Rapidrain님의 문제의 원인은 아닙니다. 문제는 위치와 속도가 어떻게 업데이트되는지에 있습니다. 다음은 상미분 방정식(ODE)을 풀기 위한 수치 해석 기법에 대한 매우 간략한 설명입니다. <hCoT> unpredictable </hCoT>우선, <hCoT> unpredictable </hCoT>Rapidrain님, 당신은 2차 초기값 문제라고 불리는 것을 풀려고 하고 있습니다. 2차라는 것은 1차(속도) 및 2차(가속도) 도함수가 있다는 의미이고, 초기값이라는 것은 시작 시간에 위치와 속도를 알고 있고 어떤 종료 시간에 그것들을 찾고 싶다는 의미입니다. 1차 ODE 기법 1차 <hCoT> The context discusses numerical techniques for solving first-order ODEs, particularly Euler's method. </hCoT>초기값 문제를 풀기 위한 많은 기법이 존재합니다. 이 2차 ODE를 1차 ODE로 변환하여 이러한 기법들을 활용할 수 있습니다. 모든 2차 ODE는 0차 및 1차 도함수로 구성된 두 배 크기의 상태 벡터를 만들어 1차 ODE로 다시 표현할 수 있습니다. 예를 들어, dotx(t) = v(t), ddotx(t) = a(t)는 u(t) = (x(t), v(t)), dotu(t) = (v(t), a(t))가 됩니다. 가장 간단한 1차 ODE 해결책은 오일러 방법입니다: u(t + Deltat) = u(t) + Deltat, dotu(t) 오일러 방법을 절대 사용해서는 안 됩니다. 그러나 다른 거의 모든 적분 기법이 더 똑똑한 오일러 유형의 단계를 수행하는 것으로 볼 수 있기 때문에 작동 방식을 이해하는 것이 중요합니다. 2차 ODE의 경우 오일러 방법은 다음과 같습니다. \begin{aligned} \vec x(t+\Delta t) &= \vec x(t) + \Delta t , \vec v(t) \ \vec v(t+\Delta t) &= \vec v(t) + \Delta t , \vec a(t) \end{aligned} 오일러 방법보다 훨씬 우수한 1차 ODE 해결책이 많이 있습니다. 룽게-쿠타 적분기는 t와 t+Δt 사이의 여러 중간 단계를 거쳐 u(t+Δt)에 대한 추정치에 도달합니다. 예측자/수정자 방법은 이전 값의 기록을 유지하여 한 알고리즘을 사용하여 u(t+Δt)를 예측하고 다른 알고리즘을 사용하여 수정할 수 있도록 합니다. 자세한 내용은 룽게-쿠타, 다단계 방법, 예측자-수정자를 구글에서 검색해보세요. 2차 ODE 기법 다른 접근 방식은 이것이 풀려는 2차 문제라는 사실을 활용하는 것입니다. 2차 ODE에 대한 오일러 방법의 등가물은 다음을 통해 단계를 수행하는 것입니다. \begin{aligned} \vec v(t+\Delta t) &= \vec v(t) + \Delta t , \vec a(t) \ \vec x(t+\Delta t) &= \vec x(t) + \Delta t , \vec v(t+\Delta t) \end{aligned} 이것은 오일러-크로머 방법, <hCoT> The paragraph discusses numerical methods for second order ODEs, ending with the Euler-Cromer method. </hCoT>심플렉틱 <hCoT> The paragraph discusses numerical methods for second order ODEs, ending with the Euler-Cromer method. </hCoT>오일러 방법 및 기타 여러 이름으로 불립니다. 이 접근 방식과 기본 오일러 방법의 유일한 차이점은 위치와 속도가 업데이트되는 순서입니다. 단순히 속도를 먼저 업데이트하도록 전환하는 것만으로도 엄청난 차이가 발생합니다. 기본 오일러 방법은 에너지 보존에 전혀 근접하지 못합니다. 이 접근 방식은 그렇게 합니다. 그러나 오일러-크로머는 여전히 형편없습니다. 이 접근 방식에 대한 간단한 수정은 위치 및 속도 계산을 시간 단계의 절반만큼 오프셋하는 것입니다. 이것이 도약(leapfrog), 위치 벌렛(position verlet) 및 속도 벌렛(velocity verlet) 적분이 수행하는 작업입니다. 자세한 내용은 이 이름들을 구글에서 검색해보세요. 더욱 발전된 것은 가우스-잭슨 기법입니다. 위치 벌렛의 변형을 시도해 보시는 것이 좋겠습니다. t=0에서 가속도 벡터를 계산하여 이를 부트스트랩해야 합니다. \begin{aligned} \vec x(t+\Delta t/2 ) &= \vec x(t) + \frac 1 2 \Delta t , \vec v(t) \ \vec v(t+\Delta t/2 ) &= \vec v(t) + \frac 1 2 \Delta t , \vec a \ & \text{중간점 가속도 계산 및 저장},\vec a = f(\vec x(t+\Delta t/2 )) \ \vec v(t+\Delta t) &= \vec v(t+\Delta t/2 ) + \frac 1 2 \Delta t , \vec a \ \vec x(t+\Delta t) &= \vec x(t+\Delta t/2 ) + \frac 1 2 \Delta t , \vec v(t+\Delta t) \end{aligned} 이것은 계산적으로 오일러-크로머보다 비용이 더 들지 않지만(일반적으로 비용은 도함수 계산에 있음) 훨씬 더 정확합니다. 2013년 8월 5일 voko 확실히 아시겠지만, 이 문제에서 ODE를 푸는 것은 전혀 불필요할 수 있습니다. 그러면 문제가 완전히 제거될 것입니다. 이것이 제가 Rapidrain님에게 고전적인 접근 방식을 연구하도록 촉구하는 핵심입니다. 2013년 8월 5일 Rapidrain 아주 좋습니다, DH님. "가서 찾아봐"보다 훨씬 도움이 됩니다. 그런데 질문이 있습니다: 당신의 방정식은 x(t + delt) = x(t) + deltv(t)를 보여줍니다. 우변에도 가속도에 의해 이동한 거리가 포함되어야 하지 않나요: x(t) + deltv(t) + (1/2)acc(t)(delt)**2 ?? 당신의 알고리즘을 시도해보겠습니다.\n

Example 1: 수학 도움 - 방정식이 일차 방정식인지 판별해주세요..도와주세요! 방정식이 일차 방정식인지 판별해주세요..도와주세요!<hCoT> Requesting help to identify if an equation is linear based on its characteristics. </hCoT> <hCoT> unpredictable </hCoT>1. fracx2= <hCoT> Determine if the equation is linear; it appears to continue with a simple number. </hCoT> 10+ frac2y3 7n−8m=4−2m <hCoT> Check if equations fit the linear form a x + b y = c; both are linear. </hCoT> <hCoT> unpredictable </hCoT> <hCoT> unpredictable </hCoT>원래 <hCoT> Determine if each equation is linear, as they involve first power variables only. </hCoT> Phresh님이 게시함 fracx2=10+ frac2y3 y= frac32( fracx2−10) y= frac32( fracx2−10) 그리고 7n−8m=4−2m 7n−8m=4−2m 6m=4−7n 6m=4−7n m= frac4−7n6 m= frac4−7n6 <hCoT> The equations are linear if each variable is to the first power and not multiplied. </hCoT> 이것도 직선입니다.\n

Example 2: 홈메이드 맥앤치즈를 훨씬 더 맛있게 만드는 것이 가능할까요? 저희는 그렇다고 생각합니다! 저희의 두 번째 <hCoT> unpredictable </hCoT>추천 <hCoT> The context discusses enhancing homemade Mac-n-Cheese, likely introducing an ingredient or recipe next. </hCoT>이달의 <hCoT> The context introduces a new recipe, likely part of a monthly series. </hCoT>레시피는 저희의 새로운 절인 아티초크 하트를 즐기면서 <hCoT> The recipe enhances Mac-n-Cheese with Marinated Artichoke Hearts, promoting a delicious addition. </hCoT>맛있는 지중해 풍미를 <hCoT> unpredictable </hCoT>클래식한 <hCoT> The recipe adds a Mediterranean twist to classic Mac-n-Cheese comfort food. </hCoT>위안 음식에 더하는 훌륭한 방법입니다! 부활절 일요일이나 주중 어느 날 저녁에도 완벽한 사이드 디쉬가 됩니다! 말할 것도 없이, 이건 O.M.G. 급으로 맛있습니다! <hCoT> The passage introduces a Mac-n-Cheese recipe, likely leading to ingredients or cooking steps. </hCoT>중간 <hCoT> The context describes a Mediterranean Mac-n-Cheese recipe that likely requires a cooking vessel. </hCoT>크기의 소스 팬에, 중 <hCoT> unpredictable </hCoT>강 <hCoT> The context involves cooking Mac-n-Cheese with a Mediterranean twist, likely leading to "heat." </hCoT>불로 버터와 밀가루를 넣고, 혼합물이 거품을 내는 동안 2-3분간 저어주세요. 우유를 천천히 저으면서 완전히 섞일 때까지 넣어주세요. 혼합물을 약 7분간 저으면서 걸쭉해지고 거품이 날 때까지 조리하세요. 불을 끄고; 마늘 스프레드, 각각 1컵의 <hCoT> The passage details a revised Mac-n-Cheese recipe, likely calling for cheese next. </hCoT>치즈를 넣고 저어주세요. <hCoT> The recipe enhances Mac-n-Cheese, suggesting to add cheese and seasonings next. </hCoT>소금과 후추로 간을 맞추세요. 익힌 마카로니 위에 붓고, 시금치, 아티초크, 남은 다진 치즈를 넣고 저어주세요. 베이킹 그릇에 담고 위에 빵가루 토핑을 뿌려주세요. 빵가루가 황금빛 갈색이 될 때까지 몇 분간 브로일러 아래에 두세요.\n

Example 3: 남편과 제가 집에서 꽤 광범위한 네트워크를 구축했다는 것은 비밀이 아닙니다. 제가 NT4 <hCoT> The speaker’s home network journey began with studying NT4 for IT certifications like MCSE. </hCoT>MCSE를 목표로 공부하던 아주 오래 전부터 시작되었고, 수년에 걸쳐 새로운 제품이 출시됨에 따라 저희는 학습을 더욱 발전시키기 위해 해당 제품들을 네트워크에 추가했습니다. 어제 저희는 도메인 컨트롤러를 초기화하고 <hCoT> unpredictable </hCoT>2008에서 새로 시작했습니다. 역할 추가 마법사를 사용하는 것이 <hCoT> unpredictable </hCoT>약간 <hCoT> The context discusses setting up Windows Server, implying a potentially confusing process with "Add Roles." </hCoT>혼란스러워서, 더 익숙한 dcpromo로 돌아갔는데, 이것이 훨씬 더 이해하기 쉬웠고 2003과 크게 <hCoT> The context shows familiarity with network systems, favoring dcpromo as simple and not overwhelming. </hCoT>다르지 않다고 느껴졌습니다. 물론, AD 역할은 이제 확장되었고 반짝이는 새것이므로 마법사 진행 중에 주의를 기울여야 합니다. 그냥 다음, 다음, 완료를 클릭하지 마세요. 물론, 저희 Hyper-V 머신도 2008을 실행하고 있지만, 저는 그 설치와 <hCoT> unpredictable </hCoT>거의 관련이 없었습니다 – 남편이 어느 날 밤 잠 못 이루던 깊은 밤에 해치웠거든요. 처음에는 DNS 설정에 몇 가지 문제가 있었습니다. 역방향 조회 영역이 생성되지 않았고, 제가 수정해야 할 다른 몇 가지 사항도 있었습니다. 자체 테스트가 계속 실패해서 DNS 설치가 100% 완벽하다고 아직 확신하지 못해 약간 걱정되지만, 지금은 네트워크가 작동하고 있으니 당장은 너무 많이 건드리지 않을 것입니다 (즉, 나중에 수정할 것입니다). 저희는 또한 SQL 통합 작업을 진행해 왔고, SQL2008 백엔드를 사용하여 ASP.net으로 인트라넷을 다시 작성하려고 시도할 것입니다. 몇 년 동안 이렇게 하겠다고 벼르고 있었는데, 이제 그 때가 온 것 같습니다. 저희가 새로 시작하기로 결정한 이유 중 하나는 <hCoT> The context discusses system setups and installations, suggesting the next word relates to services on an "old" system. </hCoT>이전 도메인에 다양한 서비스를 설치하면서 스키마를 약간 엉망으로 만들었기 때문입니다. 특히 저희가 제대로 정리하지 않았기 때문이죠 – 애플리케이션을 제대로 <hCoT> The new setup was complicated by leftover configurations and uncleaned remnants from the old domain. </hCoT>제거하지 않고 머신을 재설치하는 그런 종류의 일들 말입니다. 여기서 큰 원흉 중 하나는 LCS였습니다. 물론, 저희가 이런 실수를 하는 것은 이것이 가정 환경이기 때문이고, 그래서 9 시그마를 달성하는 것이 중요하지는 않습니다. 하지만 저희는 또한 언젠가 실제 기업 환경에 적용할 수 있는 몇 가지 <hCoT> unpredictable </hCoT>좋은 <hCoT> The context reflects learning from past mistakes in managing a home IT setup, suggesting insights. </hCoT>교훈을 배웠습니다. 그리고 집에서는 100% 가동 시간이 중요하지 않지만, 저희는 가능한 한 <hCoT> The context highlights a relaxed approach to uptime in a home network setup. </hCoT>오래 가동 상태를 유지하려고 노력합니다. 특히 저희는 Exchange를 통해 모든 가족 외출 일정을 잡고 인트라넷 웹을 통해 예산과 쇼핑 목록을 추적하는 등, 집안일을 유지하기 위해 실제로 이러한 서비스 중 일부를 사용하기 때문입니다. 그리고 인터넷 연결은 가능한 한 계속 연결되어 있어야 합니다. 왜냐하면 저는 중독자이고 우리 딸은 숙제에 필요하기 때문입니다.\n

Example 4: 로니 데일라는 함덴 구장의 울퉁불퉁하고 갈아엎어진 경기장 상태 때문에 자신의 팀이 레인저스에게 리그컵에서 굴욕을 안기지 못했다고 비난했습니다. 휴식 후 <hCoT> unpredictable </hCoT>데일라는 <hCoT> unpredictable </hCoT>SPFL에 끔찍한 경기장 상태가 충분하지 않다고 경고하며, 그의 팀이 패스 축구를 하려는 노력을 망쳤다고 말했습니다. 하프타임에 선수들에게 리드를 굳히라고 지시했는지 묻자 노르웨이 출신 감독은 이렇게 주장했습니다: '저는 그렇게 말하지 않았습니다 <hCoT> Deila responds to whether he instructed players to consolidate their halftime lead. </hCoT>– 저는 3골을 넣으라고 말했습니다. '하지만 우리는 정말로 경기를 끝내버리고 싶었습니다.<hCoT> Ronny Deila explained his strategy to decisively win the match, aiming to "kill the game off." </hCoT> '하지만 우리는 다른 방식으로 경기를 끝냈습니다 – 우리는 수비가 견고했고 상대방을 골문에서 멀리 떨어뜨려 놓았습니다. '우리는 더 공격하고 싶었지만, 변명을 하자면 우리는 패스 위주의 팀인데 그 경기장에서는 공을 패스할 기회가 전혀 없었습니다.<hCoT> Deila discusses how the poor pitch hindered his team's ability to play and attack effectively. </hCoT> <hCoT> unpredictable </hCoT>커먼웰스 게임 이후 다시 깔린 국립 경기장의 표면은 토요일 다른 준결승전에서 던디 유나이티드가 애버딘을 상대로 승리하는 동안 심하게 망가졌습니다. 함덴 plc와 SPFL에 3월 15일 시즌 첫 주요 결승전 전에 경기장 표면이 적절히 수리되도록 촉구하며 데일라는 덧붙였습니다: '스코틀랜드 축구를 발전시키려면 축구를 할 수 있는 경기장이 필요합니다. '만약 전국적으로 4~5개월 동안 형편없는 경기장에서 경기를 해야 한다면 모든 경기는 공중볼 다툼이 될 것입니다. '챔피언스 리그에 대해 이야기한다면, 근처에도 못 갑니다. '이곳은 국가대표팀 경기장입니다 – 훨씬 더 좋아야 합니다. 전반전 팀의 경기력에는 만족했지만 후반전에는 그렇지 못했던 데일라는 올드펌 더비를 처음 경험한 것을 만끽했습니다. '이보다 <hCoT> Football pitches impact game quality; conditions must improve; “better” emphasizes this need. </hCoT>좋을 순 없습니다. 아주 좋은 날이었습니다. '경기장 분위기는 믿을 수 없을 정도였습니다. 셀틱 감독은 이제 오늘 던디 유나이티드의 스튜어트 암스트롱에 대한 증액된 제안과 함께 이적 보강에 관심을 돌릴 것입니다. 금요일 150만 파운드의 제안이 거절되었고, 그 미드필더는 이적할 경우 컵 타이 규정으로 리그컵 결승전에 출전하지 못하게 됩니다. <hCoT> unpredictable </hCoT>자세한 설명을 <hCoT> unpredictable </hCoT>피하며 데일라는 덧붙였습니다: '우리는 내일 선수들을 추가하고 모두를 지키고 싶습니다. 우리는 지금 중요한 시기에 있습니다. 화요일이면 답을 알게 될 것이고, 그것은 당신과 나 모두에게 좋을 것입니다. 셀틱은 또한 여름에 태너다이스 윙어 게리 맥케이-스티븐과 사전 계약을 확보했으며, 지금 그를 확보하기 위해 25만 파운드를 지불할지 결정해야 합니다. 볼턴 원더러스도 오늘 셀틱의 골잡이 크리스 커먼스 영입을 위한 공식 제안을 할 수 있지만, 데일라는 31세의 이 선수가 잔류할 것이라고 주장했습니다. 셀틱은 또한 대한민국 자유계약선수 김보경 영입을 고려하고 있습니다.\n

Now please give me your prediction for the Thought and Next Word based on the following context:\n

context: {이 대통령은 우선 김민석 국무총리 후보자에 대해 "4선 의원이자 민주당의 수석최고위원으로 국정 전반에 대한 통찰력이 매우 깊은 분"이라며 "당과 국회에서 정책과 전략을 이끌고 국민의 }\n

{\n
\t"Thought":""\n
\t"Next Word":""\n
}\n
"""

# One satellite, two planets and movement\n\n1. Aug 5, 2013\n\nRapidrain\n\nI am trying to write a program to show the flight of a satellite in the neighbourhood of two large planets. In all of this the mass of the satellite is negligible.\n\n I have the potential energy from planet1 = pe1 and\nthe <hCoT> The context involves celestial mechanics, likely leading to potential energy from the second planet. </hCoT>potential energy from planet2 = pe2 and\nthe <hCoT> unpredictable </hCoT>kinetic <hCoT> The context involves simulating a satellite's energy interactions with two planets, focusing on movement. </hCoT>energy of the satellite = ke\n\nUsing the sum of the two planets' acc vectors to create a !! single !! acc vector I can calculate the next position using the current position, the velocity vector and the movement caused by the !! single !! acc vector.\n\n This is good; (it works fine in a single planet and satellite model).\n\n The new velocity vector can also be similarly deduced adding the induced velocity from the acc vector to the original velocity vector.\n\n This is also good; (it also works fine in a single planet and satellite model).\n\n However Total Energy is just a bit off. Using my model with a short sliver of time I have a decrease of total energy by a factor 6.5 * 10**-4. Not a really big number but I want to find how I can reduce it to <hCoT> The context addresses energy conservation in a satellite simulation near two planets, aiming for improvement. </hCoT>0.0.\n\n I have three possibilities of tweaking the model to reach change in TE = 0.0 :<hCoT> Three methods exist to adjust the model and eliminate the total energy decrease. </hCoT>\n\n <hCoT> The context involves adjusting a physics model to achieve zero change in total energy. </hCoT>1. only increase the velocity and thereby the kinetic energy\n\n2. only increase the distance from the two planets and thereby the potential energy\n\n3. increase both vel and dist (in a certain proportion) to increase both KE and PE\n\nDoes physics, nature, mathematics or logic define which of these three paths to explore?\n\n2. Aug 5, 2013\n\nvoko\n\nThis is known as Euler's three body problem. I suggest you loop that up and think whether you really need to do what you are doing.\n\n 3. Aug 5, 2013\n\nRapidrain\n\nSorry voko, but I don't understand what you mean by \"loop that up\".\n\n And really need to do what I am doing? Please explain.\n\n 4. Aug 5, 2013\n\nvoko\n\nFind the information on Euler's three body problem. Wikipedia has a page on that. If English is not your native language, you may want to search for the information in your language.\n\n 5. Aug 5, 2013\n\nRapidrain\n\nAgain Voko, what do you mean by 'loop that up'? Is this the designation of how one solves Euler's three bodies?\n\n6. Aug 5, 2013\n\nvoko\n\n\"Look that up\" = \"find that information\". Do not re-invent the wheel.\n\n 7. Aug 5, 2013<hCoT> The dialogue shows voko clarifying \"look that up\" about Euler's three body problem. </hCoT>\n\n<hCoT> unpredictable </hCoT>D H\n\nStaff Emeritus\nAlso known as <hCoT> The context discusses Euler's three body problem and clarifying \"look that up.\" </hCoT>\"the <hCoT> \"Also known as typically introduces an alternative term or concept related to the topic.\" </hCoT>problem of two fixed centers\".\n\n That, however, is not the cause Rapidrain's problem. The issue is how position and velocity are being updated. What follows is a very brief tutorial in numerical techniques to solve an ordinary differential equation (ODE).\n\n First <hCoT> unpredictable </hCoT>off, <hCoT> unpredictable </hCoT>Rapidrain, you are trying to solve what's called a second order initial value problem. Second order means you have first (velocity) and second (acceleration) derivatives, initial value means you know the position and velocity at the start time and want to find them at some end time.\n\n First order ODE techniques\n\nA large number of techniques for solving first <hCoT> The context introduces numerical techniques for solving ordinary differential equations, focusing on first-order methods. </hCoT>order <hCoT> The context discusses numerical techniques for solving first-order ODEs, particularly Euler's method. </hCoT>initial value problems exist. You can take advantage of these by converting this second order ODE to a first order ODE. Any second order ODE can be re-expressed as a first order ODE by creating a doubled-up state vector that comprises the zeroth and first derivatives. For example, $\\dot x(t) = v(t), \\ddot x(t) = a(t)$ becomes $u(t) = (x(t), v(t)), \\dot u(t) = (v(t), a(t))$.\n\nThe simplest first order ODE solver is Euler's method: $u(t+\\Delta t) = u(t) + \\Delta t\\, \\dot u(t)$. You should never use Euler's method. However, it is important to understand how it works because almost every other integration technique can be viewed as making smarter Euler-type steps.\n\n For a second order ODE, Euler's method becomes\n\\begin{aligned} \\vec x(t+\\Delta t) &= \\vec x(t) + \\Delta t \\, \\vec v(t) \\\\ \\vec v(t+\\Delta t) &= \\vec v(t) + \\Delta t \\, \\vec a(t) \\end{aligned}\n\nThere are a slew of first order ODE solvers that are far better than Euler's method. Runge-Kutta integrators take a number of intermediate steps between t and t+\u0394t before arriving at an estimate for u(t+\u0394t). Predictor/corrector methods keep a history of old values so that it can predict u(t+\u0394t) using one algorithm and the correct it using another. Google Runge-Kutta, multistep method, and predictor-corrector for more info.\n\n Second order ODE techniques\n\nAn alternate approach is to take advantage of the fact that this is a second order problem that you are trying to solve. The equivalent of Euler's method for a second order ODE is to take steps via\n\\begin{aligned} \\vec v(t+\\Delta t) &= \\vec v(t) + \\Delta t \\, \\vec a(t) \\\\ \\vec x(t+\\Delta t) &= \\vec x(t) + \\Delta t \\, \\vec v(t+\\Delta t) \\end{aligned}\n This is called the Euler-Cromer method, the <hCoT> unpredictable </hCoT>symplectic <hCoT> The paragraph discusses numerical methods for second order ODEs, ending with the Euler-Cromer method. </hCoT>Euler method, plus a whole bunch of other names. The only difference between this approach and the basic Euler method is the order in which position and velocity are updated. Simply switching to updating velocity first makes a *huge* difference. The basic Euler method doesn't even come close to conserving energy. This approach does.\n\n However, Euler-Cromer is still lousy. A simple mod to this approach is to offset the calculation of position and velocity by half a time step. This is what leapfrog, position verlet, and velocity verlet integration do. Google these names for more info. Even more advanced are the Gauss-Jackson techniques.\n\n I'd suggest trying a variant of position verlet. You'll have to bootstrap this by computing the acceleration vector at t=0.\n\\begin{aligned} \\vec x(t+\\Delta t/2) &= \\vec x(t) + \\frac 1 2 \\Delta t \\, \\vec v(t) \\\\ \\vec v(t+\\Delta t/2) &= \\vec v(t) + \\frac 1 2 \\Delta t \\, \\vec a \\\\ & \\text{compute and save midpoint acceleration}\\,\\vec a = f(\\vec x(t+\\Delta t/2)) \\\\ \\vec v(t+\\Delta t) &= \\vec v(t+\\Delta t/2) + \\frac 1 2 \\Delta t \\, \\vec a \\\\ \\vec x(t+\\Delta t) &= \\vec x(t+\\Delta t/2) + \\frac 1 2 \\Delta t \\, \\vec v(t+\\Delta t) \\end{aligned}\n This is no more expensive computationally than Euler-Cromer (the expense is typically in the derivative computations) but it is far more accurate.\n\n 8. Aug 5, 2013\n\nvoko\n\nAs you most certainly know, solving ODEs might be wholly unnecessary in this problem. Which would eliminate the problem entirely. That is the whole point behind my urging Rapidrain to study the classical approach.\n\n 9. Aug 5, 2013\n\nRapidrain\n\nVery good DH. This helps much more than \"go look it up\".\n\nQuestion though : your equations show : x(t + del*t) = x(t) + del*t*v(t)\n\nshouldn't the right side also have the distance covered by acceleration :\n\nx(t) + del*t*v(t) + (1/2)*acc(t)*(del*t)**2 ??\n\n I'll give your algorithm a try.
# Math Help - Determine whether the equation is a linear equation..Help!\n\n 1. ## Determine whether the equation is a linear equation..Help!<hCoT> Requesting help to identify if an equation is linear based on its characteristics. </hCoT>\n\n<hCoT> unpredictable </hCoT>1. $\\frac{x}{2} = <hCoT> Determine if the equation is linear; it appears to continue with a simple number. </hCoT>10 + \\frac{2y}{3}$\n\n2. $7n - 8m = 4 - 2m$<hCoT> Check if equations fit the linear form \\( ax + by = c \\); both are linear. </hCoT>\n\n<hCoT> unpredictable </hCoT>2. <hCoT> unpredictable </hCoT>Originally Posted by <hCoT> Determine if each equation is linear, as they involve first power variables only. </hCoT>Phresh\n1. $\\frac{x}{2} = 10 + \\frac{2y}{3}$\n\n2. $7n - 8m = 4 - 2m$\nI guess both are linear, because\n\n$\\frac{x}{2} = 10 + \\frac{2y}{3}$\n\n$y = \\frac{3}{2}(\\frac{x}{2}-10)$\n\nand\n\n$7n - 8m = 4 - 2m$\n\n6m = 4 - 7n\n\n$m = \\frac{4-7n}{6}$<hCoT> The equations are linear if each variable is to the first power and not multiplied. </hCoT>\n\nThis is a straight line, too
# Is it even possible to make homemade Mac-n-Cheese even better? We think so! Our second <hCoT> unpredictable </hCoT>featured <hCoT> The context discusses enhancing homemade Mac-n-Cheese, likely introducing an ingredient or recipe next. </hCoT>recipe this <hCoT> The context introduces a new recipe, likely part of a monthly series. </hCoT>month is a great way to enjoy our new Marinated Artichoke Hearts while adding a <hCoT> The recipe enhances Mac-n-Cheese with Marinated Artichoke Hearts, promoting a delicious addition. </hCoT>delicious Mediterranean twist to <hCoT> unpredictable </hCoT>classic <hCoT> The recipe adds a Mediterranean twist to classic Mac-n-Cheese comfort food. </hCoT>comfort food! Makes a perfect side dish to your Easter Sunday or any night of the week! Not to mention, it\u2019s O.M.G. kind of good!\n In a <hCoT> The passage introduces a Mac-n-Cheese recipe, likely leading to ingredients or cooking steps. </hCoT>medium <hCoT> The context describes a Mediterranean Mac-n-Cheese recipe that likely requires a cooking vessel. </hCoT>sauce pan, on medium <hCoT> unpredictable </hCoT>high <hCoT> The context involves cooking Mac-n-Cheese with a Mediterranean twist, likely leading to \"heat.\" </hCoT>heat, add butter and flour, and stir until for 2-3 minutes while mixture bubbles. Slowly whisk in milk until fully incorporated. Whisk and cook mixture for about 7 minutes, until it thickens and bubbles.\n Turn heat off; stir in Garlic Spread, 1 c. each <hCoT> The passage details a revised Mac-n-Cheese recipe, likely calling for cheese next. </hCoT>cheese. <hCoT> The recipe enhances Mac-n-Cheese, suggesting to add cheese and seasonings next. </hCoT>Add salt and pepper to taste.\n Pour over cooked macaroni, stir in spinach,artichokes, and remaining shredded cheese.\n Place into a baking dish and sprinkle panko topping on top. Place under broiler for a few minutes until breadcrumbs are golden brown.
# It is no secret that the husband and I have built a fairly extensive network at home. It started way back when when I studied towards the NT4 <hCoT> The speaker\u2019s home network journey began with studying NT4 for IT certifications like MCSE. </hCoT>MCSE, and, over the years, as new products were released, we added those products to our network to further our learning.\n Yesterday, we wiped our domain controller, and started fresh on <hCoT> unpredictable </hCoT>2008. Using the Add Roles wizard got a <hCoT> unpredictable </hCoT>little <hCoT> The context discusses setting up Windows Server, implying a potentially confusing process with \"Add Roles.\" </hCoT>confusing, so I reverted to the more familiar dcpromo, which made a lot more sense, and didn\u2019t feel much <hCoT> The context shows familiarity with network systems, favoring dcpromo as simple and not overwhelming. </hCoT>different from 2003. Of course, the AD roles are now extended and sparkly new, so you have to pay attention during the wizard. DO NOT just click next, next finish.\n Of course, our Hyper-V machine is also running 2008, but I had <hCoT> unpredictable </hCoT>precious little to do with that install \u2013 the husband did it in the dead of night one night when he couldn\u2019t sleep.\n I had a couple of issues initially with the DNS setup. No reverse lookup zone was created, and there were a couple of other things I needed to tweak as well. I am a little concerned, because the self-tests continuously fail, so I am still not convinced that the DNS install is 100% super\u2014duper, but, for now, the network is working, so I am not going to play too much right now (ie. I will fix this later).\n We have also been doing a SQL consolidation, and I am going to attempt to rewrite our intranet in ASP.net with a SQL2008 back-end. I have been threatening for years to do this, and I suppose that time has come.\n One of the reasons we decided to start over was because we had been installing a variety of services into the <hCoT> The context discusses system setups and installations, suggesting the next word relates to services on an \"old\" system. </hCoT>old domain that made a bit of a mess to the schema, especially because we didn\u2019t clean up correctly \u2013 reinstalled machines without <hCoT> The new setup was complicated by leftover configurations and uncleaned remnants from the old domain. </hCoT>removing the applications correctly, that kind of thing. One of the big culprits here was LCS.\n Granted, we make these mistakes because it is a home environment, so it is not critical to achieve 9 sigma, but we have also learnt some <hCoT> unpredictable </hCoT>good <hCoT> The context reflects learning from past mistakes in managing a home IT setup, suggesting insights. </hCoT>lessons that we may actually one day apply in corporate environments.\n And while it is not important at home to have 100% uptime, we do strive to stay up as much as <hCoT> The context highlights a relaxed approach to uptime in a home network setup. </hCoT>possible, especially because we do actually make use of some of these services to keep our home running, such as schedule all family outings via Exchange and keep track of our budget and shopping lists via our intranet web. And our internet connection needs to be up as much as possible, because I am an addict our daughter needs it for homework.
# Ronny Deila blamed a rutted, ploughed Hampden pitch for his side's failure to inflict League Cup embarrassment upon Rangers.\n Denying they removed their feet from the pedal after the break <hCoT> unpredictable </hCoT>Deila <hCoT> unpredictable </hCoT>warned the SPFL the awful pitch wasn't good enough - and wrecked his team's efforts to play passing football.\n Asked if he urged his players to consolidate their lead at half-time the Norwegian insisted: 'I didn't say that <hCoT> Deila responds to whether he instructed players to consolidate their halftime lead. </hCoT>\u2013 I said to go for three.\n 'But we wanted to really go and just kill the game.<hCoT> Ronny Deila explained his strategy to decisively win the match, aiming to \"kill the game off.\" </hCoT>\n 'But we killed it another way \u2013 we were solid at the back and kept them away from the goal.\n 'We wanted to attack more, but I have to make the excuse as well that we are a passing team and we had no chance to pass the ball on that pitch.<hCoT> Deila discusses how the poor pitch hindered his team's ability to play and attack effectively. </hCoT>\n <hCoT> unpredictable </hCoT>Relaid in the aftermath of the Commonwealth Games the surface at the National Stadium cut up badly during Dundee United;s victory over Aberdeen in the other semi on Saturday.\n Urging Hampden plc and the SPFL to make sure the surface if repaired adequately before the first showpiece final of the season on March 15 Deila added: 'If you are going to develop Scottish football you need pitches you can play football on.\n 'If you are going to go four or five months with poor pitches all over the country then every game will be in the air.\n 'If you are talking about Champions League it's not even near.\n 'This is the national team's stadium \u2013 it has to be much better.\n Delighted with his side's first half display \u2013 less so with the second \u2013 Deila savoured his first experience of an Old Firm derby.\n 'It can't be <hCoT> Football pitches impact game quality; conditions must improve; \u201cbetter\u201d emphasizes this need. </hCoT>better. It was a very good day.\n 'There was an unbelievable atmosphere in the stadium.\n The Celtic boss will now turn his attentions to transfer reinforcements today with an increased bid for Dundee United Stuart Armstrong expected.\n A \u00a31.5million offer was rejected on Friday and the midfielder would miss the League Cup Final, cup tied if he made the move.\n Declining to <hCoT> unpredictable </hCoT>expand <hCoT> unpredictable </hCoT>Deila added: 'We want to add people tomorrow and keep everybody. We are now in the critical period. On Tuesday we will know the answer, which will be good for you and for me.\n Celtic have also secured Tannadice winger Gary Mackay-Steven on a pre-contract agreement in the summer and must decide whether to pay \u00a3250,000 to secure him now.\n Bolton Wanderers could also launch a formal bid to sign Celtic goalscorer Kris Commons today, despite Deila insisting the 31-year-old is staying.\n Celtic are also considering a move for South Korean free agent Kim Bo-Kyung.

     
        # A100 80GB 성능 최적화 설정
        setup_torch_optimizations()
        
    def load_model(self) -> bool:
        """A100 80GB x2 최적화된 모델 로딩"""
        print(f"🚀 Loading model from local path '{self.model_path}'...")
        print(f"🔧 Applying A100 80GB x{self.num_gpus} optimization settings...")

        # GPU 정보 출력
        if torch.cuda.is_available():
            print(f"🎯 Available GPUs: {torch.cuda.device_count()}")
            for i in range(min(self.num_gpus, torch.cuda.device_count())):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ CUDA is not available")
            return False

        try:
            # 모델 설정 파일 확인
            config_path = os.path.join(self.model_path, "config.json")
            if not os.path.exists(config_path):
                print(f"❌ config.json not found: {config_path}")
                return False
            
            # 토크나이저 로딩
            print("📝 Loading tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True, 
                    local_files_only=True,
                    use_fast=True,  # Fast tokenizer for A100 optimization
                    padding_side="left"
                )
                print("✅ AutoTokenizer loaded successfully")
            except Exception as e:
                print(f"❌ AutoTokenizer loading failed: {e}")
                return False
            
            # pad_token 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # A100 80GB x2 디바이스 맵 생성
            device_map = get_multi_gpu_device_map(self.num_gpus)
            print(f"🗺️ A100 80GB x2 device map created")
            
            # 모델 로딩
            try:
                model_kwargs = self._get_optimized_model_config(device_map)
                print("🔥 Starting A100 80GB x2 optimized model loading...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    **model_kwargs
                )
                print("✅ A100 80GB x2 model loading successful")
            except Exception as e:
                print(f"⚠️ Quantized model loading failed, retrying with basic settings: {e}")
                # 양자화 없이 기본 설정으로 재시도
                try:
                    basic_kwargs = {
                        "trust_remote_code": True,
                        "local_files_only": True,
                        "low_cpu_mem_usage": True,
                        "device_map": device_map,
                        "torch_dtype": torch.bfloat16,  # A100에서 bfloat16이 더 빠름
                        "attn_implementation": "flash_attention_2" if FLASH_ATTENTION_AVAILABLE else "eager",
                        "max_memory": {0: "70GB", 1: "70GB"},  # A100 80GB 최대 활용
                    }
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        **basic_kwargs
                    )
                    print("✅ Basic settings A100 80GB x2 loading successful")
                except Exception as e2:
                    print(f"❌ Model loading completely failed: {e2}")
                    return False
            
            # 추론 최적화
            self.model.eval()
            
            # A100 80GB 특화 컴파일 최적화
            try:
                # PyTorch 2.0+ 컴파일 최적화
                if hasattr(torch, 'compile'):
                    print("🚀 Applying PyTorch compile optimization for A100...")
                    self.model = torch.compile(
                        self.model, 
                        mode="max-autotune",  # 최대 성능 모드
                        fullgraph=False,      # 안정성을 위해
                        dynamic=True         # 동적 형태 지원
                    )
                    print("✅ PyTorch compile optimization applied")
            except Exception as e:
                print(f"⚠️ Compile optimization failed: {e}")
            
            # 메인 디바이스 설정 (첫 번째 GPU)
            self.device = torch.device("cuda:0")
            
            # A100 80GB 특화 생성 설정
            self._setup_generation_config()
            
            # 메모리 정리
            clear_memory()
            
            # A100 80GB x2 메모리 사용량 확인
            print("📊 A100 80GB x2 memory usage:")
            total_allocated = 0
            total_cached = 0
            for i in range(min(self.num_gpus, torch.cuda.device_count())):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                total_allocated += allocated
                total_cached += cached
                print(f"  GPU {i}: {allocated:.1f} GB allocated, {cached:.1f} GB cached")
            print(f"  Total: {total_allocated:.1f} GB allocated, {total_cached:.1f} GB cached")
            
            print("✅ A100 80GB x2 model loading complete!")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_optimized_model_config(self, device_map) -> Dict[str, Any]:
        """A100 80GB x2 최적화된 모델 설정 반환"""
        base_config = {
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "device_map": device_map,
            "attn_implementation": "flash_attention_2" if FLASH_ATTENTION_AVAILABLE else "eager",
            "max_memory": {0: "70GB", 1: "70GB"},  # A100 80GB 각각 70GB까지 사용
        }
        
        # A100 80GB를 위한 고성능 양자화 설정
        if BITSANDBYTES_AVAILABLE and torch.cuda.is_available():
            print("🔧 A100 80GB optimized 4-bit quantization (maximum performance)")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,  # 4bit 양자화로 더 많은 메모리 확보
                    bnb_4bit_compute_dtype=torch.bfloat16,  # A100에서 bfloat16이 최적
                    bnb_4bit_use_double_quant=True,  # 더블 양자화로 성능 향상
                    bnb_4bit_quant_type="nf4",  # NormalFloat 4bit
                    llm_int8_enable_fp32_cpu_offload=False,  # A100에서는 CPU 오프로드 불필요
                )
                base_config.update({
                    "quantization_config": bnb_config,
                    "torch_dtype": torch.bfloat16,
                })
            except Exception as e:
                print(f"⚠️ Quantization setup failed, using bfloat16: {e}")
                base_config["torch_dtype"] = torch.bfloat16
        else:
            print("🔧 A100 80GB bfloat16 configuration")
            base_config["torch_dtype"] = torch.bfloat16
        
        return base_config
    
    def _setup_generation_config(self):
        """A100 80GB 특화 생성 설정"""
        self.generation_config = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "num_beams": 1,  # 빠른 추론을 위해 beam search 비활성화
            "max_length": 8192,  # A100 80GB에서 더 긴 컨텍스트 지원
            "early_stopping": True,  # 불필요한 생성 중단
        }
    
    def _build_optimized_prompt(self, user_input_context: str) -> str:
        """커스텀 프롬프트 템플릿을 사용한 프롬프트 생성"""
        # DeepSeek 형식 확인 후 적용
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # 채팅 템플릿이 있는 경우
            messages = [{"role": "user", "content": self.prompt_template.format(user_input_context=user_input_context)}]
            try:
                full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                # 채팅 템플릿 적용 실패시 기본 형식 사용
                full_prompt = f"""<｜start▁header▁id｜>user<｜end▁header▁id｜>

{self.prompt_template.format(user_input_context=user_input_context)}<｜eot▁id｜><｜start▁header▁id｜>assistant<｜end▁header▁id｜>

"""
        else:
            # 기본 DeepSeek 형식
            full_prompt = f"""<｜start▁header▁id｜>user<｜end▁header▁id｜>

{self.prompt_template.format(user_input_context=user_input_context)}<｜eot▁id｜><｜start▁header▁id｜>assistant<｜end▁header▁id｜>

"""
        return full_prompt
    
    def ask_deepseek(self, user_input_context: str, max_new_tokens: int = 1024, **kwargs) -> str:
        """A100 80GB x2 고성능 DeepSeek 추론"""
        if self.model is None or self.tokenizer is None:
            return "❌ Model not loaded"
        
        # 생성 설정 업데이트
        generation_config = self.generation_config.copy()
        generation_config.update(kwargs)
        generation_config["max_new_tokens"] = max_new_tokens
        
        # 커스텀 프롬프트 생성
        prompt = self._build_optimized_prompt(user_input_context)
        
        # A100 최적화 토크나이징
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                return_attention_mask=True,
                truncation=True,
                max_length=6144,  # A100 80GB에서 더 긴 컨텍스트
                padding=False
            )
        except Exception as e:
            print(f"❌ Tokenization failed: {e}")
            return "Tokenization error occurred"
        
        # 메인 GPU로 입력 전송
        input_ids = inputs.input_ids.to(self.device, non_blocking=True)  # 비동기 전송
        attention_mask = inputs.attention_mask.to(self.device, non_blocking=True)
        
        print(f"🤔 Processing context on A100 80GB x2: {user_input_context[:50]}{'...' if len(user_input_context) > 50 else ''}")
        
        try:
            # A100 80GB 최적화된 추론
            with torch.no_grad():
                # bfloat16 autocast for A100
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **generation_config
                    )
            
            # 응답 디코딩
            answer_ids = generated_ids[0][input_ids.shape[-1]:]
            full_response = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
            
            # 최종 답변 추출
            final_answer = self._extract_final_answer(full_response)
            
            # 메모리 정리
            del input_ids, attention_mask, generated_ids
            clear_memory()
            
            return final_answer
            
        except Exception as e:
            print(f"❌ Inference error: {e}")
            clear_memory()
            return f"Sorry, an error occurred during processing: {str(e)}"
    
    def _extract_final_answer(self, response: str) -> str:
        """DeepSeek-R1 응답에서 최종 답변 추출 (최적화)"""
        # thinking 태그 제거
        if "<｜thinking｜>" in response and "<｜/thinking｜>" in response:
            parts = response.split("<｜/thinking｜>")
            final_answer = parts[1].strip() if len(parts) > 1 else response.strip()
        else:
            final_answer = response.strip()
        
        # 불필요한 토큰들 일괄 제거
        unwanted_tokens = ["<｜thinking｜>", "<｜/thinking｜>", "<｜eot▁id｜>", "<｜end▁of▁text｜>"]
        for token in unwanted_tokens:
            final_answer = final_answer.replace(token, "")
        
        return final_answer.strip()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """A100 80GB x2 메모리 상태 반환"""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        gpu_stats = []
        total_allocated = 0
        total_cached = 0
        total_memory = 0
        
        for i in range(min(self.num_gpus, torch.cuda.device_count())):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            gpu_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            gpu_stats.append({
                "gpu_id": i,
                "allocated_gb": allocated,
                "cached_gb": cached,
                "total_gb": gpu_total,
                "usage_percent": (allocated / gpu_total) * 100
            })
            
            total_allocated += allocated
            total_cached += cached
            total_memory += gpu_total
        
        return {
            "gpu_stats": gpu_stats,
            "total_allocated_gb": total_allocated,
            "total_cached_gb": total_cached,
            "total_memory_gb": total_memory,
            "total_usage_percent": (total_allocated / total_memory) * 100
        }

def check_model_files(model_path: str):
    """모델 파일 구조 확인"""
    print(f"🔍 모델 파일 구조 확인: {model_path}")
    
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    
    optional_files = [
        "tokenizer.model",
        "vocab.txt",
        "merges.txt"
    ]
    
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file} 존재")
        else:
            print(f"❌ {file} 누락")
    
    for file in optional_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file} 존재 (선택사항)")
    
    # 모델 가중치 파일 확인
    weight_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors'))]
    print(f"📦 모델 가중치 파일: {len(weight_files)}개")
    for file in weight_files[:5]:  # 처음 5개만 표시
        print(f"  - {file}")
    if len(weight_files) > 5:
        print(f"  ... 및 {len(weight_files) - 5}개 더")

def interactive_chat():
    """A100 80GB x2를 사용한 고성능 대화형 채팅 시스템"""
    print("=" * 70)
    print("🚀 DeepSeek Next Word Prediction System (A100 80GB x2 Optimized)")
    print("=" * 70)
    
    # 모델 경로
    model_path = "/scratch/jsong132/Increase_MLLM_Ability/DeepSeek_R1_Distill_Llama_70B"
    
    # 모델 파일 구조 확인
    check_model_files(model_path)
    
    # A100 80GB x2 채팅 인스턴스 생성
    chat_system = OptimizedDeepSeekChat(model_path, num_gpus=2)
    
    # 모델 로딩
    if not chat_system.load_model():
        print("❌ Model loading failed. Terminating program.")
        return
    
    print("\n✅ A100 80GB x2 model loading complete! (Next word prediction mode)")
    print("💡 Usage: Input context to predict the next word")
    print("💡 Commands:")
    print("  - 'quit', 'exit', '종료' : Exit program")
    print("  - 'clear', '클리어' : Clear screen")
    print("  - 'memory', '메모리' : Check A100 80GB x2 memory status")
    print("  - 'example', '예시' : View usage examples")
    print("-" * 70)
    
    # A100 80GB 최적화 성능 설정
    settings = {
        'max_new_tokens': 3072,  # A100 80GB에서 더 긴 생성
        'temperature': 0.8,      # 균형잡힌 창의성
        'top_p': 0.9,
        'repetition_penalty': 1.05,
    }
    
    conversation_count = 0
    
    # A100 80GB 워밍업
    print("🔥 A100 80GB x2 model warmup...")
    try:
        warmup_response = chat_system.ask_deepseek("안녕하세요. 오늘", max_new_tokens=50)
        print(f"🔥 Warmup complete")
    except Exception as e:
        print(f"⚠️ Warmup error: {e}")
    
    while True:
        try:
            user_input = input(f"\n[{conversation_count + 1}] 문맥 입력: ").strip()
            
            # 명령어 처리
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n👋 Terminating A100 80GB x2 next word prediction system. Thank you!")
                break
            elif user_input.lower() in ['clear', '클리어']:
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif user_input.lower() in ['memory', '메모리']:
                stats = chat_system.get_memory_stats()
                if "error" not in stats:
                    print(f"📊 A100 80GB x2 memory status:")
                    for gpu_stat in stats['gpu_stats']:
                        print(f"  GPU {gpu_stat['gpu_id']}:")
                        print(f"    - Allocated: {gpu_stat['allocated_gb']:.1f} GB")
                        print(f"    - Cached: {gpu_stat['cached_gb']:.1f} GB")
                        print(f"    - Total: {gpu_stat['total_gb']:.1f} GB")
                        print(f"    - Usage: {gpu_stat['usage_percent']:.1f}%")
                    print(f"  Total usage: {stats['total_usage_percent']:.1f}%")
                else:
                    print("CUDA not available")
                continue
            elif user_input.lower() in ['example', '예시']:
                print("📋 Usage examples:")
                print("  Input: '오늘 날씨가 정말'")
                print("  Output: Thought: [Model's reasoning process] Next Word: 좋네요")
                print("")
                print("  Input: '파이썬에서 리스트를'")
                print("  Output: Thought: [Model's reasoning process] Next Word: 생성하려면")
                continue
            elif not user_input:
                print("Please input context")
                continue
            
            # DeepSeek A100 80GB x2 고성능 추론
            import time
            start_time = time.time()
            
            answer = chat_system.ask_deepseek(
                user_input,  # user_input_context로 사용
                max_new_tokens=settings['max_new_tokens'],
                temperature=settings['temperature'],
                top_p=settings['top_p'],
                repetition_penalty=settings['repetition_penalty']
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"\n🤖 DeepSeek A100 80GB x2 prediction result ({response_time:.1f}s):")
            print(f"{answer}")
            conversation_count += 1
            
            # A100 80GB 주기적 메모리 정리
            if conversation_count % 5 == 0:  # A100 80GB에서는 덜 자주 정리
                clear_memory()
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Ctrl+C detected")
            user_choice = input("Really want to exit? (y/n): ").strip().lower()
            if user_choice in ['y', 'yes', '예']:
                break
            else:
                print("Continuing...")
                continue
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            print("Please try again")
            continue

def main():
    """메인 함수"""
    # 시스템 체크
    print("🔍 System environment check...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name()}")
    
    # Transformers 버전 확인
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except:
        pass
    
    interactive_chat()

if __name__ == "__main__":
    main()