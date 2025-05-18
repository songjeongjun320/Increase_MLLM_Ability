import torch
from transformers import AutoModelForCausalLM, AutoTokenizer # 필요한 라이브러리 임포트

# --- 모델 및 토크나이저 로드 ---
# 실제 모델 파일들이 저장된 경로로 수정해주세요.
model_path = "/scratch/jsong132/Increase_MLLM_Ability/Llama3:1_8B_Instruct" # 여기에 실제 경로를 입력하세요

print(f"'{model_path}' 경로에서 토크나이저 로드 중...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("토크나이저 로드 성공.")
except Exception as e:
    print(f"토크나이저 로드 실패: {e}")
    print(f"경로 '{model_path}'에 토크나이저 파일들이 올바르게 있는지 확인하세요.")
    exit()

print(f"\n'{model_path}' 경로에서 모델 로드 중...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # 또는 torch.float16 (GPU 지원 여부 확인)
        device_map="auto"            # 사용 가능한 장치에 자동으로 모델 분산 (GPU, CPU)
                                     # CPU만 사용 시: device_map="cpu"
    )
    print("모델 로드 성공.")
    model.eval() # 추론 모드로 설정
except Exception as e:
    print(f"모델 로드 실패: {e}")
    print("RAM/VRAM이 충분한지, 다운로드된 파일들이 올바른지 확인하세요.")
    print("Llama-3.1-8B 모델은 bfloat16으로도 최소 16GB 이상의 VRAM이 필요할 수 있습니다.")
    exit()
# ----------------------------------------------------

# 프롬프트 템플릿 정의
prompt_template = """Task Instruction: Given certain text, you need to predict the next word of it. Moreover, before your output, you could first give short thoughts about how you infer the next word based on the provided context.

Here are five examples for the task:
Example 0: {‘극성, 비양성 용매’의 예는 무엇인가요? 몇 가지 <hCoT> unpredictable </hCoT> 예로는 메틸렌 <hCoT> The context seeks examples of polar, aprotic solvents; methylene chloride may follow. </hCoT>클로라이드, <hCoT> unpredictable </hCoT>디에틸 <hCoT> The context lists examples of polar, aprotic solvents like methylene chloride and diethyl ether. </hCoT>에터, 클로로폼 등이 있습니다. 극성 비양성 용매는 유의미한 전하 분리가 있는 <hCoT> A polar aprotic solvent lacks hydrogen bonds, focusing on its chemical properties as a molecule. </hCoT>분자(즉, 극성 용매)로, 용매로 사용될 수 있지만 <hCoT> Polar aprotic solvents have charge separation and do not donate hydrogen ions. </hCoT>산-염기 <hCoT> unpredictable </hCoT>평형을 일으키지 않는 용매입니다. 따라서 물과 플루오르화수소는 <hCoT> The context clarifies \"polar, aprotic solvents\" and excludes water and hydrogen fluoride. </hCoT>확실히 극성 분자이지만, 양성자(${H}^{+}$)를 쉽게 교환하므로 비양성 용매에 해당하지 <hCoT> unpredictable </hCoT>않습니다.}
Example 1: {# 마우스 드래그에 따라 다른 객체 주위에서 객체가 부드럽게 궤도 회전하도록 제한하기\n\n저는 Unity에서 작업하고 있습니다.\n\n저는 각각 2D에서 자유롭게 드래그할 수 있는 특별한 객체들을 가지고 있습니다.\n\n이 객체들은 다른 객체들과 너무 가까워질 수 없습니다 (최소 허용되는 Vector3.distance는 1.5f입니다).\n\n또한 특별한 커넥터들을 가지고 있는데, 이 커넥터는 이 특별한 객체들 중 2개를 자유롭게 연결할 수 있습니다. 두 객체가 <hCoT> The context describes dragging 2D objects with constraints and potential actions upon connecting them. </hCoT>연결되면, 드래그하는 것이 더 제한됩니다.\n\n이 2개의 객체는 서로 너무 멀어질 수 없습니다 (최대 허용되는 Vector3.distance는 5f입니다).\n\n이 2가지 제한은 작동하지만, 문제는 이 제한이 발생하면 객체들이 거의 고정된 상태로 남아 있으며, 제한을 넘어서려고 할 때 객체가 고정된 자리에 그대로 멈추게 된다는 점입니다.\n\n제가 원하는 것은 드래그된 객체가 연결된 객체 주위를 부드럽게 궤도처럼 회전하도록 만드는 것입니다. 두 가지 제한 중 하나에 도달했을 때, 이 동작이 발생해야 합니다.\n\n제가 생각한 가장 좋은 예는 앵그리버드의 슬링샷입니다.\n\n슬링샷이 최대 반동에 도달하면, 그 최대 거리에서 부드럽게 위나 아래로 궤도를 따라 움직일 수 있습니다. 이와 같은 부드러운 효과가 너무 가까운 경우와 너무 먼 경우 모두에서 발생해야 합니다.\n\n제 설명이 충분히 이해되었기를 <hCoT> unpredictable </hCoT>바랍니다. 답을 찾으려 시도했지만 작동하는 해결책을 찾지 못했습니다.\n\n제가 찾은 가장 <hCoT> The context describes dragging objects in 2D with movement restrictions and seeking smooth transitions. </hCoT>가까운 해결책은 레이와 레이의 접점 함수처럼 동작하는 방법인데, 이 방법이 제 게임에서 어떻게 구현될 수 있는지 전혀 알지 못합니다.\n\n감사합니다.\n\nRemember the BBQ.\n\n링크 제공: 앵그리버드의 당기기 메커니즘에 대한 링크: https://unity3d.com/learn/tutorials/topics/physics/making-angry-birds-style-game-part-1\n\n이미지 제공은 각 제한의 경계를 정확하게 나타내지 않으며, 단지 더 명확하게 보이도록 하기 위한 것입니다.\n\n<hCoT> The context describes dragging 2D objects with constraints and potential actions upon connecting them. </hCoT>\n\n질문을 정확히 이해한 것이라면, 이 작업은 벡터 수학을 사용하여 할 수 있습니다. 제가 자바스크립트로 예제를 만들었으니, 여기서 사용한 많은 수학적 연산들이 Unity에서 내장 클래스와 메서드를 통해 매우 쉽게 구현될 수 있다는 점을 참고하세요.\n\n공은 마우스를 창 위에 올려놓으면 따라갑니다. 가능하면 Chrome에서 실행하시길 권장합니다. 또한 \"Run Snippet\"을 클릭한 후 \"Full page\" 링크를 클릭하시길 추천합니다. 그 이유는 마우스 좌표가 스크롤바를 고려하지 않기 때문에, 페이지를 스크롤하면 마우스 좌표가 부정확해질 수 있기 때문입니다.\n\n\"mousemove\"라는 함수에서 마법이 일어납니다.\n\n추가 설명이 필요하면 언제든지 질문해주세요.\n\njavascript\nvar OUTER_BOUNDS = 160;\nvar INNER_BOUNDS = 40;\n\nvar canvas = document.getElementById(\"mainCanvas\");\nvar ctx = canvas.getContext(\"2d\");\n\nvar centerPos = {\n<hCoT> JavaScript example uses vector math for game simulation, focusing on canvas properties and coordinates. </hCoT>x: <hCoT> The context is a game focused on graphical elements and mouse interactions in JavaScript. </hCoT>(canvas.width / 2),\ny: (canvas.height / 2)\n};\n\nvar ballPos = {\nx: -1000,\ny: -1000\n};\n\nfunction mousemove(event) {\n  // 이 부분에서 중요한 일이 발생합니다.\n  // 마우스 위치에 공을 놓습니다.\n  ballPos.x = event.clientX;\n  ballPos.y = <hCoT> JavaScript updates ball position using mouse coordinates; next is event.clientY. </hCoT>event.clientY;\n\n  // 중심에서 공까지의 거리를 계산합니다. 이는 간단한 선형 대수입니다.\n  var dX = ballPos.x - centerPos.x;\n  var dY = ballPos.y - centerPos.y;\n  var distance = Math.sqrt(Math.pow(dX, 2) + Math.pow(dY, 2));\n\n  if (distance > 0) {\n    // 공에서 중심으로 향하는 단위 벡터를 계산합니다. 이는 벡터를 정규화하는 것과 같습니다.\n    var direction = {\nx: dX / distance,\ny: dY / distance\n};\n\n    // 내부 및 외부 경계를 제곱하여 비교할 때 제곱근 연산을 사용하지 않고도 동일한 비교를 할 수 있습니다.\n    if (distance > (OUTER_BOUNDS - BALL_RADIUS)) {\n      // 공이 외부 경계를 넘었습니다. 단위 벡터를 사용하여 새로운 위치를 계산합니다.\n      ballPos.x = centerPos.x + (direction.x * (OUTER_BOUNDS - BALL_RADIUS));\n      ballPos.y = centerPos.y + (direction.y * (OUTER_BOUNDS - BALL_RADIUS));\n    } else if (distance < (INNER_BOUNDS + BALL_RADIUS)) {\n      // 공이 너무 가까워졌습니다. 위와 동일한 방법으로 새로운 위치를 계산합니다.\n      ballPos.x = centerPos.x + (direction.x * (INNER_BOUNDS + BALL_RADIUS));\n      ballPos.y = centerPos.y + (direction.y * (INNER_BOUNDS + BALL_RADIUS));\n    }\n  }\n};\n\nfunction run() {\n  window.requestAnimationFrame(run);\n  draw();\n}\n\nfunction draw() {\n  // 배경 그리기.\n  ctx.fillStyle = \"#444444\";\n  ctx.fillRect(0, 0, canvas.width, canvas.height);\n\n  // 중심 공 그리기.\n  strokeCircle(centerPos.x, centerPos.y, OUTER_BOUNDS, \"#FF0000\");\n  strokeCircle(centerPos.x, centerPos.y, INNER_BOUNDS, \"#FF0000\");\n\n  // 움직이는 공 그리기.\n}\n\nfunction fillCircle(x, y, radius, color) {\n  ctx.fillStyle = color;\n  ctx.beginPath();\n  ctx.arc(x, y, radius, 0, Math.PI * 2);\n  ctx.fill();\n}\n\nfunction strokeCircle(x, y, radius, color) {\n  ctx.strokeStyle = color;\n  ctx.lineWidth = 2;\n  ctx.beginPath();\n  ctx.arc(x, y, radius, 0, Math.PI * 2);\n  ctx.stroke();\n}\n\nrun();\n\nhtml,\nbody {\n  margin: 0;\n}\n\n<canvas id=\"mainCanvas\" width=\"400\" height=\"400\"></canvas>\n\n• 정말 멋집니다!! 피드백 주셔서 감사합니다. 그 코드 조각은 정말 유용했어요! – Remember The BBQ 2016년 6월 21일 21:39\n• 시간이 될 때 제 게임에 구현해보고, 추가적인 설명이 필요하면 다시 물어보겠습니다. 다시 한 번 감사합니다!! :) – Remember The BBQ 2016년 6월 21일 21:40\n\n---\n\n하드 제한 대신에 반대되는 힘을 사용할 수 있습니다.\n\n두 객체 사이에 고무줄을 시뮬레이션하고 싶습니다. 이런 종류의 힘은 \"객체와의 거리\"를 \"고무줄의 스프링 강성\"의 거듭제곱으로 계산하여 시뮬레이션할 수 있습니다. 객체 A가 객체 B로부터 멀어지려고 할 때 이 힘을 계산하려면 \"거리 상수\"를 그 값에서 빼고, 그 후에 모든 음수 값을 0으로 설정하는 체크를 해야 합니다. 다른 거리 임계값에서는 고무줄이 끊어지도록 설정할 수 있습니다.\n\n• 피드백 감사합니다. 시간이 될 때 이 아이디어를 구현해보겠습니다 :) – Remember The BBQ 2016년 6월 21일 16:13\n• 객체를 밀 때는 addforce를 사용해야 하나요? 게임에 물리 엔진이 없어서 이게 작동할지 궁금합니다. – Remember The BBQ 2016년 6월 21일 16:13\n• 지금 생각해보니 이 방법은 효과가 없을 것 같습니다. 왜냐하면 객체가 마우스의 위치를 기준으로 매번 LateUpdate() 호출 시 재위치되니까, 이게 힘을 추가하는 걸 <hCoT> The discussion questions the effectiveness of applying force due to frequent object repositioning. </hCoT>의미 없게 만들지 않을까요? <hCoT> unpredictable </hCoT>– Remember The BBQ 2016년 6월 21일 16:29\n• 고무줄이란 SpringJoint2D를 사용한다는 의미인가요? – Remember The BBQ 2016년 6월 21일 16:51\n• 네, 밀기 작업은 addforce를 통해 해야 합니다. 마우스 코드도 변경해야 할 것 같습니다. 이제 마우스가 재위치되지 않고 마우스 클릭 지점으로 끌어당겨지도록 해야 합니다 (아마도 일정 거리까지 레이캐스팅을 사용해야 할 수도 있습니다). – Tealr 2016년 6월 29일 6:06}
Example 2: {00-01E\n\n# 00-01E\n\n작성자 - 2012년 2월 8일 게시\n\nUDC:\n536.432.1; 536.44.\nPACS:\n05.70.Jk<hCoT> The context lists classification codes indicating a likely transition to a related title or topic. </hCoT>\n\n<hCoT> The context is a bibliographic reference with classification codes for scientific documents. </hCoT>대칭적인 이진 유체 혼합물의 기체-액체 임계점에 대한 아비<hCoT> unpredictable </hCoT>니시오 <hCoT> Context is an academic citation with classification codes; next word likely relates to study. </hCoT>연구\n\n<hCoT> unpredictable </hCoT>O.V. <hCoT> The context likely discusses an academic study by O.V., predicting \"Patsahan\" as the surname. </hCoT> Patsahan\nM.P. Kozlovskii\n<hCoT> unpredictable </hCoT>R.S. <hCoT> The context indicates an academic study on binary fluid mixtures, likely introducing the author Melnyk. </hCoT> Melnyk\n\n기체-액체 임계점 근처에서 대칭적인 이진 유체 혼합물의 거동을 <hCoT> The text likely starts a physics or chemistry research paper on a binary fluid mixture. </hCoT> 조사하기 위한 미시적 접근법이 <hCoT> The context discusses a study on a symmetrical binary fluid mixture's behavior near its critical point. </hCoT> 제안됩니다. 이 문제는 외부 필드에서 3D Ising 모델의 분할<hCoT> The context explores a symmetrical binary fluid mixture's critical point and likely involves partition function calculations. </hCoT> 함수 계산으로 축소될 수 있음을 <hCoT> The context introduces a study on a binary fluid mixture's critical point and methods. </hCoT>보여줍니다. 우리는 사각 우물 대칭 이진 혼합물에 대해 임계점의 매개변수(임계 온도와 임계 밀도)를 미시적 <hCoT> The study focuses on a symmetrical binary mixture's critical point parameters, introducing a specific parameter. </hCoT> 매개변수인 $r$ (다양한 종의 입자 간 상호작용의 상대 강도)와 $\lambda$ (포텐셜 우물의 너비)를 <hCoT> The text explores calculating properties of a symmetrical binary fluid mixture at its critical point. </hCoT> 함수로 계산합니다. <hCoT> The study explores a binary fluid mixture's critical point, focusing on parameter $r$'s role. </hCoT> 얻어진 결과는 컴퓨터 시뮬레이션의 결과와 잘 일치합니다.\n\n연도:\n2000\n페이지:<hCoT> The context discusses an academic study on a symmetrical binary fluid mixture's critical point. </hCoT> \n<hCoT> The text discusses an academic study, likely leading to the article's total page count. </hCoT> 24}
Example 3: {대체 대수학 복습 | KS3 수학 자료\n\n## 알아야 할 것들\n\n기억할 사항:\n\n• 대체는 단순히 문자를 숫자로 교체하는 것을 의미합니다.\n• 숫자와 <hCoT> Context explains substitution in algebra, noting no \u201c\u00d7\u201d before a letter. </hCoT> 문자 사이에 $\\times$ 기호가 숨겨져 있으므로, 이를 <hCoT> unpredictable </hCoT> 기억해야 합니다!\n• 분수는 또 다른 방식으로 나눗셈 문제를 나타내는 방법입니다.\n\n그렇다면 대체란 무엇일까요? 대체는 한 가지를 다른 것으로 '바꾸는' <hCoT> unpredictable </hCoT> 것을 의미합니다. <hCoT> The context explains substitution in algebra, likely leading to an example related to maths. </hCoT> 수학에서는 주로 문자를 숫자로 바꾸는 것을 의미합니다.\n\n**대체를 사용하여 $x + 7$의 값을 구하시오, 여기서 x = 12입니다.**\n\n여기서 $x = 12$라고 주어졌으므로, 우리가 해야 할 일은 식에서 $x$를 12로 바꾸는 것뿐입니다!\n\n$$x+7=12+7=19$$\n\n쉽죠! 다른 연산도 똑같이 처리하면 됩니다!\n\n**대체를 사용하여 $x - 4$의 <hCoT> The context explains using substitution in algebra to simplify expressions, introducing a new problem. </hCoT> 값을 구하시오, 여기서 x = 15 <hCoT> The text explains substitution in algebra, replacing variables with numbers, using examples like \\( x - 4 \\). </hCoT> 입니다.**\n\n <hCoT> unpredictable </hCoT> $$x-4=15-4=11$$\n\n곱셈 문제는 조금 다릅니다. 왜냐하면 숫자와 문자 사이에 숨겨진 $\\times$ 기호가 있다는 것을 기억해야 하기 때문입니다.\n\n**대체를 사용하여 $5x$의 값을 구하시오, 여기서 x = 13입니다.**\n\n$$5x=5\\times x=5\\times13=65$$\n\n나눗셈 문제는 두 가지 형태로 나올 수 있습니다:\n\n**대체를 사용하여 $x\\div3$의 값을 구하시오, 여기서 x = 9입니다.**\n\n$$x\\div3=9\\div3=3$$\n\n또는 분수 형태로 나와서 변환해야 할 수 있습니다:\n\n**대체를 사용하여 $\\frac{20}{x}$의 값을 구하시오, 여기서 x = 5입니다.**\n\n$$\\frac{20}{x}=20\\div x=20\\div5=4$$\n\n## KS3 <hCoT> unpredictable </hCoT> 수학 <hCoT> The context explains substitution in math, guiding KS3 students with examples. </hCoT> 복습 카드\n\n(77개의 리뷰) ₤8.99\n\n## 예시 문제들\n\n$$12x=12\\times x=12\\times7=84$$\n\n$$\\dfrac{x}{9}=x\\div9=54\\div9=6$$\n\n## KS3 수학 복습 카드\n\n(77개의 리뷰) ₤8.99\n• 모든 주요 KS2 수학 SATs 주제 포함\n• 각 주제에 대한 연습 문제와 답안 제공}
Example 4: {랜달즈빌, 뉴욕의 이사 및 이사 회사: Movers MAX .:\nMovers MAX 이사 디렉토리는 <hCoT> The context suggests a comprehensive resource directory for all moving services and information. </hCoT> 이사 자원을 위한 원스톱 소스입니다. 중요한 이사를 하기 전에 이사에 대해 <hCoT> unpredictable </hCoT> 조사하십시오. 이 안에서는 다양한 유형의 이사 서비스에 대한 유용한 <hCoT> unpredictable </hCoT> 랜달즈빌, <hCoT> unpredictable </hCoT> 뉴욕 이사 가이드와 이사 팁을 찾을 수 있으며, 랜달즈빌, 뉴욕의 <hCoT> Movers MAX offers resources, guides, tips, and free estimates for moving services in Randallsville. </hCoT> 전문 이사 회사로부터 무료 이사 견적을 받을 수 있습니다.\n모든 랜달즈빌, <hCoT> The context discusses Randallsville, New York movers and services, hinting at forthcoming location specifics. </hCoT> 뉴욕 <hCoT> The text lists moving resources in Randallsville, New York, suggesting \"York\" is the next word. </hCoT> 이사 유형에 대한 정보가 제공되므로, 랜달즈빌, 뉴욕 어디에서든 이사에 필요한 정보를 확인할 수 있습니다. 지역 이사부터 전국 또는 전 세계로의 장거리 이사, 해외로의 국제 이사까지, 저희 <hCoT> The context outlines moving services in Randallsville, New York, including various types and logistics. </hCoT> 랜달즈빌, <hCoT> unpredictable </hCoT> 뉴욕 <hCoT> The context highlights Movers MAX's services for local and long-distance moving in Randallsville. </hCoT> 이사 <hCoT> The text details moving services in Randallsville, New York, from local to international options. </hCoT> 서비스 디렉토리는 다음 랜달즈빌, 뉴욕 이사를 계획하는 데 필요한 모든 정보를 제공합니다.\n전국적인 주거 이사 및 상업 <hCoT> The context discusses a directory for both residential and commercial moving services. </hCoT> 이사 <hCoT> The context discusses various moving services, including local and international, mentioning movers directory details. </hCoT> 회사 디렉토리는 무료 랜달즈빌, 뉴욕 이사 및 보관 견적을 모든 주나 국가에 제공할 수 있습니다. 소규모 주거지 및 가정의 지역 이사 또는 장거리 이사부터 사무실 이사 및 기업 이주까지, 랜달즈빌, 뉴욕 이사 회사는 귀하의 랜달즈빌, 뉴욕 이사를 도와드릴 수 있습니다.\n랜달즈빌, 뉴욕 이사 회사는 랜달즈빌, 뉴욕의 지역 이사 회사 디렉토리로, 귀하의 지역 이사를 도와줄 준비가 된 랜달즈빌, 뉴욕의 이사 회사들을 제공합니다. 랜달즈빌, <hCoT> The text details various moving services, indicating next word may specify a location or move type. </hCoT> 뉴욕으로 이사하시거나 랜달즈빌, 뉴욕에서 이사하시든, 저희 지역 이사 회사는 전방위적인 주거 이사 서비스와 YOU-Pack 이사 또는 셀프 서비스 이사를 도와드릴 수 있습니다.\n차를 이사해야 하시나요? 랜달즈빌, 뉴욕 이사 회사는 랜달즈빌, 뉴욕의 자동차 운송 회사 견적과 신뢰할 수 있는 랜달즈빌, 뉴욕의 자동차 운송 회사를 제공하여 랜달즈빌, 뉴욕 내 또는 전 세계 어디로든 차를 이사할 수 있도록 돕습니다.\n <hCoT> unpredictable </hCoT> 랜달즈빌, 뉴욕 이사 회사가 귀하의 다음 랜달즈빌, 뉴욕 이사를 도와드리며, 지금 바로 무료 랜달즈빌, 뉴욕 이사 회사 견적을 받아보세요!}
Example 5: {우리는 가끔 온라인 쿠폰과 기타 특별 혜택을 제공합니다. <hCoT> Customers can explore additional ways to find deals beyond online coupons, like subscribing. </hCoT> 또는 제품 연구에 참여하고 싶다면 '홈 제품 배치'를 체크하고 몇 가지 질문에 답해주세요. 무엇을 기다리고 있나요?}

Now please give me your prediction for the thought and next word based on the following context:

{미주라에서 초보자를 위한 바비큐 }

Thought:
Next Word:"""

# 실제로는 의미 있는 예시를 제공하는 것이 좋습니다.
examples = [
    "Context: 미주라에서 초보자를 위한 바비큐 클래스가 열립니다!",
    "Context: 맛있는 바비큐 만드는 실력을 더 향상시키고 싶으세요?",
    "Context: 기회가 올 거예요, 지금 달력에 적어두세요.",
    "Context: 9월 22일 목요일, 월드 클래스 바비큐 챔피언 토니 발레이(Lonestar Smoke Rangers 소속)와 함께 하세요.",
    "Context: 그는 요리 실력을 향상시키고 싶은 모든 사람들을 위한 초급 수업을 가르칠 예정이다."
]

# 실제 예측을 원하는 컨텍스트
current_context = "클래스가"


print("--- 전체 프롬프트 ---")
print(prompt_template)
print("---------------------\n")

# 토큰화
# Llama 3는 왼쪽 패딩을 사용하지 않는 것이 일반적입니다 (학습 시).
# `padding_side='left'`는 배치 추론 시 필요할 수 있지만, 단일 프롬프트에서는 불필요합니다.
inputs = tokenizer(prompt_template, return_tensors="pt", truncation=True).to(model.device)

# 생성 파라미터 설정
# max_new_tokens: "Thought:" 와 "Next Word:" 내용을 포함할 충분한 길이로 설정
# Llama 3 모델은 특정 EOS 토큰을 가질 수 있으므로, tokenizer.eos_token_id를 사용하는 것이 좋습니다.
generation_kwargs = {
    "max_new_tokens": 3000,          # 생성할 최대 토큰 수 (Thought + Next Word)
    "do_sample": True,             # 샘플링 사용 여부
    "temperature": 0.6,            # 낮을수록 결정적, 높을수록 다양 (Instruct 모델은 낮은 값 선호)
    "top_p": 0.9,                  # 누적 확률 p 이상의 토큰만 고려
    "eos_token_id": tokenizer.eos_token_id, # 명시적으로 EOS 토큰 ID 설정
    "pad_token_id": tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id, # pad_token_id가 없으면 eos_token_id 사용
}

print("모델 생성 중...")
try:
    with torch.no_grad(): # 추론 시에는 기울기 계산이 필요 없습니다.
        outputs = model.generate(**inputs, **generation_kwargs)

    # 생성된 텍스트만 디코딩 (입력 프롬프트 제외)
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n--- 모델 생성 결과 (Thought & Next Word) ---")
    print(generated_text)
    print("--------------------------------------------")

except Exception as e:
    print(f"텍스트 생성 중 오류 발생: {e}")
