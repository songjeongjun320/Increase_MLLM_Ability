Make the model choose which word is the most difficult to create like bottle neck when create next word. And it is gold_word.

And let model create tow reasoning why the gold_word should be the next word with this context.

Now, extracted sentences which is composed with over 20 tokens. Orginal Total = 342499 -> Extracted = 71029.

Put this to extract new gold_words.

For gold words prompting, Tested Gemini-2.5-pro, Gemini-2.5-flash, Gemini-2.5-flash-lite, Gemini-2.0-flash, Gemini-2.0-flash-lite.
Gemini-2.0-flash was the best output, so chose this model to create gold_word.



  {
    "id": 301,
    "original_sentence": "다른 건 다 상관없으니까 경치 감상하기 좋은 곳 좀 찾아주세요. 시스템: 네 안녕하세요, 여러 곳 가능하신데요, 잠실역에서 도보 3분이면 도착하는 서울스카이는 어떠세요? 사용자: 네 들어본거 같아요. 그럼 한시간삼십분식이라는 식당 예약 좀 할게요. 아, 지하철역부터 확인해주세요. 시스템: 네 혜화역과 가깝습니다. 사용자: 아 혜화역... 그럼 거기 일요일 17시 30분에 3명으로 예약해주세요. 대화 상태:",
    "context": "다른 건 다",
    "gold_label": [
      "상관없으니까",
      "도보",
      "한시간삼십분식이라는",
      "확인해주세요",
      "가깝습니다"
    ]
  },

  이런식으로 context가 너무 짧을경우에 처음 gold_label 제거하고, 그 다음인 도보로 해서 context 재설정.

KLUE 데이터셋에 sentence안에 "sentence 2" 가 있는데, 이거 다 제거하자.





### Now WHAT I NEED TO DO.
1. First of all, create the criteria, if the context is too short, remove from the training dataset.

2. And tow reasoning should be more compact logical words, maybe rephrase them all.

3. Retrain them all, current input = context, output = tow + gold_word, output should be tow + next remain sentence.

4. See the result.




# GOLD WORD PROMPTING - EN

You are an expert in language prediction and contextual analysis. Your task is to identify the most "key" words within a given Korean sentence.

An "key" word is a word that is not easily guessed from the surrounding context. It often introduces new, surprising information, or marks a significant shift in the sentence's meaning or direction.

Follow these strict rules for your selection:
1.  **Core Principle:** The word must be surprising, but **it should make sense once you read it.** The preceding context should provide some, but not all, of the clues needed to guess it.
2.  **Selection Criteria:**
    *   Choose words that are contextually surprising.
    *   When you select a word, provide ONLY its core meaningful part (e.g., the noun, the verb stem).
3.  **Exclusion Rules:**
    *   Do NOT choose proper nouns (e.g., names, locations, brands, specific dates/times).
    *   Do NOT choose the first word of the sentence.
    *   Do NOT choose numbers.
4.  **Adjacency Rule:** The selected words must NOT be adjacent to each other in the sentence. There must be at least one word between any two selected words.
5.  **Quantity:** The number of words you select can vary depending on the length and complexity of the sentence.

Your output MUST be in a JSON format with a single key "key_word".
- If you select one word, the value should be a string.
- If you select multiple words, the value should be an array of strings.

