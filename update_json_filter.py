import json

def update_json_filter(file_path, target_id="1_35988", min_length=36):
    """
    기존 JSON 파일에서 특정 ID 이후의 데이터만 필터링하여 
    원본 파일을 업데이트하는 함수
    """
    # 원본 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # target_id 찾기
    target_index = -1
    for i, item in enumerate(data):
        if item.get('id') == target_id:
            target_index = i
            break
    
    if target_index == -1:
        print(f"ID '{target_id}' not found.")
        return
    
    # target_id 이전 데이터는 그대로 유지
    before_data = data[:target_index + 1]  # target_id까지 포함
    
    # target_id 이후 데이터 필터링
    after_data = data[target_index + 1:]
    filtered_after_data = []
    removed_count = 0
    
    for item in after_data:
        sentence = item.get('sentence', '')
        if len(sentence) > min_length:
            filtered_after_data.append(item)
        else:
            removed_count += 1
    
    # 최종 데이터 = 이전 데이터 + 필터링된 이후 데이터
    final_data = before_data + filtered_after_data
    
    # 원본 파일 업데이트
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    
    print("File updated successfully:")
    print(f"- Original total: {len(data)} items")
    print(f"- Before ID '{target_id}': {len(before_data)} items (kept)")
    print(f"- After ID '{target_id}': {len(after_data)} items")
    print(f"- Removed (<=36 chars): {removed_count} items")
    print(f"- Final total: {len(final_data)} items")

if __name__ == "__main__":
    file_path = "4_tow_generation/org_data/kornli_kobest-kostrategyqa.json"
    update_json_filter(file_path)