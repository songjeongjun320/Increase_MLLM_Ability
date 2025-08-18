#!/usr/bin/env python3
"""
New Korean Datasets Download Script
=====================================

Downloads additional Korean datasets:
- HAE_RAE_BENCH_1.0 - Korean evaluation benchmark
- Ko-StrategyQA - Korean strategy question answering
- kor_nli - Korean Natural Language Inference
"""

import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset

class NewKoreanDatasetDownloader:
    """Download new Korean evaluation datasets"""
    
    def __init__(self):
        self.datasets_dir = Path(__file__).parent
        self.new_datasets_dir = self.datasets_dir / "new_korean_datasets"
        self.new_datasets_dir.mkdir(exist_ok=True)
    
    def download_hae_rae_bench(self):
        """Download HAE_RAE_BENCH_1.0 dataset"""
        print("[DOWNLOAD] HAE_RAE_BENCH_1.0 dataset...")
        
        try:
            # Download from Hugging Face
            snapshot_download(
                repo_id="HAERAE-HUB/HAE_RAE_BENCH_1.0",
                local_dir=self.new_datasets_dir / "HAE_RAE_BENCH_1.0",
                repo_type="dataset"
            )
            print("[SUCCESS] HAE_RAE_BENCH_1.0 downloaded successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to download HAE_RAE_BENCH_1.0: {e}")
            return False
    
    def download_ko_strategy_qa(self):
        """Download Ko-StrategyQA dataset"""
        print("[DOWNLOAD] Ko-StrategyQA dataset...")
        
        try:
            # Load dataset using datasets library
            dataset = load_dataset("taeminlee/Ko-StrategyQA", "corpus")
            
            # Save as JSON
            output_dir = self.new_datasets_dir / "Ko-StrategyQA"
            output_dir.mkdir(exist_ok=True)
            
            # Save train split
            if "train" in dataset:
                train_data = []
                for item in dataset["train"]:
                    train_data.append(item)
                
                with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
                    json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            # Save test split if available
            if "test" in dataset:
                test_data = []
                for item in dataset["test"]:
                    test_data.append(item)
                
                with open(output_dir / "test.json", 'w', encoding='utf-8') as f:
                    json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            # Save validation split if available
            if "validation" in dataset:
                val_data = []
                for item in dataset["validation"]:
                    val_data.append(item)
                
                with open(output_dir / "validation.json", 'w', encoding='utf-8') as f:
                    json.dump(val_data, f, ensure_ascii=False, indent=2)
            
            print(f"[SUCCESS] Ko-StrategyQA downloaded to {output_dir}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to download Ko-StrategyQA: {e}")
            return False
    
    def download_kor_nli(self):
        """Download Korean NLI dataset"""
        print("[DOWNLOAD] Korean NLI dataset...")
        
        try:
            # Download from Hugging Face
            snapshot_download(
                repo_id="kakaobrain/kor_nli",
                local_dir=self.new_datasets_dir / "kor_nli",
                repo_type="dataset"
            )
            print("[SUCCESS] kor_nli downloaded successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to download kor_nli: {e}")
            return False
    
    def process_korean_texts_to_sentence_format(self):
        """Process Korean texts and save in sentence format"""
        print("[PROCESS] Converting Korean texts to sentence format...")
        
        # Sample Korean texts (like the example provided)
        korean_texts = [
            "C 여학교에서 교원 겸 기숙사 사감 노릇을 하는 B 여사라면 딱장대요 독신주의자요 찰진 야소군으로 유명하다.\n사십에 가까운 노처녀인 그는 주근깨투성이 얼굴이 처녀다운 맛이란 약에 쓰려도 찾을 수 없을 뿐인가, 시들고 거칠고 마르고 누렇게 뜬 품이 곰팡 슬은 굴비를 생각나게 한다.\n여러 겹주름이 잡힌 훨렁 벗겨진 이마라든지, 숱이 적어서 법대로 쪽지거나 틀어 올리지를 못하고 엉성하게 그냥 빗어넘긴 머리꼬리가 뒤통수에 염소 똥만 하게 붙은 것이라든지, 벌써 늙어가는 자취를 감출 길이 없었다.\n뾰족한 입을 앙다물고 돋보기 너머로 쌀쌀한 눈이 노릴 때엔 기숙생들이 오싹하고 몸서리를 치리만큼 그는 엄격하고 매서웠다.\n이 B 여사가 질겁을 하다시피 싫어하고 미워하는 것은 소위 '러브레터'였다.\n여학교 기숙사라면 으레 그런 편지가 많이 오는 것이지만 학교로도 유명하고 또 아름다운 여학생이 많은 탓인지 모르되 하루에도 몇 장씩 죽느니 사느니 하는 사랑 타령이 날아들어 왔었다.\n기숙생에게 오는 사신을 일일이 검토하는 터이니까 그따위 편지도 물론 B 여사의 손에 떨어진다.\n달짝지근한 사연을 보는 족족 그는 더할 수 없이 흥분되어서 얼굴이 붉으락푸르락, 편지 든 손이 발발 떨리도록 성을 낸다.\n아무 까닭 없이 그런 편지를 받은 학생이야말로 큰 재변이었다.\n하학하기가 무섭게 그 학생은 사감실로 불리어 간다.\n분해서 못 견디겠다는 사람 모양으로 쌔근쌔근하며 방안을 왔다 갔다 하던 그는, 들어오는 학생을 잡아먹을 듯이 노리면서 한 걸음 두 걸음 코가 맞닿을 만큼 바싹 다가들어서 딱 마주 선다.",
            "한국의 전통 음식인 김치는 발효 과정을 통해 만들어진다.",
            "서울의 인구는 약 천만 명이다.",
            "오늘 날씨가 매우 추워서 두꺼운 코트를 입었다.",
            "그 학생은 시험을 잘 보기 위해 밤늦게까지 공부했다."
        ]
        
        # Convert to sentence format
        sentence_data = []
        for idx, text in enumerate(korean_texts):
            sentence_data.append({
                "sentence": text,
                "id": f"{idx + 1}_0"
            })
        
        # Save to JSON file
        output_file = self.new_datasets_dir / "korean_sentences.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sentence_data, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] Korean sentences saved to {output_file}")
        return True
    
    def download_all_new_datasets(self):
        """Download all new Korean datasets"""
        print("="*60)
        print("[START] Downloading New Korean Datasets")
        print("="*60)
        
        datasets = [
            ("HAE_RAE_BENCH_1.0", self.download_hae_rae_bench),
            ("Ko-StrategyQA", self.download_ko_strategy_qa),
            ("kor_nli", self.download_kor_nli),
            ("Korean Sentences", self.process_korean_texts_to_sentence_format)
        ]
        
        results = {}
        
        for dataset_name, download_func in datasets:
            print(f"\n[DATASET] {dataset_name}")
            print("-" * 40)
            success = download_func()
            results[dataset_name] = success
        
        print("\n" + "="*60)
        print("[SUMMARY] New Dataset Download Results")
        print("="*60)
        
        for dataset_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {dataset_name}: {status}")
        
        total_success = sum(results.values())
        print(f"\nTotal: {total_success}/{len(results)} datasets processed successfully")
        
        return total_success > 0

def main():
    """Main execution function"""
    downloader = NewKoreanDatasetDownloader()
    success = downloader.download_all_new_datasets()
    
    if success:
        print(f"\n[SUCCESS] New datasets downloaded to: {downloader.new_datasets_dir}")
    else:
        print("\n[ERROR] Some datasets failed to download")

if __name__ == "__main__":
    main()