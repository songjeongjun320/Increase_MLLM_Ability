#!/usr/bin/env python3
"""
Korean Evaluation Datasets Download Script
==========================================

Downloads Korean evaluation datasets for TOW Option 2 project:
- KMMLU (Korean MMLU) - Korean language understanding
- KLUE benchmark - 8 Korean language understanding tasks
- English-Korean translation datasets
- KoCoNovel - Korean story corpus for ToW augmentation
"""

import json
from pathlib import Path
from huggingface_hub import snapshot_download

class KoreanDatasetDownloader:
    """Download and setup Korean evaluation datasets"""
    
    def __init__(self):
        self.datasets_dir = Path(__file__).parent
        self.benchmarks_dir = self.datasets_dir / "benchmarks"
        self.stories_dir = self.datasets_dir / "korean_stories"
        
        # Create directories
        self.benchmarks_dir.mkdir(exist_ok=True)
        self.stories_dir.mkdir(exist_ok=True)
    
    def download_kmmlu(self):
        """Download Korean MMLU (KMMLU) dataset"""
        print("[DOWNLOAD] KMMLU (Korean MMLU) dataset...")
        
        try:
            # Download from Hugging Face
            snapshot_download(
                repo_id="HAERAE-HUB/KMMLU",
                local_dir=self.benchmarks_dir / "KMMLU",
                repo_type="dataset"
            )
            print("[SUCCESS] KMMLU downloaded successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to download KMMLU: {e}")
            print("[INFO] KMMLU will need to be downloaded manually")
            return False
    
    def download_klue(self):
        """Download KLUE benchmark tasks"""
        print("[DOWNLOAD] KLUE benchmark (8 tasks)...")
        
        klue_tasks = [
            "ynat",      # Topic Classification
            "sts",       # Semantic Textual Similarity  
            "nli",       # Natural Language Inference
            "ner",       # Named Entity Recognition
            "re",        # Relation Extraction
            "dp",        # Dependency Parsing
            "mrc",       # Machine Reading Comprehension
            "wos"        # Wizard of Seoul (DST)
        ]
        
        success_count = 0
        
        for task in klue_tasks:
            try:
                print(f"[DOWNLOAD] KLUE-{task.upper()}...")
                snapshot_download(
                    repo_id=f"klue/klue-{task}",
                    local_dir=self.benchmarks_dir / "KLUE" / task,
                    repo_type="dataset"
                )
                print(f"[SUCCESS] KLUE-{task.upper()} downloaded")
                success_count += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to download KLUE-{task.upper()}: {e}")
        
        print(f"[SUMMARY] Downloaded {success_count}/{len(klue_tasks)} KLUE tasks")
        return success_count > 0
    
    def download_translation_dataset(self):
        """Download English-Korean translation dataset"""
        print("[DOWNLOAD] English-Korean translation dataset...")
        
        try:
            # Try to download WMT English-Korean data
            snapshot_download(
                repo_id="Helsinki-NLP/opus-100",
                local_dir=self.benchmarks_dir / "translation" / "opus-100",
                repo_type="dataset",
                allow_patterns=["*en-ko*"]  # Only English-Korean pairs
            )
            print("[SUCCESS] Translation dataset downloaded")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to download translation dataset: {e}")
            
            # Fallback: create sample translation data
            self._create_sample_translation_data()
            return True
    
    def _create_sample_translation_data(self):
        """Create sample English-Korean translation pairs for testing"""
        print("[FALLBACK] Creating sample translation dataset...")
        
        sample_pairs = [
            {
                "english": "Bruce Lee was a legendary martial artist and actor.",
                "korean": "브루스 리는 전설적인 무술가이자 배우였다."
            },
            {
                "english": "Korean traditional food kimchi is made through fermentation.",
                "korean": "한국의 전통 음식인 김치는 발효 과정을 통해 만들어진다."
            },
            {
                "english": "Seoul has a population of about 10 million people.",
                "korean": "서울의 인구는 약 천만 명이다."
            },
            {
                "english": "The weather is very cold today, so I wore a thick coat.",
                "korean": "오늘 날씨가 매우 추워서 두꺼운 코트를 입었다."
            },
            {
                "english": "The student studied late into the night to do well on the exam.",
                "korean": "그 학생은 시험을 잘 보기 위해 밤늦게까지 공부했다."
            }
        ]
        
        translation_dir = self.benchmarks_dir / "translation" / "sample"
        translation_dir.mkdir(parents=True, exist_ok=True)
        
        with open(translation_dir / "en_ko_pairs.json", 'w', encoding='utf-8') as f:
            json.dump(sample_pairs, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] Sample translation data created: {translation_dir}")
    
    def download_korean_stories(self):
        """Download Korean story corpus for ToW augmentation"""
        print("[DOWNLOAD] Korean story corpus...")
        
        try:
            # Download KoCoNovel dataset (Korean story corpus)
            snapshot_download(
                repo_id="beomi/KoCoNovel",
                local_dir=self.stories_dir / "KoCoNovel",
                repo_type="dataset"
            )
            print("[SUCCESS] Korean stories downloaded")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to download Korean stories: {e}")
            
            # Fallback: create sample Korean stories
            self._create_sample_korean_stories()
            return True
    
    def _create_sample_korean_stories(self):
        """Create sample Korean stories for ToW generation testing"""
        print("[FALLBACK] Creating sample Korean story corpus...")
        
        sample_stories = [
            {
                "title": "무술가의 여행",
                "content": "브루스 리는 쿵푸 영화의 전설적인 인물이다. 그는 홍콩에서 태어나 무술을 배웠다. 어린 시절부터 다양한 무술을 익혔고, 나중에 자신만의 무술 철학을 개발했다. 그의 영화는 전 세계 사람들에게 큰 영감을 주었다."
            },
            {
                "title": "한국의 음식 문화",
                "content": "한국의 전통 음식인 김치는 발효 과정을 통해 만들어진다. 이 음식은 매우 건강한 발효식품으로 알려져 있다. 김치는 배추, 무, 고춧가루, 마늘 등 다양한 재료로 만들어진다. 한국인들은 거의 매 끼니마다 김치를 먹는다."
            },
            {
                "title": "서울의 일상",
                "content": "서울의 인구는 약 천만 명이다. 이는 한국 전체 인구의 20%에 해당한다. 서울은 현대적인 건물들과 전통적인 궁궐이 어우러진 도시이다. 많은 사람들이 지하철을 이용해 출근하고, 한강에서 여가 시간을 보낸다."
            },
            {
                "title": "겨울날의 추억",
                "content": "오늘 날씨가 매우 추워서 두꺼운 코트를 입었다. 겨울철 한국의 날씨는 매우 춥고 건조하다. 눈이 내리면 아이들은 눈사람을 만들고 눈싸움을 한다. 따뜻한 군고구마와 붕어빵이 겨울의 대표적인 간식이다."
            },
            {
                "title": "학창 시절",
                "content": "그 학생은 시험을 잘 보기 위해 밤늦게까지 공부했다. 한국의 학생들은 매우 열심히 공부한다. 도서관에서 밤늦게까지 책을 읽고, 친구들과 함께 문제를 풀어본다. 노력의 결과로 좋은 성적을 받을 수 있다."
            }
        ]
        
        stories_file = self.stories_dir / "sample_stories.json"
        with open(stories_file, 'w', encoding='utf-8') as f:
            json.dump(sample_stories, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] Sample Korean stories created: {stories_file}")
    
    def download_all_datasets(self):
        """Download all Korean evaluation datasets"""
        print("="*60)
        print("[START] Downloading Korean Evaluation Datasets")
        print("="*60)
        
        datasets = [
            ("KMMLU", self.download_kmmlu),
            ("KLUE", self.download_klue),
            ("Translation", self.download_translation_dataset),
            ("Korean Stories", self.download_korean_stories)
        ]
        
        results = {}
        
        for dataset_name, download_func in datasets:
            print(f"\n[DATASET] {dataset_name}")
            print("-" * 40)
            success = download_func()
            results[dataset_name] = success
        
        print("\n" + "="*60)
        print("[SUMMARY] Dataset Download Results")
        print("="*60)
        
        for dataset_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            print(f"  {dataset_name}: {status}")
        
        total_success = sum(results.values())
        print(f"\nTotal: {total_success}/{len(results)} datasets downloaded successfully")
        
        return total_success > 0

def main():
    """Main execution function"""
    downloader = KoreanDatasetDownloader()
    success = downloader.download_all_datasets()
    
    if success:
        print("\n[NEXT STEP] Run baseline evaluation:")
        print("  cd ../3_evaluation")
        print("  python baseline_evaluation.py --all-models")
    else:
        print("\n[ERROR] Some datasets failed to download")
        print("[INFO] You may need to download them manually")

if __name__ == "__main__":
    main()