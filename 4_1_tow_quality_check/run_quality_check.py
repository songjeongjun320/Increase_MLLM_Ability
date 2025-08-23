"""
Main script to run ToW dataset quality checking and adjustment
"""

import json
import logging
from pathlib import Path
from tow_quality_checker import ToWQualityChecker
from tow_quality_adjuster import ToWQualityAdjuster

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run complete quality check and adjustment pipeline"""
    # Initialize components
    checker = ToWQualityChecker()
    adjuster = ToWQualityAdjuster(checker)
    
    # Paths
    data_dir = Path("../4_tow_generation/tow_data")
    output_dir = Path(".")
    output_dir.mkdir(exist_ok=True)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Process each JSON file
    for json_file in data_dir.glob("*.json"):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {json_file.name}")
        logger.info(f"{'='*60}")
        
        try:
            # Load data
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} samples from {json_file.name}")
            
            # Step 1: Initial quality assessment
            logger.info("Step 1: Initial quality assessment...")
            initial_report = checker.generate_quality_report(
                data,
                output_path=output_dir / f"{json_file.stem}_initial_quality_report.txt"
            )
            print(initial_report)
            
            # Step 2: Quality adjustment
            logger.info("Step 2: Quality adjustment...")
            adjusted_data, stats = adjuster.adjust_dataset(
                data,
                remove_low_quality=True,
                min_quality_threshold=4  # Keep samples meeting at least 4/6 criteria
            )
            
            # Generate adjustment report
            adjustment_report = adjuster.generate_adjustment_report(
                stats,
                output_path=output_dir / f"{json_file.stem}_adjustment_report.txt"
            )
            print(adjustment_report)
            
            # Step 3: Save adjusted data
            output_file = output_dir / f"{json_file.stem}_quality_adjusted.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(adjusted_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Adjusted data saved to: {output_file}")
            
            # Step 4: Final quality assessment
            logger.info("Step 4: Final quality assessment...")
            final_report = checker.generate_quality_report(
                adjusted_data,
                output_path=output_dir / f"{json_file.stem}_final_quality_report.txt"
            )
            print(final_report)
            
            # Step 5: Generate high-quality filtered dataset
            high_quality_data = checker.filter_high_quality_samples(
                adjusted_data,
                min_criteria_met=5  # Very high quality: 5/6 criteria
            )
            
            if high_quality_data:
                high_quality_file = output_dir / f"{json_file.stem}_high_quality.json"
                with open(high_quality_file, 'w', encoding='utf-8') as f:
                    json.dump(high_quality_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"High-quality samples saved to: {high_quality_file}")
                logger.info(f"High-quality sample count: {len(high_quality_data)}")
            
            logger.info(f"✅ Successfully processed {json_file.name}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {json_file.name}: {str(e)}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info("Quality check and adjustment completed!")
    logger.info(f"{'='*60}")
    logger.info("Generated files:")
    logger.info("- *_initial_quality_report.txt: Initial quality assessment")
    logger.info("- *_adjustment_report.txt: Adjustment statistics")
    logger.info("- *_final_quality_report.txt: Final quality assessment")
    logger.info("- *_quality_adjusted.json: Quality-adjusted dataset")
    logger.info("- *_high_quality.json: High-quality filtered dataset")

if __name__ == "__main__":
    main()