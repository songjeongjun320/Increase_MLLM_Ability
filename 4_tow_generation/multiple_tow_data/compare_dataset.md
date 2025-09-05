============================================================
ANALYSIS RESULTS
============================================================
Total items processed: 5,954
Items with hCoT tokens: 5,953
Items without hCoT tokens: 1

hCoT Token Counts:
Total <hCoT> tokens: 72,975
Total </hCoT> tokens: 72,975
Average hCoT pairs per item: 12.26

Context Length Before First hCoT Analysis:
Average words before first hCoT: 66.40
Ratio (words before first hCoT / total words): 0.075

Percentiles for Context Length Before First hCoT:
  10th percentile: 13 words
  25th percentile: 21 words
  50th percentile: 36 words
  75th percentile: 65 words
  90th percentile: 125 words
  Min context length before first hCoT: 1 words
  Max context length before first hCoT: 4860 words

Context Length Distribution Before First hCoT:
  0-5 words context: 85 items (1.4%)
  6-10 words context: 289 items (4.9%)
  11-20 words context: 1,041 items (17.5%)
  21-50 words context: 2,439 items (41.0%)
  51-100 words context: 1,254 items (21.1%)
  101+ words context: 845 items (14.2%)

Token Length Analysis:
Average estimated tokens per completion: 1172.32
Min estimated tokens: 27
Max estimated tokens: 71463

Token Length Percentiles:
  10th percentile: 227 tokens
  25th percentile: 340 tokens
  50th percentile: 581 tokens
  75th percentile: 1103 tokens
  90th percentile: 2356 tokens

Token Length Distribution (Dynamic Categories):
  0-100 tokens: 48 items (0.8%)
  101-200 tokens: 389 items (6.5%)
  201-500 tokens: 2,105 items (35.4%)
  501-1000 tokens: 1,749 items (29.4%)
  1001-1500 tokens: 636 items (10.7%)
  1501-2000 tokens: 314 items (5.3%)
  2001-2500 tokens: 163 items (2.7%)
  2501-3000 tokens: 124 items (2.1%)
  3001-3500 tokens: 76 items (1.3%)
  3501-4000 tokens: 53 items (0.9%)
  4001-4500 tokens: 46 items (0.8%)
  4501-5000 tokens: 37 items (0.6%)
  5001-5500 tokens: 28 items (0.5%)
  5501-6000 tokens: 18 items (0.3%)
  6001-6500 tokens: 22 items (0.4%)
  6501-7000 tokens: 14 items (0.2%)
  7001-7500 tokens: 18 items (0.3%)
  7501-8000 tokens: 13 items (0.2%)
  8001-8500 tokens: 19 items (0.3%)
  8501-9000 tokens: 6 items (0.1%)

Training Recommendations:
  Maximum token length in dataset: 71463
  Average token length: 1172
  ⚠️  Warning: Some samples exceed 4000 tokens. Consider:
     - Using models with larger context windows (8K+)
     - Truncating very long samples
     - Implementing dynamic batching

Memory Estimation (rough):
  For batch size 1: ~142926 tokens per sample
  For batch size 4: ~571704 tokens per batch
  For batch size 8: ~1143408 tokens per batch

============================================================
ANALYSIS RESULTS
============================================================
Total items processed: 44,127
Items with ToW tokens: 44,127
Items without ToW tokens: 0

ToW Token Counts:
Total <ToW> tokens: 129,093
Total </ToW> tokens: 129,093
Average ToW pairs per item: 2.93

Context Length Before First ToW Analysis:
Average words before first ToW: 18.39
Average total words per completion: 327.04
Ratio (words before first ToW / total words): 0.056

Percentiles for Context Length Before First ToW:
  10th percentile: 12 words
  25th percentile: 13 words
  50th percentile: 16 words
  75th percentile: 20 words
  90th percentile: 25 words
  Min context length before first ToW: 11 words
  Max context length before first ToW: 359 words

Context Length Distribution Before First ToW:
  0-5 words context: 0 items (0.0%)
  6-10 words context: 0 items (0.0%)
  11-20 words context: 34,076 items (77.2%)
  21-50 words context: 9,508 items (21.5%)
  51-100 words context: 192 items (0.4%)
  101+ words context: 351 items (0.8%)

Token Length Analysis:
Average estimated tokens per completion: 434.44
Min estimated tokens: 97
Max estimated tokens: 4813

Token Length Percentiles:
  10th percentile: 172 tokens
  25th percentile: 283 tokens
  50th percentile: 409 tokens
  75th percentile: 558 tokens
  90th percentile: 694 tokens

Token Length Distribution (Dynamic Categories):
  0-100 tokens: 3 items (0.0%)
  101-200 tokens: 5,591 items (12.7%)
  201-500 tokens: 24,100 items (54.6%)
  501-1000 tokens: 13,710 items (31.1%)
  1001-1500 tokens: 585 items (1.3%)
  1501-2000 tokens: 88 items (0.2%)
  2001-2500 tokens: 29 items (0.1%)
  2501-3000 tokens: 16 items (0.0%)
  3001-3500 tokens: 2 items (0.0%)
  3501-4000 tokens: 1 items (0.0%)
  4001-4500 tokens: 1 items (0.0%)
  4501-4813 tokens: 1 items (0.0%)
  4814+ tokens: 0 items (0.0%)

Training Recommendations:
  Maximum token length in dataset: 4813
  Average token length: 434
  ⚠️  Warning: Some samples exceed 4000 tokens. Consider:
     - Using models with larger context windows (8K+)
     - Truncating very long samples
     - Implementing dynamic batching

Memory Estimation (rough):
  For batch size 1: ~9626 tokens per sample
  For batch size 4: ~38504 tokens per batch
  For batch size 8: ~77008 tokens per batch