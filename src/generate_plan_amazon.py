import json
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Configuration
MODEL_NAME = "gpt2-medium"      
NUM_BUCKETS = 20                
MAX_MODEL_CTX = 1024            

def main():
    print(f"1. Loading Amazon Reviews dataset...")
    dataset = load_dataset("amazon_polarity", split="test[:2000]") 
    
    print(f"2. Tokenizing 2000 reviews to find lengths...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    lengths = []
    for text in dataset["content"]:
        token_ids = tokenizer.encode(text, truncation=True, max_length=MAX_MODEL_CTX)
        lengths.append(len(token_ids))

    print(f"   - Found {len(lengths)} reviews.")
    print(f"   - Min length: {min(lengths)}")
    print(f"   - Max length: {max(lengths)}")
    print(f"   - Avg length: {np.mean(lengths):.1f}")

    # 3. Select 20 representative "buckets"
    percentiles = np.linspace(0, 100, NUM_BUCKETS + 1)[1:] 
    bucket_lengths = np.percentile(lengths, percentiles)
    
    # FIX: Explicitly cast numpy int64 to python int
    unique_buckets = sorted(list(set(int(x) for x in bucket_lengths)))
    
    unique_buckets = [x for x in unique_buckets if x > 1]

    print(f"\n3. Generated {len(unique_buckets)} unique context buckets from data distribution:")
    print(unique_buckets)

    plan = {
        "contexts": unique_buckets,
        "source": "amazon_polarity",
        "notes": f"Generated from {len(lengths)} real review lengths."
    }

    with open("single_round_plan.json", "w") as f:
        json.dump(plan, f, indent=2)

    print("\n✅ Saved to single_round_plan.json")
    print("Now run step6_measure.py with --iters 100")

if __name__ == "__main__":
    main()