#!/usr/bin/env python3
import csv
import os

def main():
    # File definitions
    old_file = "training_dataset_large.csv" # Your original 200k points
    new_file = "training_dataset.csv"       # Your new profiled + 20k points
    out_file = "combined_dataset.csv"

    # 1. Establish the Target Schema from the newest data
    if not os.path.exists(new_file):
        print(f"[ERROR] Could not find {new_file} to extract the schema.")
        return

    with open(new_file, 'r', newline='') as f:
        reader = csv.reader(f)
        headers = next(reader, [])
    
    print(f"[INFO] Merging into schema with {len(headers)} columns.")

    # 2. Initialize the combined file
    with open(out_file, 'w', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=headers)
        writer.writeheader()

        # 3. Process the Legacy 200k Data
        old_count = 0
        if os.path.exists(old_file):
            print(f"[INFO] Reading legacy data from {old_file}...")
            with open(old_file, 'r', newline='') as f_old:
                reader_old = csv.DictReader(f_old)
                for row in reader_old:
                    # Fill missing memory/PCIe columns with -1 for ML compatibility
                    for h in headers:
                        if h not in row:
                            row[h] = -1
                    writer.writerow(row)
                    old_count += 1
            print(f"  -> Processed {old_count} legacy rows with -1 padding.")
        else:
            print(f"[WARNING] {old_file} not found. Skipping legacy merge.")

        # 4. Process the New Profiled Data (including 20k stress test)
        new_count = 0
        print(f"[INFO] Merging new profiled data from {new_file}...")
        with open(new_file, 'r', newline='') as f_new:
            reader_new = csv.DictReader(f_new)
            for row in reader_new:
                # Ensure all headers are present (safety check)
                for h in headers:
                    if h not in row:
                        row[h] = -1
                writer.writerow(row)
                new_count += 1
        print(f"  -> Processed {new_count} new rows.")

    print(f"\n✅ Success! {old_count + new_count} rows saved to {out_file}")

if __name__ == "__main__":
    main()