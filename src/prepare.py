import pandas as pd
import sys
import os


def main():
    if len(sys.argv) != 3:
        print("Usage: python src/prepare.py <input_file> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file)

    print("Cleaning data...")
    df = df.dropna(subset=["tweet", "label"])

    output_file = os.path.join(output_dir, "train.csv")
    print(f"Saving prepared data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
