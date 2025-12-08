import json
import glob
import os

def analyze_review_lengths(input_dir):
    """
    Analyze the lengths of reviews in stringified_*.json files.
    """
    pattern = os.path.join(input_dir, "stringified_*.json")
    input_files = glob.glob(pattern)
    
    if not input_files:
        print(f"No stringified_*.json files found in {input_dir}")
        return
    
    all_lengths = []
    total_cases = 0
    file_stats = []
    
    for input_file in input_files:
        print(f"Processing {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        file_cases = len(data)
        file_reviews = 0
        file_lengths = []
        cases_with_empty_input = 0
        
        for item in data:
            total_cases += 1
            input_list = item.get("input", [])
            if not input_list:
                cases_with_empty_input += 1
            
            num_reviews = len(input_list)
            file_reviews += num_reviews
            for review in input_list:
                length = len(review)
                all_lengths.append(length)
                file_lengths.append(length)
        
        avg_reviews_per_case = file_reviews / file_cases if file_cases > 0 else 0
        file_stats.append((os.path.basename(input_file), file_cases, file_reviews, avg_reviews_per_case, cases_with_empty_input))
    
    if not all_lengths:
        print("No reviews found.")
        return
    
    total_reviews = len(all_lengths)
    average_length = sum(all_lengths) / total_reviews
    count_above_200 = sum(1 for length in all_lengths if length > 200)
    proportion_above_200 = count_above_200 / total_reviews * 100
    count_above_500 = sum(1 for length in all_lengths if length > 500)
    proportion_above_500 = count_above_500 / total_reviews * 100
    
    print("\nPer-file Statistics:")
    for filename, cases, reviews, avg, empty_input in file_stats:
        print(f"{filename}: {cases} cases, {reviews} reviews, avg {avg:.2f} reviews/case, {empty_input} cases with empty input")
    
    print("\nOverall Analysis Results:")
    print(f"Total cases: {total_cases}")
    print(f"Total reviews: {total_reviews}")
    print(f"Average reviews per case: {total_reviews / total_cases:.2f}")
    print(f"Average review length: {average_length:.2f} characters")
    print(f"Number of reviews > 200 characters: {count_above_200}")
    print(f"Proportion of reviews > 200 characters: {proportion_above_200:.2f}%")
    print(f"Number of reviews > 500 characters: {count_above_500}")
    print(f"Proportion of reviews > 500 characters: {proportion_above_500:.2f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze review lengths in stringified JSON files")
    parser.add_argument("--input-dir", default="dataset/data/raw", help="Directory containing stringified JSON files")
    args = parser.parse_args()
    
    analyze_review_lengths(args.input_dir)