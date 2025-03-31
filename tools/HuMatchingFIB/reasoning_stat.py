import json
import csv
from collections import defaultdict
import os

def analyze_reasoning_stats(input_file_path, output_dir):
    total_lines = 0
    total_blanks = 0
    total_reasoning_segments = 0
    total_review_segments = 0
    total_introduction_segments = 0
    total_conclusion_segments = 0

    dimension1_counts = defaultdict(int)
    dimension2_counts = defaultdict(int)
    dimension3_counts = defaultdict(int)
    dimension4_counts = defaultdict(int)

    dimension1_distribution_simple_assertion = defaultdict(int)
    dimension1_distribution_complex_thought = defaultdict(int)
    
    dimension1_distribution_only_current_blank = defaultdict(int)
    dimension1_distribution_Modify_Previous_Blanks = defaultdict(int)
    dimension1_distribution_Current_Blank_and_Consecutive_Blank = defaultdict(int)
    
    dimension1_distribution_contain_language_transfer = defaultdict(int)
    dimension1_distribution_no_language_transfer = defaultdict(int)
    
    specific_condition_count = 0

    with open(input_file_path, 'r', encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line)
                total_lines += 1
                total_blanks += len(data['std_ans'])            
                
                if isinstance(data, dict) and "segment_dim" in data and isinstance(data["segment_dim"], dict):
                    segments = data["segment_dim"].get("segments", [])
                    
                    # Ensure segments is a list
                    if isinstance(segments, list):
                        for segment in segments:
                            if isinstance(segment, dict) and "category" in segment:
                                if segment["category"] == "Reasoning":
                                    total_reasoning_segments += 1
                                elif segment["category"] == "Review":
                                    total_review_segments += 1
                                elif segment["category"] == "Introduction":
                                    total_introduction_segments += 1
                                elif segment["category"] == "Conclusion":
                                    total_conclusion_segments += 1
                                    
                                if "Dimension1" in segment and segment["category"] == "Reasoning":
                                    dimension1_value = segment["Dimension1"]
                                    dimension1_counts[dimension1_value] += 1
                                
                                if "Dimension2" in segment and segment["category"] == "Reasoning":
                                    dimension2_value = segment["Dimension2"]
                                    dimension2_counts[dimension2_value] += 1
                                    
                                    if dimension2_value == "Simple Assertion":
                                        if "Dimension1" in segment:
                                            dimension1_distribution_simple_assertion[segment["Dimension1"]] += 1
                                    if dimension2_value == "Complex Thought":
                                        if "Dimension1" in segment:
                                            dimension1_distribution_complex_thought[segment["Dimension1"]] += 1

                                if "Dimension3" in segment and segment["category"] == "Reasoning":
                                    dimension3_value = segment["Dimension3"]
                                    dimension3_counts[dimension3_value] += 1
                                    
                                    if dimension3_value == "Only Current Blank" and "Dimension1" in segment:
                                        dimension1_distribution_only_current_blank[segment["Dimension1"]] += 1
                                    if dimension3_value == "Modify Previous Blanks" and "Dimension1" in segment:
                                        dimension1_distribution_Modify_Previous_Blanks[segment["Dimension1"]] += 1
                                    if dimension3_value == "Current Blank and Consecutive Blank" and "Dimension1" in segment:
                                        dimension1_distribution_Current_Blank_and_Consecutive_Blank[segment["Dimension1"]] += 1
                                        
                                if "Dimension4" in segment and segment["category"] == "Reasoning":
                                    dimension4_value = segment["Dimension4"]
                                    dimension4_counts[dimension4_value] += 1
                                    
                                    if dimension4_value == "Contains Language Transfer" and "Dimension1" in segment:
                                        dimension1_distribution_contain_language_transfer[segment["Dimension1"]] += 1
                                    if dimension4_value == "No Language Transfer" and "Dimension1" in segment:
                                        dimension1_distribution_no_language_transfer[segment["Dimension1"]] += 1
                                    
                                if segment.get("Dimension2") == "Simple Assertion" and segment.get("Dimension1") == "Completely Correct":
                                    specific_condition_count += 1                                
                            else:
                                print(f"Unexpected segment format at line {total_lines}:")
                    else:
                        print(f"Unexpected segments format at line {total_lines}: ")
                else:
                    print(f"Unexpected data format at line {total_lines}: ")
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {total_lines}: {e}")
                continue

    avg_reasoning_segments = total_reasoning_segments / total_lines if total_lines > 0 else 0
    avg_review_segments = total_review_segments / total_lines if total_lines > 0 else 0
    avg_intro_segments = total_introduction_segments / total_lines if total_lines > 0 else 0
    avg_conclusion_segments = total_conclusion_segments / total_lines if total_lines > 0 else 0
    avg_blanks = total_blanks / total_lines if total_lines > 0 else 0

    os.makedirs(output_dir, exist_ok=True)
    
    json_output = {
        'statistics': {
            'Total Lines': total_lines,
            'Total Reasoning Segments': total_reasoning_segments,
            'Average Blanks per Question': round(avg_blanks, 2),
            'Average Introduction Segments per Line': round(avg_intro_segments, 2),
            'Average Reasoning Segments per Line': round(avg_reasoning_segments, 2),
            'Average Review Segments per Line': round(avg_review_segments, 2),
            'Average Conclusion Segments per Line': round(avg_conclusion_segments, 2)
        },
        'dimension_analysis': {
            'Dimension1': dict(dimension1_counts),
            'Dimension2': dict(dimension2_counts),
            'Dimension3': dict(dimension3_counts),
            'Dimension4': dict(dimension4_counts),
            'Dimension1_Simple_Assertion': dict(dimension1_distribution_simple_assertion),
            'Dimension1_Complex_Thought': dict(dimension1_distribution_complex_thought),
            'Dimension1_Only_Current_Blank': dict(dimension1_distribution_only_current_blank),
            'Dimension1_Modify_Previous_Blanks': dict(dimension1_distribution_Modify_Previous_Blanks),
            'Dimension1_Current_Blank_and_Consecutive_Blank': dict(dimension1_distribution_Current_Blank_and_Consecutive_Blank),
            'Dimension1_Contain_Language_Transfer': dict(dimension1_distribution_contain_language_transfer),
            'Dimension1_No_Language_Transfer': dict(dimension1_distribution_no_language_transfer)
        }
    }
    
    json_path = os.path.join(output_dir, 'analysis_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=4, ensure_ascii=False)
    
    # Simple complex count
    csv_path = os.path.join(output_dir, 'dimension1_simple_complex_distribution.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Value', 'Simple Assertion', 'Complex Thought'])
        
        all_dim1_values = set(dimension1_distribution_simple_assertion.keys()) | set(dimension1_distribution_complex_thought.keys())
        for value in sorted(all_dim1_values):
            simple_count = dimension1_distribution_simple_assertion.get(value, 0)
            complex_count = dimension1_distribution_complex_thought.get(value, 0)
            writer.writerow([value, simple_count, complex_count])
    
    print(f"Analysis complete. Results saved to:")
    print(f"- JSON file: {json_path}")
    print(f"- CSV file: {csv_path}")
    
    # Blank count
    csv_path = os.path.join(output_dir, 'dimension1_blank_distribution.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Value', 'Only Current Blank', 'Modify Previous Blanks', 'Current Blank and Consecutive Blank'])
        
        all_dim1_values = set(dimension1_distribution_only_current_blank.keys()) | set(dimension1_distribution_Modify_Previous_Blanks.keys()) | set(dimension1_distribution_Current_Blank_and_Consecutive_Blank.keys())
        for value in sorted(all_dim1_values):
            only_blank_count = dimension1_distribution_only_current_blank.get(value, 0)
            modified_blanl_count = dimension1_distribution_Modify_Previous_Blanks.get(value, 0)
            current_consecutive_count = dimension1_distribution_Current_Blank_and_Consecutive_Blank.get(value, 0)
            writer.writerow([value, only_blank_count, modified_blanl_count, current_consecutive_count])
    
    print(f"Analysis complete. Results saved to:")
    print(f"- CSV file: {csv_path}")
    
    # Transfer count
    csv_path = os.path.join(output_dir, 'dimension1_lang_transfer_distribution.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Value', 'Contains Language Transfer', 'No Language Transfer'])
        
        all_dim1_values = set(dimension1_distribution_contain_language_transfer.keys()) | set(dimension1_distribution_no_language_transfer.keys())
        for value in sorted(all_dim1_values):
            tansfer_count = dimension1_distribution_contain_language_transfer.get(value, 0)
            no_transfer_count = dimension1_distribution_no_language_transfer.get(value, 0)
            writer.writerow([value, tansfer_count, no_transfer_count])
    
    print(f"Analysis complete. Results saved to:")
    print(f"- CSV file: {csv_path}")
    
    return json_output

if __name__ == "__main__":
    input_file = 'Your_file_with_segmented_classified_response.jsonl'
    output_dir = 'Your_desired_output_dir'
    analyze_reasoning_stats(input_file, output_dir)