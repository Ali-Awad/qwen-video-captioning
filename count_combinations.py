import os
import json
import argparse
from collections import defaultdict

def count_combinations(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    counts = defaultdict(int)
    
    # Initialize all keys to 0
    keys = [
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    ]
    for k in keys:
        counts[k] = 0

    files = [f for f in os.listdir(directory) if f.endswith('_analysis.json')]
    
    print(f"Scanning {len(files)} files in {directory}...")
    
    error_count = 0
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            winter_weather = data.get('weather', {}).get('winter weather')
            hazardous_present = data.get('hazardous event', {}).get('present')
            
            # Ensure they are booleans or handle missing data appropriately
            if winter_weather is None:
                # print(f"Warning: 'winter weather' missing in {filename}")
                pass
            if hazardous_present is None:
                # print(f"Warning: 'hazardous event.present' missing in {filename}")
                pass
                
            # Only count if both are present (or treat None as distinct if needed, but here we skip or count as None)
            if winter_weather is not None and hazardous_present is not None:
                counts[(winter_weather, hazardous_present)] += 1
            
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {filename}")
            error_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            error_count += 1
            
    # Prepare table data
    print("\nResults:")
    print("-" * 55)
    print(f"| {'Winter Weather':<15} | {'Hazardous Event':<15} | {'Count':<10} |")
    print("-" * 55)
    
    # Sort for consistent output
    sorted_keys = sorted(counts.keys(), key=lambda x: (not x[0], not x[1])) # True first
    
    for winter, present in sorted_keys:
        print(f"| {str(winter):<15} | {str(present):<15} | {counts[(winter, present)]:<10} |")
        
    print("-" * 55)
    
    total = sum(counts.values())
    print(f"\nTotal files counted: {total}")
    if error_count > 0:
        print(f"Files with errors: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count combinations of weather and hazardous events in caption files.")
    parser.add_argument('directory', nargs='?', default='Human_Captions', help="Directory containing caption JSON files (default: Human_Captions)")
    
    args = parser.parse_args()
    
    count_combinations(args.directory)
