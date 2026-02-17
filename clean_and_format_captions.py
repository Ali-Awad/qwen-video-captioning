import os
import json
import re
import argparse

# Default configuration
DEFAULT_CAPTIONS_DIR = 'Human_Captions'
DEFAULT_INPUT_DIR = 'input'
VIDEO_EXT = '.mp4'
JSON_SUFFIX = '_analysis.json'

def fix_json_content(content):
    """
    Attempts to fix common JSON syntax errors.
    """
    # Fix 1: Remove trailing commas before closing braces/brackets
    content = re.sub(r',(\s*[}\]])', r'\1', content)
    return content

def clean_orphans(captions_dir, input_dir):
    """
    Deletes JSON files that don't have a corresponding video file.
    """
    print(f"--- Checking for orphaned JSON files in {captions_dir} ---")
    deleted_count = 0
    if not os.path.exists(captions_dir) or not os.path.exists(input_dir):
        print(f"Error: Directories {captions_dir} or {input_dir} not found.")
        return

    for filename in os.listdir(captions_dir):
        if filename.endswith(JSON_SUFFIX):
            base_name = filename.replace(JSON_SUFFIX, '')
            video_path = os.path.join(input_dir, base_name + VIDEO_EXT)
            
            if not os.path.exists(video_path):
                file_to_delete = os.path.join(captions_dir, filename)
                try:
                    os.remove(file_to_delete)
                    print(f"Deleted orphan: {filename}")
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {filename}: {e}")
    
    print(f"Deleted {deleted_count} orphaned files.\n")

def process_files(captions_dir):
    """
    Iterates through files, fixes formatting, and removes metadata.
    """
    print(f"--- Processing files in {captions_dir} (Fix Format & Remove Metadata) ---")
    processed_count = 0
    fixed_count = 0
    error_count = 0
    
    keys_to_remove = ['usage_metadata', 'file_metadata']

    if not os.path.exists(captions_dir):
        print(f"Error: Directory {captions_dir} not found.")
        return

    for filename in os.listdir(captions_dir):
        if not filename.endswith(JSON_SUFFIX):
            continue
            
        file_path = os.path.join(captions_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 1. Try to load JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # 2. If load fails, try to fix syntax
                fixed_content = fix_json_content(content)
                try:
                    data = json.loads(fixed_content)
                    fixed_count += 1
                except json.JSONDecodeError as e:
                    print(f"Failed to fix JSON syntax in {filename}: {e}")
                    error_count += 1
                    continue

            # 3. Remove metadata
            modified = False
            for key in keys_to_remove:
                if key in data:
                    del data[key]
                    modified = True
            
            # Always write back to ensure consistent indentation/formatting
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            processed_count += 1

        except Exception as e:
            print(f"Unexpected error processing {filename}: {e}")
            error_count += 1

    print(f"Processed {processed_count} files.")
    print(f"Repaired syntax in {fixed_count} files.")
    print(f"Errors encountered: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up orphaned caption files and format JSONs.")
    parser.add_argument('--captions', default=DEFAULT_CAPTIONS_DIR, help=f"Directory containing caption JSON files (default: {DEFAULT_CAPTIONS_DIR})")
    parser.add_argument('--input', default=DEFAULT_INPUT_DIR, help=f"Directory containing input video files (default: {DEFAULT_INPUT_DIR})")
    
    args = parser.parse_args()
    
    clean_orphans(args.captions, args.input)
    process_files(args.captions)
