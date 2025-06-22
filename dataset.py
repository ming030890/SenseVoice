import os
import json
import zipfile

def create_zip_from_jsonl(jsonl_file, output_zip):
    source_files = set()
    
    # Read JSONL and extract source paths
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            source_path = data.get("source")
            if source_path and os.path.exists(source_path):
                source_files.add(source_path)
            else:
                print(f"Warning: File not found: {source_path}")

    # Create ZIP file with all source files
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for file_path in source_files:
            zipf.write(file_path, arcname=os.path.basename(file_path))
    print(f"Created zip: {output_zip} with {len(source_files)} files.")

def main():
    for split in ['cantonese/dev', 'cantonese/test', 'cantonese/train']:
        jsonl_path = f"{split}.jsonl"
        zip_path = f"{split}.zip"
        if os.path.exists(jsonl_path):
            create_zip_from_jsonl(jsonl_path, zip_path)
        else:
            print(f"Skipping {jsonl_path}, file does not exist.")

if __name__ == "__main__":
    main()
