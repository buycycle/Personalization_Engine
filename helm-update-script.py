import yaml
import argparse
import sys

def update_yaml_tag(file_path, image_tag):
    try:
        # Load YAML file
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        # Update the tag for all meta_name entries
        updated = False
        for version in data.get("api", {}).get("versions", []):
            if "meta_name" in version:
                version["tag"] = image_tag
                updated = True
                print(f"Updated 'tag' to '{image_tag}' for 'meta_name: {version['meta_name']}'")

        if not updated:
            print("No 'meta_name' entries found to update.")

        # Save updated YAML back to the file
        with open(file_path, "w") as file:
            yaml.safe_dump(data, file, default_flow_style=False)

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Update the 'tag' value in a YAML file.")
    parser.add_argument("file_path", help="Path to the YAML file")
    parser.add_argument("image_tag", help="New image tag to set")

    args = parser.parse_args()

    # Call the function with arguments
    update_yaml_tag(args.file_path, args.image_tag)
