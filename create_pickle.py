import torch
import sys
import ast

# Check if the correct number of arguments are provided
if len(sys.argv) != 4:
    print("Usage: python create_pickle.py <image_size_tuple_string> <label_map_dict_string> <output_path>")
    sys.exit(1)

# Parse the command-line arguments
image_size_str = sys.argv[1]
label_map_str = sys.argv[2]
output_path = sys.argv[3]

# Safely evaluate the string representations of the tuple and dictionary
try:
    image_size = ast.literal_eval(image_size_str)
    if not isinstance(image_size, tuple) or len(image_size) != 2:
        raise ValueError("Image size must be a tuple of two integers.")
except (ValueError, SyntaxError) as e:
    print(f"Error parsing image_size: {e}")
    sys.exit(1)

try:
    label_map = ast.literal_eval(label_map_str)
    if not isinstance(label_map, dict):
        raise ValueError("Label map must be a dictionary.")
except (ValueError, SyntaxError) as e:
    print(f"Error parsing label_map: {e}")
    sys.exit(1)

# Create the dataset_settings dictionary
dataset_settings = {
    "image_size": image_size,
    "label_map": label_map
}

# Save the dictionary to the specified path
torch.save(dataset_settings, output_path)

print(f"Successfully saved dataset settings to {output_path}")
