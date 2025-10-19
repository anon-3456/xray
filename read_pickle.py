import torch
import sys

# Check if the correct number of arguments are provided
if len(sys.argv) != 2:
    print("Usage: python read_pickle.py <path_to_pickle_file>")
    sys.exit(1)

# Parse the command-line arguments
pickle_path = sys.argv[1]

# Load the pickle file
data = torch.load(pickle_path, weights_only=False)

# Print the data
print(data)
