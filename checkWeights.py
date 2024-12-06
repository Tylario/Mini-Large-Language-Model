import json

def check_weights_architecture(filename='neuralNetworkWeights.json'):
    try:
        with open(filename, 'r') as f:
            weights_data = json.load(f)
        
        architecture = weights_data['architecture']
        print("\nSaved model architecture:")
        print(f"Input size: {architecture['input_size']}")
        print(f"Hidden layers: {architecture['hidden_layers']}")
        print(f"Output size: {architecture['output_size']}")
        
    except FileNotFoundError:
        print(f"\nNo weights file found at: {filename}")
    except KeyError:
        print("\nInvalid weights file format - missing architecture information")
    except json.JSONDecodeError:
        print("\nInvalid JSON file")

if __name__ == "__main__":
    check_weights_architecture()
