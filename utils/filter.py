import sys

def filter_file(input_file):
    try:
        with open(input_file, 'r') as f:
            for line in f:
                if "ERROR" not in line and "DEPOLARIZE" not in line:
                    print(line, end='')
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <input_file>")
    else:
        input_file = sys.argv[1]
        filter_file(input_file)
