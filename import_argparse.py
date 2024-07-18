import argparse

parser = argparse.ArgumentParser(description='PyTorch STAM Kinetics Inference')

# Define the arguments
parser.add_argument('--modelPath', required=True, help='Path to the model file')
parser.add_argument('--streaming', action='store_true', help='Enable streaming')
# parser.add_argument('--inputPath', required=True, help='Path to the input file')
# parser.add_argument('--outputPath', required=True, help='Path to the output file')
# parser.add_argument('--sequenceLength', type=int, default=16, help='Length of the sequence')
# parser.add_argument('--skip', type=int, default=2, help='Number of frames to skip')
# parser.add_argument('--showInfo', action='store_true', help='Show additional information')

args = parser.parse_args()

print(f"Model Path: {args.modelPath}")
print(f"Streaming: {args.streaming}")
