# import required packages
import torch
from UtilsFiles.Fight_utils import loadModel, predict_on_video, start_streaming
import argparse


torch.backends.cudnn.benchmark = True

# fetching the arguments from the commandline
parser = argparse.ArgumentParser(description='PyTorch STAM Kinetics Inference')
parser.add_argument('--modelPath')
parser.add_argument('--streaming', action='store_true')
parser.add_argument('--inputPath')
parser.add_argument('--outputPath')
parser.add_argument('--sequenceLength', type=int, default=16)
parser.add_argument('--skip', type=int, default=2)
parser.add_argument('--showInfo', action='store_true')




def main():
    # parsing args
    args = parser.parse_args()

    model = loadModel(args.modelPath)
    # Perform Fight Detection on the Test Video.

    if args.streaming==True:
        start_streaming(model,args.inputPath)
        
    else:
        predict_on_video(args.inputPath, args.outputPath, model, args.sequenceLength, args.skip, args.showInfo)

if __name__ == '__main__':
    main()