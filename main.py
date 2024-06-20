import os
import subprocess
import argparse
import warnings

def main():

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    generateDetectionExamples = ['python', ROOT_DIR + '/generateDetectionExamples.py']
    trainDODUO = ['python', ROOT_DIR + '/trainDODUO.py']
    evalDODUO = ['python', ROOT_DIR + '/evalDODUO.py']
    generateCorrectionExamples = ['python', ROOT_DIR + '/generateCorrectionExamples.py']
    trainCorrectionModel = ['python', ROOT_DIR + '/trainCorrectionModel.py']
    evalCorrectionModel = ['python', ROOT_DIR + '/evalCorrectionModel.py']
    
    if args.quick:
        generateDetectionExamples.append('--quick')
        trainDODUO.append('--quick')
        generateCorrectionExamples.append('--quick')
        trainCorrectionModel.append('--quick')
    if args.batchSize:
        generateDetectionExamples.extend(['--batchSize', str(batchSizes[0])])
        trainDODUO.extend(['--batchSize', str(batchSizes[1])])
        evalDODUO.extend(['--batchSize', str(batchSizes[2])])
        trainCorrectionModel.extend(['--batchSize', str(batchSizes[3])])
        evalCorrectionModel.extend(['--batchSize', str(batchSizes[4])])
    if args.data:
        trainDODUO.append('--data')
        trainCorrectionModel.append('--data')
        evalCorrectionModel.append('--data')
    if args.model: 
        evalCorrectionModel.append('--model')

    if not args.model: #only start evalCorrectionModel if args.model
        if not args.data: #skip data generation process when data flag is set
            subprocess.run(generateDetectionExamples)
        subprocess.run(trainDODUO)
        subprocess.run(evalDODUO)
        if not args.data: #skip data generation process when data flag is set
            subprocess.run(generateCorrectionExamples)
        subprocess.run(trainCorrectionModel)
    subprocess.run(evalCorrectionModel)                                     

if __name__ == '__main__':

    def batchSizeType(value):
        try:
            return int(value)
        except ValueError:
            try:
                return [int(v) for v in value.split(',')]
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid batchSize value: {value}")

    parser = argparse.ArgumentParser(description='TAFT Configuration')
    parser.add_argument('--quick', action='store_true', help='Enable quick mode')
    parser.add_argument('--data', action='store_true', help='Use data from the paper instead of creating new data')
    parser.add_argument('--model', action='store_true', help='Use model from the paper')
    parser.add_argument('--batchSize', type=batchSizeType, default=[32, 32, 16, 1, 8], help='Set size of the batches. Provide a single value or a comma-separated list of values (default: [32, 32, 16, 1, 8])')
    args = parser.parse_args()

    if isinstance(args.batchSize, int):
        batchSizes = [args.batchSize, args.batchSize, args.batchSize, args.batchSize, args.batchSize]  # Apply the same batchSize to all stages
    else:
        batchSizes = args.batchSize  # Use the provided list of batch sizes
    
    warnings.filterwarnings('ignore')
    
    main()