import os
import re
import sys
import glob
import random
import argparse
import warnings
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

def validateConversion(sourceFormat, targetFormat, requirements):
    sourceComponents = re.findall(r'\b\w+\b', sourceFormat)
    targetComponents = re.findall(r'\b\w+\b', targetFormat)

    if requirements == {} and len(sourceComponents) == 1 and len(targetComponents) == 1: # like Country, Continent, Sex ...
        return True
    
    targetComponents = list(set(targetComponents) - set(sourceComponents)) #components that are given

    for key, value in requirements.items(): #components that are indirectly given (e.g. forename in sourceFormat and initial in targetFormat)
        if key in targetComponents and value in sourceComponents:
            targetComponents.remove(key)

    if len(targetComponents) > 0:
        return False
    else:
        return True

def adjustExamples(module, sourceFormat, targetFormat):
    elements = [module.getElement() for _ in range(50)]
    inputs = []
    outputs = []
    for element in elements:
        inputs.append(module.adjustElement(element, sourceFormat))
        outputs.append(module.adjustElement(element, targetFormat))
    return inputs, outputs

#only add as many elements as the input size of the transformer model allows (max. 512 tokens)
def truncateExampleSlow(inputs, outputFormat, outputs):
    inputString = f'{outputFormat} reshape: {inputs[0]}'
    outputString = outputs[0]
    for i in range(1, len(inputs)): 
        tempInput = f'{inputString} [ROW] {inputs[i]}'
        tempOutput = f'{outputString} [ROW] {outputs[i]}'
        inputLength = len(tokenizer(tempInput).input_ids)
        outputLength = len(tokenizer(tempOutput).input_ids)
        if inputLength < MAX_LEN and outputLength < MAX_LEN:
            inputString = tempInput
            outputString = tempOutput
        else: 
            break
    return inputString, outputString

def truncateExampleFast(inputs, outputFormat, outputs):
    inputString = f'{outputFormat} reshape: {" [ROW] ".join(inputs)}'
    temp = tokenizer(inputString)['input_ids'][:MAX_LEN]
    lastIdx = len(temp) - 1 - temp[::-1].index(32100) #[ROW]
    truncatedList = temp[:lastIdx]
    rowCount = truncatedList.count(32100)
    
    #if output still larger than input
    outputString = f'{outputs[0]} {" [ROW] ".join(outputs[:rowCount])}'
    temp = tokenizer(outputString)['input_ids'][:MAX_LEN]
    lastIdx = len(temp) - 1 - temp[::-1].index(32100) #[ROW]
    truncatedList = temp[:lastIdx]
    rowCount = truncatedList.count(32100)
    
    inputString = f'{outputFormat} reshape: {" [ROW] ".join(inputs[:rowCount])}'
    outputString = f'{" [ROW] ".join(outputs[:rowCount])}'
    return inputString, outputString

def generateExample(numExamples, type, domain):
    examples = []
    module = importlib.import_module(type)
    requirements = {}
    try: 
        requirements = module.requirements #load requirements
    except: 
        pass
        
    for i in tqdm(range(numExamples)):
        while(True):
            sourceFormat = random.choice(module.formats)
            targetFormat = random.choice(module.formats)
            if validateConversion(sourceFormat, targetFormat, requirements):
                break  
        inputs, outputs = adjustExamples(module, sourceFormat, targetFormat)
        inputString, outputString = truncateExampleFast(inputs, targetFormat, outputs)
        examples.append((inputString, outputString))
    df = pd.DataFrame(examples, columns=['input', 'output'])
    df.to_parquet(f'{dataPath}data/Correction/{domain}/{type}.parquet', engine='pyarrow')

def main():
    print('start the generation process for correction data')
    for type in types:
        print(f'generate Train data for {type} type')
        generateExample(trainNum, type, 'train')
        print(f'generate Val data for {type} type')
        generateExample(valNum, type, 'val')
        print(f'generate Test data for {type} type')
        generateExample(testNum, type, 'test')
    
if __name__ == "__main__":
    trainNum = 200000
    valNum = 10000
    testNum = 1000

    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser(description="TAFT Data Generation Correction Stage")
    parser.add_argument('--quick', action='store_true', help='Enable quick mode')
    args = parser.parse_args()
    if args.quick:
        trainNum = 100
        valNum = 10
        testNum = 10   
    
    dataPath = os.path.dirname(os.path.abspath(__file__)) + '/'
    fuzzyGenerators = glob.glob(f'{dataPath}fuzzyGenerators/*.py')
    types = [os.path.splitext(os.path.basename(fuzzyGenerator))[0] for fuzzyGenerator in fuzzyGenerators]
    sys.path.append(f'{dataPath}fuzzyGenerators')

    PRE_TRAINED_MODEL_NAME = "google/flan-t5-base"
    MAX_LEN = 200

    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, use_fast=False)
    tokenizer.add_tokens(list(['[ROW]']))

    main()