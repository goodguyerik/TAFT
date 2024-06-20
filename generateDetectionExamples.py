import os
import sys
import glob
import torch
import string
import random
import warnings
import argparse
import importlib
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from functools import reduce
from sklearn.svm import OneClassSVM
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

def generatePositiveExamples(sampleNum, type, numExamples):
    module = importlib.import_module(type)
    columns = []
    for i in tqdm(range(numExamples)):
        count = random.randint(1, sampleNum)
        columnFormat = random.choice(module.formats)
        elements = []
        for _ in range(count):
            elements.append(module.adjustElement(module.getElement(), columnFormat))
        elements.extend(['nan'] * (sampleNum - len(elements)))
        columns.append(elements)
    df = pd.DataFrame(columns)
    df.columns = df.columns.astype(str)
    df.to_parquet(f'{dataPath}data/Detection/{type}.parquet', engine='pyarrow')

class createDataset(Dataset):
    
    def __init__(self, sources, tokenizer):
        self.sources = sources
        self.tokenizer = tokenizer
                 
    def __len__(self):
        return len(self.sources)
   
    def __getitem__(self, item):
        source = self.sources[item]
        
        encoding = self.tokenizer.encode_plus(
            source,
            padding = 'max_length',
            max_length = 512,
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        
        return {
            "text": source,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }

def createDataLoader(df, input_, tokenizer, batchSize):
    ds = createDataset(
        sources = df[input_].to_numpy(),
        tokenizer = tokenizer
    )
    worker = devCount
    if devCount > 4:
        worker = 4
    
    return DataLoader(
        ds,
        num_workers = devCount,
        batch_size = batchSize
    )

class DebertaClassifier(nn.Module):
    def __init__(self, config):
        super(DebertaClassifier, self).__init__()
        
        self.deberta = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.deberta.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, attention_mask):
        
        out = self.deberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        hidden_state = out[0]                    
        x = hidden_state[:, 0]
        
        return x

def embedData(df, column, model):
    dataLoader = createDataLoader(df, column, tokenizer, BATCH_SIZE)

    model = model.eval()
    predTensor = torch.empty(0).to(device)
    
    with torch.no_grad():
        for d in tqdm(dataLoader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            outputs = model(input_ids = input_ids,attention_mask = attention_mask)
    
            predTensor = torch.cat((predTensor, outputs), 0)
               
    predTensor = predTensor.cpu()
    predTensor = predTensor.numpy()
    return predTensor

def generateSequence(df):
    df['sequence'] = df.apply(lambda row: ' [ROW] '.join(map(str, row)), axis=1)
    return df

chars = string.ascii_letters + string.digits + string.punctuation

def addRandomChar(element):
    randomChar = random.choice(chars)
    randomPosition = random.randint(0, len(element))
    return element[:randomPosition] + randomChar + element[randomPosition:]

def deleteRandomChar(element):
    try:
        randomPosition = random.randint(0, len(element) - 1)
    except: 
        return element
    return element[:randomPosition] + element[randomPosition + 1:]

def replaceRandomChar(element, targetChar):
    indices = [i for i, char in enumerate(element) if char in targetChar]
    randomChar = random.choice(chars)
    try:
        randomPosition = random.choice(indices)
    except:
        return element
    return element[:randomPosition] + randomChar + element[randomPosition + 1:]

def mutate(module, sampleNum):
    sampleNum_ = random.randint(1, 100)
    columnFormat = random.choice(module.formats)
    elements = []
    if sampleNum_ > sampleNum:
        sampleNum_ = sampleNum
    for i in range(sampleNum_):
        elements.append(module.adjustElement(module.getElement(), columnFormat))
    
    avgSpecialChars = sum(1 for element in elements for c in element if c in string.punctuation) / len(elements)
    avgLength = sum(len(element) for element in elements) / len(elements)
    avgDigits = sum(1 for element in elements for c in element if c.isdigit()) / len(elements)
        
    for i in range(sampleNum_):
        element = elements[i]

        for _ in range(random.randint(1, 5)):
            prob1 = random.random()
            prob2 = random.random()
            prob3 = random.random()
            prob4 = random.random()
            prob5 = random.random()
            if avgLength < 4:
                if prob1 < 0.5:
                    element = addRandomChar(element)
                if prob2 < 0.5:
                    element = replaceRandomChar(element, string.ascii_letters)
            if avgLength > 10:
                if prob1 < 0.5:
                    element = deleteRandomChar(element)
            if avgSpecialChars >= 1:
                    if prob3 < 0.5:
                        element = replaceRandomChar(element, string.punctuation)
            if avgDigits >= 1:
                    if prob4 < 0.5:
                        element = replaceRandomChar(element, string.digits)
            if prob5 < 0.5:
                element = replaceRandomChar(element, ' ') #for whitespaces
            element = replaceRandomChar(element, chars)
        
        elements[i] = element
    elements += ['nan'] * (sampleNum - sampleNum_)
    return elements

def randomStrategy(numExamples, sampleNum):
    print('use Random strategy')
    temp = gitTables.sample(numExamples)
    columns = ['fileName', 'index', 'columnName']
    columns.extend([f'row{i}' for i in range(sampleNum, 100)])
    return temp.drop(columns, axis=1)

def mutationStrategy(numExamples, sampleNum):
    print('use Mutation strategy')
    typeCount = len(types)
    split = int(numExamples / typeCount)
    remainder = numExamples - split * typeCount
    
    examples = []

    for type in types: #produce examples using each type fuzzy generator
        module = importlib.import_module(type)
        for _ in tqdm(range(split)):  
            examples.append(mutate(module, sampleNum))
    for _ in range(remainder):
        examples.append(mutate(module, sampleNum))
    return examples

def occSvmStrategy(numExamples, sampleNum):
    print('use OCC-SVM strategy')
    columns = ['fileName', 'index', 'columnName']
    columns.extend([f'row{i}' for i in range(5, sampleNum)])
    gitTablesForOccSvm = gitTables.drop(columns, axis=1) #only include the first SAMPLENUM rows of each column
    #gitTablesForOccSvm = gitTablesForOccSvm.sample(100)
    gitTablesForOccSvm = generateSequence(gitTablesForOccSvm)
    y = embedData(gitTablesForOccSvm, 'sequence', model) #embeds gitTables for inference

    dfs = []
    for type in types:  
        positiveExamples = pd.read_parquet(f'{dataPath}data/Detection/{type}.parquet')
        columns =  [f'{i}' for i in range(5, sampleNum)]
        positiveExamples = positiveExamples.drop(columns, axis=1) #only include the first SAMPLENUM rows of each column
        positiveExamples = generateSequence(positiveExamples)
        if len(positiveExamples) >= 20000:
            positiveExamples = positiveExamples.sample(20000)
        X = embedData(positiveExamples, 'sequence', model)

        clf = OneClassSVM(gamma='auto', nu=0.1)
        clf.fit(X)

        preds = clf.predict(y)
        temp = gitTables.iloc[preds == -1]
        temp = temp.reset_index()
        dfs.append(temp)

    negs = reduce(lambda left, right: pd.merge(left, right, on=list(left.columns)), dfs)
    if len(negs) > numExamples:
        negs = negs.sample(numExamples)
    return negs

def generateNegativeExamples(strategyName, numExamples, sampleNum):
    examples = strategies[strategyName](numExamples, sampleNum)
    columns =  [f'row{i}' for i in range(sampleNum)]
    df = pd.DataFrame (examples, columns = columns)
    df.to_parquet(f'{dataPath}data/Detection//{strategyName}.parquet', engine='pyarrow')

def combineExamples():
    train = []
    val = []
    test = []
    for type in types:
        df = pd.read_parquet(f'{dataPath}data/Detection/{type}.parquet')
        newColumns = {str(i): f'row{i}' for i in range(len(df.columns))} #align column names
        df = df.rename(columns=newColumns)
        df['label'] = type
        train.append(df[:int(((numPosExamples/12)*10))]) #:50000
        val.append(df[int(((numPosExamples/12)*10)):int(((numPosExamples/12)*11))]) #50000:55000
        test.append(df[int(((numPosExamples/12)*11)):]) #55000:
    
    for strategy in strategies:
        df = pd.read_parquet(f'{dataPath}data/Detection/{strategy}.parquet')
        df['label'] = strategy
        train.append(df[:int(((numNegExamples/12)*10))]) #200000
        val.append(df[int(((numNegExamples/12)*10)):int(((numNegExamples/12)*11))]) #200000:220000
        test.append(df[int(((numNegExamples/12)*11)):]) #220000:

    train  = pd.concat(train)
    val = pd.concat(val)
    test = pd.concat(test)
    train.to_parquet(f'{dataPath}data/Detection/train.parquet')
    val.to_parquet(f'{dataPath}data/Detection/val.parquet')
    test.to_parquet(f'{dataPath}data/Detection/test.parquet')

def main():
    for type in types:
        print(f'generate Examples for the {type} type')
        generatePositiveExamples(sampleNum, type, numPosExamples)
    
    for strategyName, _ in strategies.items():
        print('generate Examples for the abstain type')
        generateNegativeExamples(strategyName, numNegExamples, sampleNum)

    print('generate train, validation and test dataset')
    combineExamples()

if __name__ == "__main__":

    sampleNum = 100
    numNegExamples = 240000
    numPosExamples = 60000
    gitTableSize = 500000

    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser(description="TAFT Data Generation")
    parser.add_argument('--quick', action='store_true', help='Enable quick mode')
    parser.add_argument('--batchSize', type=int, default=1, help='Set size of the batches')

    args = parser.parse_args()
    if args.quick:
        sampleNum = 5
        numNegExamples = 2400
        numPosExamples = 600
        gitTableSize = 5000

    
    dataPath = os.path.dirname(os.path.abspath(__file__)) + '/'
    fuzzyGenerators = glob.glob(f'{dataPath}fuzzyGenerators/*.py')
    types = [os.path.splitext(os.path.basename(fuzzyGenerator))[0] for fuzzyGenerator in fuzzyGenerators]
    sys.path.append(f'{dataPath}fuzzyGenerators')
    
    gitTables = pd.read_parquet(f'{dataPath}data/gitTables.parquet')
    gitTables = gitTables.sample(gitTableSize)
    gitTables = gitTables.reset_index(drop = True)
    sys.path.append(f'{dataPath}fuzzyGenerators')

    #setup for OCC-SVM
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    devCount = torch.cuda.device_count()
    
    BATCH_SIZE = args.batchSize
    HIDDEN_SIZE = 1024
    PRE_TRAINED_MODEL_NAME = "microsoft/deberta-v3-base"
    
    config = {
        'hidden_size': HIDDEN_SIZE
    }
    
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    tokenizer.add_tokens(list(['[ROW]']))

    model = DebertaClassifier(config)
    if  devCount > 1:
        model = nn.DataParallel(model, device_ids=list(range(devCount)))
    model.to(device) 

    strategies = {
        'randomStrategy': randomStrategy,
        'mutationStrategy': mutationStrategy,
        'occSvmStrategy': occSvmStrategy
    }

    main()
    