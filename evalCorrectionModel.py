import os
import re
import sys
import glob
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

class createDataset(torch.utils.data.Dataset):
    
    def __init__(self, sources, targets, tokenizer):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
                 
    def __len__(self):
        return len(self.sources)
   
    def __getitem__(self, item):
        source = self.sources[item]
        target = self.targets[item]
        if(target != target):
            target = ''
        
        encoding_source = self.tokenizer.encode_plus(
            source,
            padding = 'max_length',
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        
        encoding_target = self.tokenizer.encode_plus(
            target,
            padding = 'max_length',
            max_length = MAX_LEN,
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        
        labels = encoding_target['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding_source["input_ids"].flatten(),
            "attention_mask": encoding_source["attention_mask"].flatten(),
            "labels": labels.flatten()
        }
    
def createDataLoader(df, source, target, tokenizer, batchSize):
    ds = createDataset(
        sources = df[source].to_numpy(),
        targets = df[target].to_numpy(),
        tokenizer = tokenizer
    )
    
    worker = devCount
    if devCount > 4:
        worker = 4
    
    return torch.utils.data.DataLoader(
        ds,
        num_workers = worker,
        batch_size = batchSize
    )

def postprocessText(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]
    return preds, labels

def computeMetrics(preds, labels, decodedData):
    if isinstance(preds, tuple):
        preds = preds[0]
    decodedPreds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decodedLabels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decodedPreds, decodedLabels = postprocessText(decodedPreds, decodedLabels)
    decodedData.append([decodedPreds, decodedLabels])
    predictionLens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    return  predictionLens, decodedData

def evalModel(
    model,
    dataLoader,
    device,
    decodedData
):
    accelerator = Accelerator()
    
    model, dataLoader = accelerator.prepare(model, dataLoader)
    
    model.eval()
    for d in tqdm(dataLoader):
        input_ids = d["input_ids"].to(device)
        labels = d["labels"].to(device)
                
        predictions = accelerator.unwrap_model(model).generate(input_ids=input_ids, max_length=MAX_LEN)
        outputs, labels = accelerator.gather_for_metrics((predictions, labels))
        labels = labels.cpu()
        outputs = outputs.cpu()
        len, decodedData = computeMetrics(outputs, labels, decodedData)
    return decodedData

def assessColumns(decodedData, length):
    correctColumns = []
    incorrectColumns = []
    for k in range(length):
        for i in range(BATCH_SIZE):
            try:
                if decodedData[k][0][i] == decodedData[k][1][i]:
                    correctColumns.append([decodedData[k][0][i], decodedData[k][1][i]])
                else:
                    incorrectColumns.append([decodedData[k][0][i], decodedData[k][1][i], k+i])
            except:
                pass
    return correctColumns, incorrectColumns

def countFailures(incorrectColumns):
    length = len(incorrectColumns)
    failures = 0
    values = 0
    for k in range(length):
        preds = incorrectColumns[k][0].split(' [ROW] ')
        labels = incorrectColumns[k][1].split(' [ROW] ')
        count = 0
        temp = []
        for i in range(min(len(preds), len(labels))):
            if preds[i] != labels[i]:
                count += 1
                temp.append([preds[i], labels[i]])
        count += abs(len(labels) - len(preds))
        failures += count
        values += len(labels)
    return failures, values

def main():
    allFailures = 0
    allValues = 0
    for type in types: 
        decodedData = []
        decodedData = evalModel(model, dataLoaders[type], device, decodedData)
        correctColumns, incorrectColumns = assessColumns(decodedData, LEN_DATALOADERS[type])
        failures, total = countFailures(incorrectColumns)
        allFailures += failures
        allValues += total

        print(f'{type} type was transformed {len(correctColumns)} times correctly and {len(incorrectColumns)} incorrectly')
    p = allFailures * 100 / allValues
    print(f'{allFailures} out of {allValues} column values were transformed incorrectly, i.e. only {round(p, 2)} percent of all column values were transformed incorrectly')

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="TAFT Correction Eval")
    parser.add_argument('--batchSize', type=int, default=1, help='Set size of the batches')
    parser.add_argument('--data', action='store_true', help='Use data from the paper instead of creating new data')
    parser.add_argument('--model', action='store_true', help='Use model from the paper')
    args = parser.parse_args()
    
    dataPath = os.path.dirname(os.path.abspath(__file__)) + '/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    devCount = torch.cuda.device_count()
    
    fuzzyGenerators = glob.glob(f'{dataPath}fuzzyGenerators/*.py')
    types = [os.path.splitext(os.path.basename(fuzzyGenerator))[0] for fuzzyGenerator in fuzzyGenerators]
    sys.path.append(f'{dataPath}fuzzyGenerators')
    
    PRE_TRAINED_MODEL_NAME = "google/flan-t5-large"
    BATCH_SIZE = args.batchSize #8
    MAX_LEN = 200
    
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    tokenizer.add_tokens(list(['[ROW]']))

    dfs = {}
    for type in types:

        if args.data:
            df = pd.read_parquet(f'{dataPath}data/Correction/test/paper_{type}.parquet')
        else:
            df = pd.read_parquet(f'{dataPath}data/Correction/test/{type}.parquet')
        dfs[type] = df

    LEN_DATALOADERS = {}

    dataLoaders = {}
    for type in types:
        dataLoader = createDataLoader(dfs[type], 'input', 'output', tokenizer, BATCH_SIZE)
        dataLoaders[type] = dataLoader
        LEN_DATALOADERS[type] = len(dataLoader)

    models = glob.glob(f'{dataPath}models/Correction/checkpoint-*/model.safetensors')
    highestCheckpoint = max(models, key=lambda p: int(re.search(r'checkpoint-(\d+)', p).group(1)))
    
    model = AutoModelForSeq2SeqLM.from_pretrained(PRE_TRAINED_MODEL_NAME)
    if args.model:
        dict = load_file(f'{dataPath}models/Correction/paperModel/model.safetensors')
    else:
        dict = load_file(highestCheckpoint)
    model.load_state_dict(dict, strict=False)
    model.to(device)
    
    main()