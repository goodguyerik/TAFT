import os
import sys
import glob
import torch
import random
import warnings
import argparse
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def generateInput(row):
    return f"{row['0']} [ROW] {row['1']} [ROW] {row['2']} [ROW] {row['3']} [ROW] {row['4']}"

#Dataset and DataLoader class to handle and load data
class createDataset(Dataset):
    
    def __init__(self, sources, targets, tokenizer):
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
                 
    def __len__(self):
        return len(self.sources)
   
    def __getitem__(self, item):
        source = self.sources[item]
        target = self.targets[item]
        
        encoding = self.tokenizer.encode_plus(
            source,
            padding = 'max_length',
            max_length = 512,
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            'targets': torch.tensor(target),
            'sources': source,
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
    
    return DataLoader(ds, num_workers = worker, batch_size = batchSize)

def computeMetrics(target, pred):
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='macro')
    acc = accuracy_score(target, pred)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def evalModel(
    model, 
    dataLoader, 
    device
):
    model = model.eval()
    predTensor = torch.empty(0).to(device)
    targetTensor = torch.empty(0).to(device)
    sources = []
    
    with torch.no_grad():
        for d in tqdm(dataLoader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            sources.extend(d['sources'])
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            
            _, preds = torch.max(outputs, dim = 1)

            preds = preds.to(device)

            predTensor = torch.cat((predTensor, preds), 0)
            targetTensor = torch.cat((targetTensor, targets), 0)
            
    targetTensor = targetTensor.cpu()    
    predTensor = predTensor.cpu()
    metrics = computeMetrics(targetTensor, predTensor)
    return metrics, predTensor, targetTensor, sources

def main():
    results = defaultdict(lambda: defaultdict(list))
    for model in models:
        split = 'whitespace'
        tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        test = pd.read_parquet(f'{dataPath}data/Detection/realTest.parquet')
        test['output'] = labelEncoder.fit_transform(test['label']) 
        dataLoader = createDataLoader(test, 'input', 'output', tokenizer, BATCH_SIZE)
        if model.startswith('ROW'):
            split = 'ROW'
            tokenizer.add_tokens(list(['[ROW]']))
            test['inputROW'] = test.apply(generateInput, axis=1)
            dataLoader = createDataLoader(test, 'inputROW', 'output', tokenizer, BATCH_SIZE)
    
        class DetectionModel(torch.nn.Module):
            def __init__(self, config):
                super(DetectionModel, self).__init__()
                
                self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
                self.bert.resize_token_embeddings(len(tokenizer))
                self.l = torch.nn.Linear(config['hiddenSize'], config['numLabels'])
        
            def forward(self, input_ids, attention_mask):
                
                out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
                hidden_state = out[0]                    
                x = hidden_state[:, 0]
                x = self.l(x)
                
                return x
    
        test = pd.read_parquet(f'{dataPath}data/Detection/realTest.parquet')
        test['output'] = labelEncoder.fit_transform(test['label']) 
    
        test['inputROW'] = test.apply(generateInput, axis=1)
        testDataLoader = createDataLoader(test, 'input', 'output', tokenizer, BATCH_SIZE)
        testRowDataLoader = createDataLoader(test, 'inputROW', 'output', tokenizer, BATCH_SIZE)



        
        strategy = 'mutation'
        if 'random' in model:
            strategy = 'random'
        elif 'occ' in model:
            strategy = 'OCCSVM'
        checkpoint = torch.load(f'{dataPath}models/Detection/{model}.bin')
        model_ = DetectionModel(config)
                
        if next(iter(checkpoint.keys())).startswith('module'):
            new_state_dict = {k[7:]: v for k, v in checkpoint.items()}
            model_.load_state_dict(new_state_dict)
        else:
            # If not, simply load the state dictionary
            model_.load_state_dict(checkpoint)
                
        model_.to(device)

        met, pred, target, source = evalModel(model_, dataLoader, device)
        results[split][strategy].append(met["f1"])
    
    def computeAverage(lst):
        return sum(lst) / len(lst) if lst else 0

    for outerKey, innerDict in results.items():
        print(f"Averages for {outerKey}:")
        for innerKey, lst in innerDict.items():
            average = computeAverage(lst)
            print(f"  {innerKey}: {average:.3f}")

if __name__ == "__main__":
    dataPath = os.path.dirname(os.path.abspath(__file__)) + '/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    devCount = torch.cuda.device_count()

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description="TAFT DODUO Eval")
    parser.add_argument('--batchSize', type=int, default=1, help='Set size of the batches')
    args = parser.parse_args()
    
    strategies = ['randomStrategy','mutationStrategy','occSvmStrategy']
    split = ['whitespace', 'ROW']

    models = glob.glob(f'{dataPath}models/Detection/*')
    temp = []
    for model in models:
        temp.append(model.split('/')[-1].replace('.bin', ''))
    models = temp
    
    BATCH_SIZE = args.batchSize #32
    HIDDEN_SIZE = 768
    NUM_LABELS = 9
    
    PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
    
    config = {
        'hiddenSize': HIDDEN_SIZE,
        'numLabels': NUM_LABELS
    }
    labelEncoder = LabelEncoder()  # Initialize label encoder

    main()