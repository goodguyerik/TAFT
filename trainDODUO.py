import os
import sys
import torch
import random
import argparse
import warnings
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def generateInput(row, split, full=True):
    row = [str(element) for element in list(row)]
    if not full:
        count = random.randint(1,5)
        row = row[:count]
    row = [x for x in row if x != 'nan']
    return f'{split}'.join(row)

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

def trainEpoch(
    model,
    dataLoader,
    lossFn,
    optimizer,
    device,
    scheduler
):
    
    model = model.train()
    losses = []
    predTensor = torch.empty(0).to(device)
    targetTensor = torch.empty(0).to(device)
    
    for d in tqdm(dataLoader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        
        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
        
        _, preds = torch.max(outputs, dim = 1)
        loss = lossFn(outputs, targets)
        
        losses.append(loss.item())
        
        preds = preds.to(device)
        
        predTensor = torch.cat((predTensor, preds), 0)
        targetTensor = torch.cat((targetTensor, targets), 0)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    scheduler.step()
    targetTensor = targetTensor.cpu()    
    predTensor = predTensor.cpu()
    metrics = computeMetrics(targetTensor, predTensor)
    return np.mean(losses), metrics

def evalModel(
    model, 
    dataLoader, 
    lossFn, 
    device
):
    model = model.eval()
    losses = []
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
            loss = lossFn(outputs, targets)

            losses.append(loss.item())

            preds = preds.to(device)

            predTensor = torch.cat((predTensor, preds), 0)
            targetTensor = torch.cat((targetTensor, targets), 0)
            
    targetTensor = targetTensor.cpu()    
    predTensor = predTensor.cpu()
    metrics = computeMetrics(targetTensor, predTensor)
    return np.mean(losses), metrics, predTensor, targetTensor, sources

def main():
    for split_ in split:
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
        if split_ == 'ROW':
            tokenizer.add_tokens(list(['[ROW]']))

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

        if args.data:
            train = pd.read_parquet(f'{dataPath}data/Detection/paper_train.parquet')
            val = pd.read_parquet(f'{dataPath}data/Detection/paper_val.parquet')
        else:
            train = pd.read_parquet(f'{dataPath}data/Detection/train.parquet')
            val = pd.read_parquet(f'{dataPath}data/Detection/val.parquet')
        
        for strategy in strategies:
            print(f'train DODUO with {split_} delimiter and {strategy}')
            strategiesToRemove = [strategy_ for strategy_ in strategies if strategy_ != strategy]
            train_ = train[~train['label'].isin(strategiesToRemove)]
            val_ = val[~val['label'].isin(strategiesToRemove)]

            train_['label'].replace(strategies, 'Negative', inplace=True)
            val_['label'].replace(strategies, 'Negative', inplace=True)
    
            train_ = train_.sample(trainNum)
            val_ = val_.sample(valNum)
    
            columns = [f'row{i}' for i in range(5, sampleNum)]
            columns.extend(['label'])

            delimiter = ' '
            if split_ == "ROW":
                delimiter = ' [ROW] '                
            train_['input'] = train_.drop(columns=columns).apply(generateInput, args=(delimiter, True), axis=1)
            val_['input'] = val_.drop(columns=columns).apply(generateInput, args=(delimiter, True), axis=1)
    
            train_['output'] = labelEncoder.fit_transform(train_['label'])
            val_['output'] = labelEncoder.fit_transform(val_['label'])
    
            trainDataLoader = createDataLoader(train_, 'input', 'output', tokenizer, BATCH_SIZE)
            valDataLoader = createDataLoader(val_, 'input', 'output', tokenizer, BATCH_SIZE)

            for i in range(RUNS):
                print(f'RUN: {i}')

                model = DetectionModel(config)
                if devCount > 1:
                    model = torch.nn.DataParallel(model, device_ids=list(range(devCount)))
                model.to(device)

                optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
                totalSteps = len(trainDataLoader) * EPOCHS
                
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps = 0,
                    num_training_steps = totalSteps
                )
                
                lossFn = torch.nn.CrossEntropyLoss().to(device)

                bestF1 = 0
                for epoch in range(EPOCHS):
                    print(f'EPOCH {epoch + 1}/{EPOCHS}')
                    print('-' * 10)
                
                    trainLoss, trainMet = trainEpoch(
                        model,
                        trainDataLoader,
                        lossFn,
                        optimizer,
                        device,
                        scheduler
                    )
                
                    print(f'Train loss {trainLoss} Train metrics {trainMet}')
                            
                    valLoss, valMet, _, _, _ = evalModel(
                        model,
                        valDataLoader,
                        lossFn,
                        device
                    )
                
                    print(f'Val loss {valLoss} Val metrics {valMet}')
                    print()
                
                    if valMet['f1'] > bestF1:
                        torch.save(model.state_dict(), f'{dataPath}models/Detection/{split_}{strategy}{i}.bin')
                        bestF1 = valMet['f1']

if __name__ == "__main__":
    RUNS = 10
    EPOCHS = 10
    sampleNum = 100
    trainNum = 50000
    valNum = 5000

    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser(description="TAFT DODUO Training")
    parser.add_argument('--quick', action='store_true', help='Enable quick mode')
    parser.add_argument('--data', action='store_true', help='Use data from the paper instead of creating new data')
    parser.add_argument('--batchSize', type=int, default=1, help='Set size of the batches')

    args = parser.parse_args()
    if args.quick:
        RUNS = 1
        EPOCHS = 1
        sampleNum = 5
        trainNum = 500
        valNum = 50
    
    dataPath = os.path.dirname(os.path.abspath(__file__)) + '/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    devCount = torch.cuda.device_count()

    BATCH_SIZE = args.batchSize #32
    HIDDEN_SIZE = 768
    LEARNING_RATE = 7e-6
    NUM_LABELS = 9
    PRE_TRAINED_MODEL_NAME = "bert-base-uncased"

    config = {
        'hiddenSize': HIDDEN_SIZE,
        'numLabels': NUM_LABELS
    }

    strategies = ['randomStrategy','mutationStrategy','occSvmStrategy']
    split = ['whitespace', 'ROW']

    labelEncoder = LabelEncoder()  # Initialize label encoder

    main()