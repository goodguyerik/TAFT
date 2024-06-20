#code used and adapted from [1]
#[1] - https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/flan-t5-samsum-summarization.ipynb

#setup comet to monitor training
"""import comet_ml
import os
os.COMET_LOG_ASSET = True
os.environ["COMET_API_KEY"] = "XXXXXXX"
projectName = 'Corrector'
comet_ml.init(projectName)"""

import os
import glob
import torch
import evaluate
import argparse
import warnings
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

def dataPreprocessing(sample): 
    inputTokens = tokenizer(sample['input'], max_length = MAX_LEN, padding = 'max_length', truncation=True)
    outputTokens = tokenizer(sample['output'], max_length = MAX_LEN, padding = 'max_length', truncation=True)

    #replace padToken with -100 to ignore padding in the loss
    inputTokens['labels'] = [[(token if token != tokenizer.pad_token_id else -100) for token in tokens] for tokens in outputTokens['input_ids']]
    
    return inputTokens

def dataPostprocessing(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def computeMetrics(evalPreds):
    preds, labels = evalPreds
    if isinstance(preds, tuple):
        preds = preds[0]
    predsDecoded = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labelsDecoded = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    predsDecoded, labelsDecoded = dataPostprocessing(predsDecoded, labelsDecoded)
    
    result = metric.compute(predictions=predsDecoded, references=labelsDecoded, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    predictionLens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(predictionLens)
    return result

def main():
    print('start training the correction model')
    trainer.train()
    model.save_pretrained(outDir)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    parser = argparse.ArgumentParser(description="TAFT Correction Model Training")
    parser.add_argument('--quick', action='store_true', help='Enable quick mode')
    parser.add_argument('--data', action='store_true', help='Use data from the paper instead of creating new data')
    parser.add_argument('--batchSize', type=int, default=1, help='The size of the batches')
    args = parser.parse_args()
    
    dataPath = os.path.dirname(os.path.abspath(__file__)) + '/'
    fuzzyGenerators = glob.glob(f'{dataPath}fuzzyGenerators/*.py')
    types = [os.path.splitext(os.path.basename(fuzzyGenerator))[0] for fuzzyGenerator in fuzzyGenerators]
    outDir = f'{dataPath}models/Correction/'
    
    PRE_TRAINED_MODEL_NAME = "google/flan-t5-large"
    MAX_LEN = 200
    BATCH_SIZE = args.batchSize
    EPOCHS = 5
    UPDATE_STEPS = 10000
    if args.quick:
        EPOCHS = 1
        UPDATE_STEPS = 200
    
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    tokenizer.add_tokens(list(['[ROW]']))
    
    #load datasets
    trainList = []
    valList = []
    
    for type in types:

        if args.data: #use data from the paper
            type = 'paper_' + type
        
        train = pd.read_parquet(f'{dataPath}data/Correction/train/{type}.parquet')
        print(len(train))
        print(type)
        if args.quick:
            train = train.sample(int(len(train) / 2))
        val = pd.read_parquet(f'{dataPath}data/Correction/val/{type}.parquet')
        if args.quick:
            val = val.sample(int(len(val) / 2))
        trainList.append(train)
        valList.append(val)
    
    train = pd.concat(trainList)
    val = pd.concat(valList)
    
    trainDataset = Dataset.from_pandas(train)
    valDataset = Dataset.from_pandas(val)
    
    dataset = DatasetDict()
    dataset['train'] = trainDataset
    dataset['val'] = valDataset

    dataset = dataset.map(dataPreprocessing, batched = True, remove_columns = ['input', 'output'])

    model = AutoModelForSeq2SeqLM.from_pretrained(PRE_TRAINED_MODEL_NAME)
    model.config.max_length=200
    metric = evaluate.load('rouge')

    label_pad_token_id = -100

    dataCollator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )
    
    trainingArgs = Seq2SeqTrainingArguments(
        output_dir = outDir,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE,
        predict_with_generate = True,
        fp16 = False,
        learning_rate = 7e-6,
        num_train_epochs = EPOCHS,
        optim = "adamw_torch",
        logging_dir = f"{outDir}/logs",
        logging_strategy = "steps",
        logging_steps = UPDATE_STEPS,
        evaluation_strategy = "steps",
        eval_steps = UPDATE_STEPS,
        save_strategy = "steps",
        save_steps = UPDATE_STEPS,
        save_total_limit = 3,
        load_best_model_at_end = True,
        #report_to=["comet_ml"]
    )
    
    trainer = Seq2SeqTrainer(
        model = model,
        args = trainingArgs,
        data_collator = dataCollator,
        train_dataset = dataset["train"],
        eval_dataset = dataset["val"],
        compute_metrics = computeMetrics,
    )
    
    main()