"""
Finetune CodeT5+ models on any Seq2Seq LM tasks
You can customize your own training data by following the HF dataset format to cache it to args.cache_data
Author: Yue Wang
Date: June 2023
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5,6,7'
import pprint
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
import evaluate
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu, compute_bleu
import numpy as np

def metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions[0].argmax(-1)
    #labels = labels.reshape([-1])
    #preds = preds.reshape([-1])
    tokenizer = AutoTokenizer.from_pretrained('/DATA_EDS2/EAI/ALOHA/tmp/codet5p-220m')
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    ans = tokenizer.batch_decode(labels, skip_special_tokens=True)
    res = tokenizer.batch_decode(preds, skip_special_tokens=True)
    res = [str.replace('\n', '').replace('\r', '') for str in res]
    #print(ans)
    #print(res)
    with open("ans", "w") as f1, open("res", "w") as f2:
        print("\n".join(ans), file=f1)
        print("\n".join(res), file=f2)
    bleu = round(_bleu("ans", "res"), 2)
    #bleu, _, _, _, _, _ = compute_bleu([[ans]], [res], 4, True)
    #codebleu = calc_code_bleu.get_codebleu("ans", "res", args.lang)
    #return {"bleu": bleu, "codebleu": codebleu}
    return {"bleu": bleu}

def run_training(args, model, data):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        do_eval=True,
        save_strategy='steps',
        eval_strategy="steps",

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        per_device_eval_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,
        eval_accumulation_steps=10,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        eval_steps=args.eval_step,
        save_total_limit=5,
        save_steps=args.save_freq,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        compute_metrics=metrics
    )

    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')


def load_tokenize_data(args):
    # Load and tokenize data
    if os.path.exists(args.cache_data):
        train_data = load_from_disk(args.cache_data)
        print(f'  ==> Loaded {len(train_data)} samples')
        return train_data
    else:
        # Example code to load and process code_x_glue_ct_code_to_text python dataset for code summarization task
        #datasets = load_dataset("code_x_glue_ct_code_to_text", 'python', split="train")
        data_files = {'train':"./CodeT5/CodeT5p/dataset/train.json", 'val':"./CodeT5/CodeT5p/dataset/dev.json"}
        datasets = load_dataset("json", data_files=data_files)
        tokenizer = AutoTokenizer.from_pretrained(args.load)

        def preprocess_function(examples):
            #source = [' '.join(ex) for ex in examples["code_tokens"]]
            #target = [' '.join(ex) for ex in examples["docstring_tokens"]]
            source = examples["nl"]
            target = examples["code"]

            model_inputs = tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
            labels = tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)
            #model_inputs = tokenizer(source, padding="longest", truncation=True)
            #labels = tokenizer(target, padding="longest", truncation=True)
            #print("len", len(model_inputs["input_ids"]), len(labels["input_ids"]))
            #print("max", max(model_inputs["input_ids"][0]), max(labels["input_ids"][0]))

            model_inputs["labels"] = labels["input_ids"].copy()
            model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
            ]
            #print(model_inputs.keys())
            #print(len(model_inputs["input_ids"]), len(model_inputs["input_ids"][0]))
            #print(len(model_inputs["labels"]), len(model_inputs["labels"][0]))
            return model_inputs

        data = datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=datasets.column_names['train'],
            num_proc=64,
            load_from_cache_file=False,
        )
        print(f'  ==> Loaded train:{len(data['train'])} val:{len(data['val'])} samples')
        #print(len(data['train']))
        #print(len(data['val']))
        #data.save_to_disk(args.cache_data)
        return data


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    data = load_tokenize_data(args)

    if args.data_num != -1:
        data = data.select([i for i in range(args.data_num)])

    # Load model from `args.load`
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load, trust_remote_code=True)
    #tokenizer = AutoTokenizer.from_pretrained(args.load)
    #model.resize_token_embeddings(len(tokenizer.vocab))
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-source-len', default=100, type=int)
    parser.add_argument('--max-target-len', default=100, type=int)
    parser.add_argument('--cache-data', default='cache_data/code_generation', type=str)
    parser.add_argument('--lang', default='java', type=str)
    parser.add_argument('--load', default='/DATA_EDS2/EAI/ALOHA/tmp/codet5p-220m', type=str)

    # Training
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=2, type=int)
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/code_generation", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--eval-step', default=50, type=int)
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
