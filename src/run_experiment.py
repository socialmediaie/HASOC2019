import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from pytorch_transformers import *
from scipy.special import softmax

import random
import os
import logging
import json
import glob

logger = logging.getLogger(__name__)


BASE_PATH = os.environ.get("HASOC_PATH")
DATA_PATHS_TRAIN = {
    "EN": f"{BASE_PATH}/data/raw/training_data/english_dataset.tsv",
    "DE": f"{BASE_PATH}/data/raw/training_data/german_dataset.tsv",
    "HI": f"{BASE_PATH}/data/raw/training_data/hindi_dataset.tsv"
}
DATA_PATHS_DEV = {
    "EN": f"{BASE_PATH}/data/raw/teaser_data/english_dataset.tsv",
    "DE": f"{BASE_PATH}/data/raw/teaser_data/german_dataset.tsv",
    "HI": f"{BASE_PATH}/data/raw/teaser_data/hindi_dataset.tsv"
}
DATA_PATHS_TEST = {
    "EN": f"{BASE_PATH}/data/raw/test_data/english_dataset.tsv",
    "DE": f"{BASE_PATH}/data/raw/test_data/german_dataset.tsv",
    "HI": f"{BASE_PATH}/data/raw/test_data/hindi_dataset.tsv"
}
print(DATA_PATHS_TRAIN)
DATA_COLUMNS = ["row_id", "text", "task_1", "task_2", "task_3"]
NUM_LANGUAGES = len(DATA_PATHS_TRAIN)


TASK_LABEL_IDS = {
    "task_1": ["NOT", "HOF"],
    "task_2": ["HATE", "OFFN", "PRFN"],
    "task_3": ["TIN", "UNT"],
        "task_4": [
        "NOT-NONE-NONE", 
        "HOF-HATE-TIN", "HOF-HATE-UNT", 
        "HOF-OFFN-TIN", "HOF-OFFN-UNT", 
        "HOF-PRFN-TIN", "HOF-PRFN-UNT", 
    ]
}
        
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir, file_key="train"):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        df = pd.read_csv(input_file, sep="\t", header=None)
        return df.values.tolist()


class ClassificationProcessor(DataProcessor):
    """Processor for binary classification dataset."""

    def __init__(self, task_labels):
        self.labels = task_labels

    def get_examples(self, data_dir, file_key="train"):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, f"{file_key}.tsv"))
        examples = self._create_examples(lines, file_key)
        return examples

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples     
        
        
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def load_and_cache_examples(args, tokenizer, file_key="train"):
    data_dir = args.data_dir
    task = args.task
    processor = ClassificationProcessor(TASK_LABEL_IDS[task])
    output_mode = 'classification'
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.cache_dir, f"cached_{file_key}_{args.bert_model_type}_{args.max_seq_length}_{task}")
    if os.path.exists(cached_features_file):
        logger.info(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating features from dataset file at {data_dir}", )
        examples = processor.get_examples(data_dir, file_key)
        label_list = processor.get_labels()
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def evaluate(eval_dataset, model, args, task_labels=None):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    device = "cuda:0"
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    clf_report = classification_report(
        y_true=out_label_ids, y_pred=preds, output_dict=True,
        target_names=task_labels, labels=np.arange(len(task_labels))
    )
    results = {"eval_loss": eval_loss}
    for lbl, score_dict in clf_report.items():
      if isinstance(score_dict, float):
        results[lbl] = score_dict
        continue
      for score, v in score_dict.items():
        results[f"{lbl}/{score}"] = v 
    return results
    
    
def predict(eval_dataset, model, args, task_labels=None):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    device = "cuda:0"
    for batch in tqdm(eval_dataloader, desc="Predicting", disable=True):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds_probs = softmax(preds, axis=1)
    preds = np.argmax(preds_probs, axis=1)
    preds_labels = np.array(task_labels)[preds]
    clf_report = classification_report(
        y_true=out_label_ids, y_pred=preds, output_dict=True,
        target_names=task_labels, labels=np.arange(len(task_labels))
    )
    results = {"eval_loss": eval_loss}
    for lbl, score_dict in clf_report.items():
      if isinstance(score_dict, float):
        results[lbl] = score_dict
        continue
      for score, v in score_dict.items():
        results[f"{lbl}/{score}"] = v 
    return results, preds_probs, preds_labels

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(train_dataset, model, tokenizer, args, eval_dataset=None, task_labels=None):
    """ Train the model """
    save_steps = args.save_steps
    data_dir = args.data_dir
    # remove old event files
    for event_file in glob.glob(f"{args.output_dir}/events.out.tfevents.*"):
        os.remove(event_file)
    tb_writer = SummaryWriter(args.output_dir)
    train_batch_size = args.train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    max_steps = args.max_steps
    num_train_epochs = args.num_train_epochs
    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    device = 'cuda:0'

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(num_train_epochs), desc="Epoch", disable=False)
    set_seed(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
                
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    result_loss = (tr_loss - logging_loss)/args.logging_steps
                    # Log metrics
                    eval_type="train"
                    results = evaluate(train_dataset, model, args, task_labels=task_labels)
                    results["global_step"] = global_step
                    results["loss"] = result_loss
                    train_iterator.set_postfix(results)
                    epoch_iterator.set_postfix(dict(eval_type=eval_type, global_step=global_step))
                    with open(os.path.join(args.output_dir, f"{eval_type}_results.json"), "a") as results_fp:
                      print(json.dumps(results), file=results_fp)
                    for key, value in results.items():
                        tb_writer.add_scalar(f"{eval_type}/{key}", value, global_step)
                    
                    eval_type="dev"
                    results = evaluate(eval_dataset, model, args, task_labels=task_labels)
                    #print(f"[dev]\tglobal_step={global_step}, results={results}")
                    results["global_step"] = global_step
                    results["loss"] = result_loss
                    train_iterator.set_postfix(results)
                    epoch_iterator.set_postfix(dict(eval_type=eval_type, global_step=global_step))
                    with open(os.path.join(args.output_dir, f"{eval_type}_results.json"), "a") as results_fp:
                      print(json.dumps(results), file=results_fp)
                    for key, value in results.items():
                        tb_writer.add_scalar(f"{eval_type}/{key}", value, global_step)
                        
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', result_loss, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    tb_writer.close()
    return global_step, tr_loss / global_step, model



def get_arguments():
    import argparse
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model_type", default=None, type=str, required=True,
                        help="Bert Model type selected in the list: ")
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: task_1, task_2, task_3")
    parser.add_argument("--lang", default=None, type=str, required=True,
                        help="The name of the lang to train selected in the list: EN, DE, HI")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()
    return args


def setup_tensorboard(args):
    from tensorboard import notebook, manager
    import signal
    import shutil
    # Kill tensorboard
    for info in manager.get_all():
        data_source = manager.data_source_from_info(info)
        print(f"port {info.port}: {data_source} (pid {info.pid})")
        if data_source == "logdir {args.output_dir}":
            pid = info.pid
            logger.info(f"Killing tensorboard at pid: {pid}")
            os.kill(pid, signal.SIGKILL)
            break
    # Delete output directory
    if os.path.exists(args.output_dir):
        logger.info(f"Deleting {args.output_dir}")
        shutil.rmtree(args.output_dir)
    logger.info(f"Creating {args.output_dir}")
    os.makedirs(args.output_dir)
    # Start notebook
    notebook.start(f"--logdir {args.output_dir}")
    # Kill tensorboard
    for info in manager.get_all():
        data_source = manager.data_source_from_info(info)
        print(f"port {info.port}: {data_source} (pid {info.pid})")
        if data_source == "logdir {args.output_dir}":
            port = info.port
            print()
            notebook.display(port=port, height=1000)
            break
    
def train_model(args):
    #setup_tensorboard(args)
    torch.cuda.empty_cache()
    lang, task = args.lang, args.task
    task_labels = TASK_LABEL_IDS[task]
    processor = ClassificationProcessor(task_labels)
    #DATA_DIR = os.path.join("./", lang, task)
    
    BERT_MODEL_TYPE=args.bert_model_type
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_TYPE, do_lower_case=args.do_lower_case)
    
    train_dataset = load_and_cache_examples(args, tokenizer, file_key="train")
    eval_dataset = load_and_cache_examples(args, tokenizer, file_key="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, file_key="test")

    model = BertForSequenceClassification.from_pretrained(args.bert_model_type, num_labels=len(task_labels))
    model.to('cuda:0')

    global_step, tr_loss, model = train(train_dataset, model, tokenizer, args, eval_dataset, task_labels=task_labels)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    
    model_dir = os.path.join(args.output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    print(f"saving model in {model_dir}")
    model.save_pretrained(model_dir)  # save
    
    for test_key, dataset in zip(["test", "dev", "train"], [test_dataset, eval_dataset, train_dataset]):
        results, preds_probs, preds = predict(dataset, model, args, task_labels=task_labels)
        print(preds_probs.shape, preds.shape)
        print(f"{test_key}: {results}")
        df_test = pd.read_csv(os.path.join(args.data_dir, f"{test_key}.tsv"), sep="\t", header=None, names=["id", "label", "alpha", "text"])
        df_test = df_test.assign(**{f"{task}_preds": preds})
        df_test = df_test.drop(["alpha", "text"], axis=1)
        df_probs = pd.DataFrame(preds_probs, columns=[f"{l}_probs" for l in task_labels])
        df_test = pd.concat([df_test, df_probs], axis=1)
        df_test.to_csv(os.path.join(args.output_dir, f"{test_key}.tsv"), sep="\t", index=False)
    
    
    
if __name__ == "__main__":
    args = get_arguments()
    train_model(args)