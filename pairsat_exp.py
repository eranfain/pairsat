import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import argparse
from itertools import cycle
from transformers import AutoTokenizer, ModernBertModel
from torch.utils.data import Dataset, DataLoader
from eval_utils import sat_evaluation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using", DEVICE)
torch.set_float32_matmul_precision('high')


def load_data(dataset_name):
    dirname = f'dataset/{dataset_name}'
    print("Reading", dataset_name, "dataset")

    result = dict()
    for set_name in ['train', 'valid', 'test']:
        data_list = list()
        with open(os.path.join(dirname, f'{set_name}_{dataset_name}.txt'), 'r', encoding='utf-8') as infile:
            for line in infile:
                items = line.strip('\n').split('\t')
                input_text = eval(items[0])
                sat = int(items[2])
                history = ''
                for text in input_text:
                    user_utt = text.split('|||')[0]
                    ai_utt = text.split('|||')[1]
                    if ai_utt:
                        history += f'\n\nHuman: {user_utt}'
                        history += f'\n\nAssistant: {ai_utt}'

                if len(user_utt.strip()) > 0 and user_utt.strip() != 'OVERALL':
                    data_list.append({
                        'history': history.strip(),
                        'utterance': user_utt.strip(),
                        'full_conv': f'{history.strip()}\n\nHuman: {user_utt.strip()}',
                        'label': sat
                    })

            result[set_name] = pd.DataFrame(data_list)

        print('{} set, len: {}'.format(set_name, len(result[set_name])))

    return result


# ==== DATASETS ====
class ConversationOnlyDataset(Dataset):
    def __init__(self, texts, max_length, tokenizer):
        self.texts = texts
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


class LabeledDataset(Dataset):
    def __init__(self, conversations, labels, tokenizer, max_len):
        self.conversations = conversations
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        conv = self.conversations[idx]
        label = self.labels[idx]
        enc = self.tokenizer(conv, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(label, dtype=torch.float)


class PreferenceDataset(Dataset):
    def __init__(self, accepted_convs, rejected_convs, tokenizer, max_len):
        self.accepted = accepted_convs
        self.rejected = rejected_convs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.accepted)

    def __getitem__(self, idx):
        acc = self.accepted[idx]
        rej = self.rejected[idx]
        acc_enc = self.tokenizer(acc, truncation=True, padding='max_length', max_length=self.max_len,
                                 return_tensors='pt')
        rej_enc = self.tokenizer(rej, truncation=True, padding='max_length', max_length=self.max_len,
                                 return_tensors='pt')
        return {k: v.squeeze(0) for k, v in acc_enc.items()}, {k: v.squeeze(0) for k, v in rej_enc.items()}


# ==== MODEL ====
class SatisfactionModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = ModernBertModel.from_pretrained(model_name)
        self.reg_head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        score = self.reg_head(cls_output).squeeze(-1)
        score = torch.tanh(score)
        return score


# ==== TRAINING ====
def evaluate(model, evaluation_data, tokenizer, max_len, dataset_name=None, set_name=None, lamb=None,
             should_save_preds=False):
    print("evaluating")
    eval_dataset = ConversationOnlyDataset(texts=evaluation_data['full_conv'].values, max_length=max_len,
                                           tokenizer=tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=8)
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds = model(**batch)
            all_preds.append(preds.cpu())

    all_preds = torch.cat(all_preds)
    final_preds = torch.clamp(all_preds.round(), -1, 1).long() + 1
    if should_save_preds:
        evaluation_data['pred'] = final_preds
        evaluation_data.to_csv(f'pairsat_{dataset_name}_{set_name}_{lamb}.csv', index=False)

    return sat_evaluation(final_preds, evaluation_data['label'].values, sat_num=3)


def train(model, tokenizer, max_len, labeled_loader, pref_loader, optimizer, report_loss_every, evaluate_every,
          validation_data, lamb, margin, dataset_name, patience, epochs=1):
    model.train()
    best_f1 = 0.0
    should_finish = False
    for epoch in range(epochs):
        if should_finish:
            break

        report_counter = 0
        evaluate_counter = 0
        total_loss = 0.0
        temp_loss = 0.0
        curr_patience = 0

        # Cycle through the smaller loader to match the larger one
        labeled_iter = cycle(labeled_loader)
        pref_iter = cycle(pref_loader)
        num_batches = max(len(labeled_loader), len(pref_loader))

        for _ in tqdm(range(num_batches)):
            report_counter += 1
            evaluate_counter += 1
            if evaluate_counter % evaluate_every == 0:
                sat_results = evaluate(model=model, evaluation_data=validation_data, tokenizer=tokenizer,
                                       max_len=max_len)
                f1_result = sat_results[-1]
                model.train()  # go back to training
                if f1_result > best_f1:
                    print(f"Got better F1 score ({f1_result} > {best_f1})")
                    best_f1 = f1_result
                    torch.save(model.state_dict(), f"use_modern_bert_{dataset_name}_{lamb}.pth")
                    curr_patience = 0
                elif patience == curr_patience:
                    should_finish = True
                    break

                curr_patience += 1

            if report_counter % report_loss_every == 0:
                report_counter = 0
                print("Current total loss: ", temp_loss)
                temp_loss = 0.0

            # Get batches
            sup_batch, labels = next(labeled_iter)
            acc_batch, rej_batch = next(pref_iter)

            # Move to device
            sup_inputs = {k: v.to(DEVICE) for k, v in sup_batch.items()}
            labels = labels.to(DEVICE)
            acc_inputs = {k: v.to(DEVICE) for k, v in acc_batch.items()}
            rej_inputs = {k: v.to(DEVICE) for k, v in rej_batch.items()}

            # Forward pass
            sup_preds = model(**sup_inputs)
            acc_scores = model(**acc_inputs)
            rej_scores = model(**rej_inputs)

            # Compute losses
            supervised_loss = F.mse_loss(sup_preds, labels)
            ranking_loss = F.margin_ranking_loss(
                acc_scores, rej_scores,
                target=torch.ones_like(acc_scores),
                margin=margin
            )

            loss = lamb * supervised_loss + (1 - lamb) * ranking_loss
            loss_item = loss.item()
            total_loss += loss_item
            temp_loss += loss_item

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Combined Loss: {total_loss:.4f}")

def run(
        dataset_name,
        pairwise_dataset_path="dataset/anthropic_hh/train-00000-of-00001-8349d0765e6718df.parquet",
        model_name="answerdotai/ModernBERT-base",
        max_len=1024,
        lambda_val=0.6,
        report_loss_every=100,
        evaluate_every=800,
        margin=0.5,
        overall_output_file="results.csv",
        batch_size=4,
        patience=5,
        lr=2e-5,
):
    # load tokenizer
    print("loading tokenizer:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load satisfaction training data (with satisfaction labels)
    print("loading labeled SAT for", dataset_name)
    data = load_data(dataset_name)
    train_conversations = data['train']['full_conv'].values
    train_labels = [x - 1 for x in data['train']['label'].values]  # to map to [-1, 1]

    # print an example conversation
    print("Example train conversation from", dataset_name)
    print(train_conversations[0])
    print("SAT score: ", train_labels[0], "(1=Satisfied, 0=Neutral, -1=Dissatisfied)")
    print("=========================")

    # create data loader
    sat_labeled_dataset = LabeledDataset(conversations=train_conversations, labels=train_labels, tokenizer=tokenizer,
                                         max_len=max_len)
    sat_labeled_loader = DataLoader(sat_labeled_dataset, batch_size=batch_size, shuffle=True)

    # Load Anthropic HH dataset
    df_train_pref = pd.read_parquet(pairwise_dataset_path)
    df_train_pref['full_conv_accepted'] = df_train_pref.apply(lambda r: f"{r['prompt']}{r['chosen']}".strip(), axis=1)
    df_train_pref['full_conv_rejected'] = df_train_pref.apply(lambda r: f"{r['prompt']}{r['rejected']}".strip(), axis=1)
    print("Example pair from preference dataset")
    print(f"Accepted:\n{df_train_pref['full_conv_accepted'][0]}")
    print()
    print(f"Rejected:\n{df_train_pref['full_conv_rejected'][0]}")

    # create data loader
    preference_dataset = PreferenceDataset(accepted_convs=df_train_pref['full_conv_accepted'].values,
                                           rejected_convs=df_train_pref['full_conv_rejected'].values,
                                           tokenizer=tokenizer, max_len=max_len)
    preference_loader = DataLoader(preference_dataset, batch_size=batch_size, shuffle=True)

    # run training
    print("start training")
    sat_model = SatisfactionModel(model_name=model_name).to(DEVICE)
    optimizer = torch.optim.AdamW(sat_model.parameters(), lr=lr)
    print("start training")
    train(model=sat_model, labeled_loader=sat_labeled_loader, pref_loader=preference_loader,
          validation_data=data['valid'],
          optimizer=optimizer, report_loss_every=report_loss_every, evaluate_every=evaluate_every, lamb=lambda_val,
          margin=margin, dataset_name=dataset_name, patience=patience, tokenizer=tokenizer, max_len=max_len)

    # load best model from disk and evaluate
    # free up the model's memory in GPU to enable reloading
    del optimizer
    del sat_model
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    sat_model = SatisfactionModel(model_name=model_name)
    sat_model.load_state_dict(torch.load(f"use_modern_bert_{dataset_name}_{lambda_val}.pth"))
    sat_model = sat_model.to(DEVICE)
    eval_results = evaluate(model=sat_model, evaluation_data=data['test'], dataset_name=dataset_name,
                            set_name='test', lamb=lambda_val, should_save_preds=True, tokenizer=tokenizer,
                            max_len=max_len)

    # Open the CSV file in append mode
    with open(overall_output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([dataset_name, model_name, lambda_val] + eval_results)

    print(f'Test results for lambda={lambda_val}: ', eval_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run PAIRSAT model training and evaluation.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Which dataset to use. Can be either: mwoz, redial or sgd.")
    parser.add_argument("--pairwise_dataset_path", type=str, required=True,
                        default="dataset/anthropic_hh/train-00000-of-00001-8349d0765e6718df.parquet",
                        help="Path to the pairwise preference dataset to use. Should be in a parquet format.")
    parser.add_argument("--overall_output_file", type=str, required=True, default="results.csv",
                        help="Path to the overall output CSV file.")
    parser.add_argument("--model_name", type=str, required=True, default="answerdotai/ModernBERT-base",
                        help="Encoder model name to use for fine-tune.")
    parser.add_argument("--max_len", type=int, required=True, default=1024,
                        help="Max number of tokens to use for input.")
    parser.add_argument("--lambda_val", type=float, required=True, default=0.6,
                        help="Lambda value to use defines the ratio between the MSE loss (calculated based on the"
                             " labeled data) and the margin ranking loss (calculated based on the pairwise data).")
    parser.add_argument("--report_loss_every", type=int, required=True, default=100,
                        help="Number of steps to make between every loss reporting.")
    parser.add_argument("--evaluate_every", type=int, required=True, default=800,
                        help="Number of steps to make between making an evaluation on validation set.")
    parser.add_argument("--margin", type=float, required=True, default=0.5,
                        help="Margin to use for the margin ranking loss (calculated based on the pairwise data).")
    parser.add_argument("--lr", type=float, required=True, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--batch_size", type=int, required=True, default=4,
                        help="Batch size.")
    parser.add_argument("--patience", type=int, required=True, default=5,
                        help="If validation F1 metric is not getting better after this number of evaluations, training"
                             "is stopped.")
    args = parser.parse_args()

    run(**vars(args))
