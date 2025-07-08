import os
import csv
import time
import json
import joblib
import argparse
import numpy as np
import pandas as pd
from typing import List, Optional
from ast import literal_eval
from openai import AzureOpenAI
from functools import lru_cache
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from eval_utils import sat_evaluation

subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
completion_endpoint = os.getenv("COMPLETION_ENDPOINT_URL")

# Initialize Azure OpenAI client with key-based authentication
completion_client = AzureOpenAI(
    azure_endpoint=completion_endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

emb_endpoint = os.getenv("EMBEDDING_ENDPOINT_URL")
api_version = "2024-02-01"

emb_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=emb_endpoint,
    api_key=subscription_key
)


def run_completion(messages, temperature, max_tokens, completion_model_name):
    try:
        completion = completion_client.chat.completions.create(
            model=completion_model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        return completion
    except Exception as e:
        print("FAILED COMPLETION: ")
        print(e)
        print(messages)
        raise e


def load_data(dataset_name):
    dirname = f'dataset/{dataset_name}'
    print("Reading", dataset_name, "dataset")

    result = dict()
    for set_name in ['train', 'valid', 'test']:
        if os.path.exists(f'dataset/{dataset_name}/{dataset_name}_{set_name}.gzip'):
            result[set_name] = pd.read_csv(f'dataset/{dataset_name}/{dataset_name}_{set_name}.gzip', compression='gzip')
            if 'utterance_emb' in result[set_name].columns:
                result[set_name]['utterance_emb'] = result[set_name]['utterance_emb'].apply(lambda arr: eval(arr))
        else:
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
                            history += f'\nUser: {user_utt}'
                            history += f'\nAssistant: {ai_utt}'

                    if len(user_utt.strip()) > 0 and user_utt.strip() != 'OVERALL':
                        data_list.append({
                            'history': history.strip(),
                            'utterance': user_utt.strip(),
                            'label': sat
                        })

                result[set_name] = pd.DataFrame(data_list)

        print('{} set, len: {}'.format(set_name, len(result[set_name])))

    return result


class GPTResponse:
    def __init__(self, response):
        self.response_text = response.choices[0].message.content
        self.input_tokens = response.usage.prompt_tokens
        self.output_tokens = response.usage.completion_tokens


GREAT_PLANNER_PROMPT = """{{$problem_definition}}

[ouput format]
Your answer should be in the following json format
{
    "strategies": [
        "User [common verb] [appropriate object less than 5 words].",
        "User [common verb] [appropriate object less than 5 words].",
        ...
    ]
}

Below are the strategies created so far
[Effective strategies]
{{$effective_strategies}}

[Ineffective strategies]
{{$ineffective_strategies}}

Generate {{$strategy_num}} additional effective strategies that you think would help your analysis.
answer:"""

UNORTHODOX_PLANNER_PROMPT = """{{$problem_definition}}

[ouput format]
Your answer should be in the following json format
{
    "strategies": [
        "User [common verb] [appropriate object that fits the strategy]].",
        "User [common verb] [appropriate object that fits the strategy]].",
        ...
    ]
}

Below are the strategies created so far
[Effective strategies]
{{$effective_strategies}}

[Ineffective strategies]
{{$ineffective_strategies}}

In our opinion, the above strategies are too formulaic, and sometimes crazy strategies that are completely weird or nonsensical are more successful.

Generate {{$strategy_num}} strategies that sound like conversations you'd have in a problem definition situation, but don't seem to have anything to do with user satisfaction.
answer:"""

INITIAL_STRATEGIES = {
    'mwoz': ["User thanks the assistant.", "User repeats the same question.",
             "User asks about other services."],
    'redial': ["User asks for more movie recommendations.", "User expresses interest in a movie’s director.",
               "User compliments assistant’s choice.", "User requests further details on movie.",
               "User expresses interest in a specific genre."],
    'sgd': ["User expresses satisfaction with the service quality.", "User acknowledges assistant’s quick thinking.",
            "User shows appreciation for assistance.", "User empathizes with the assistant",
            "User appreciates the detailed explanation."]
}

COMMON_PREFIX_PROMPT = "You are a competent bot that generates strategies to classify conversations in which the user expresses satisfaction."
PROBLEM_FORMULATION = {
    'mwoz': f"""{COMMON_PREFIX_PROMPT}
The User and Assistant are having a conversation about making a reservation for a specific service, or looking up information such
as an address or phone number.
The types of services include taxis, restaurants, buses, hotels, attractions, and trains.
The user asks a number of questions about the service, and their satisfaction depends on the assistant’s answers.
Users are satisfied if the assistant answers their questions appropriately, but they are also dissatisfied if the service provider does not
provide the information they asked for, regardless of the assistant’s answer.""",
    'sgd': f"""{COMMON_PREFIX_PROMPT}
Assistant is a virtual assistant that provides information about Alarm, Bank, transportation(bus, flight, etc.), reservation(rental car,
restaurant etc.), Calendar, Event, Home, Hotel, Media, Movie, Music, Service, Travel, Weather and many other things people might
want to know.
A typical satisfaction for a user is when they successfully make a reservation or find the assistant’s suggestions helpful, and sometimes
they are dissatisfied with the assistant’s answer and ask for another alternative or decline.
Include specific context in your strategy for the information the assistant provides. (e.g. user requests a bus at a different time.)""",
    'redial': f"""{COMMON_PREFIX_PROMPT}
The user and the assistant have a conversation about movies, talking about the movies they’ve seen or recommending movies to each other.
The Assistant’s suggestions, questions, and reactions have a significant impact on the user’s satisfaction, which can be inferred from the user’s conversations.
The main topics of conversation are the title, actors, and genre of the movie, but they also include casual conversation."""
}

PASSAGE_GENERATOR_PROMPT = """[query]
{{$query}}

Create 5 messages that you think would come up as search results if I were to search for messages that match the query.
The messages should be very natural, colloquial, and provided in bullet type.
Answers should be of varying lengths, including short sentences of two to three words and longer sentences using up to 10 words.
your answers:"""


class PRAISE:
    def __init__(
            self,
            dataset_name,
            overall_output_file,
            initial_eps=0.1,
            ns=5,
            top_k=50,
            model_max_iter=700,
            model_c=100,
            best_validation_score=0,
            eps=0.1,
            emb_model_name="text-embedding-3-large",
            passage_gen_model_name="gpt-3.5-turbo-0125",
            planner_model_name="gpt-4-1106-preview",
            effective_strategies=None,
            ineffective_strategies=None,
            passages=None,
            model=None,
            total_input_tokens=0,
            total_output_tokens=0,
            patience=0,
            iteration=0,
            curr_eval_score=0,
            **kwargs
    ):
        self.dataset_name = dataset_name
        self.overall_output_file = overall_output_file
        self.best_validation_score = best_validation_score
        self.emb_model_name = emb_model_name
        self.passage_gen_model_name = passage_gen_model_name
        self.planner_model_name = planner_model_name
        self.eps = eps
        self.initial_eps = initial_eps
        self.ns = ns
        self.top_k = top_k
        self.effective_strategies = effective_strategies or INITIAL_STRATEGIES.get(dataset_name)
        self.ineffective_strategies = ineffective_strategies or []
        self.passages = passages or {}
        self.model_max_iter = model_max_iter
        self.model_c = model_c
        self.model: Optional[LogisticRegression] = model
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens
        self.best_validation_score = best_validation_score
        self.patience = patience
        self.iteration = iteration
        self.curr_eval_score = curr_eval_score

    @staticmethod
    def get_embeddings(texts, emb_model_name, batch_size=100):
        result = list()
        batch = None
        for i in range(0, len(texts), batch_size):
            success = False
            while not success:
                try:
                    batch = texts[i:i + batch_size]
                    emb_response = emb_client.embeddings.create(input=batch, model=emb_model_name, dimensions=1024)
                    result.extend([r.embedding for r in emb_response.data])
                    success = True
                except Exception as e:
                    print(e)
                    print(batch)
                    time.sleep(5)
        return result

    def update_token_counters(self, input_tokens, output_tokens):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    @staticmethod
    def get_response_text_json(gpt_response: GPTResponse):
        gpt_response_text: str = gpt_response.response_text
        json_start = gpt_response_text.index('{')
        json_end = gpt_response_text.rfind("}")
        return literal_eval(gpt_response_text[json_start:json_end+1])

    def generate_strategies_great_planner(self) -> List[str]:
        prompt = (
            GREAT_PLANNER_PROMPT
            .replace("{{$problem_definition}}", PROBLEM_FORMULATION.get(self.dataset_name))
            .replace("{{$effective_strategies}}", "\n".join(self.effective_strategies))
            .replace("{{$ineffective_strategies}}", "\n".join(self.ineffective_strategies))
            .replace("{{$strategy_num}}", str(self.ns))
        )
        response = run_completion(
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            completion_model_name=self.planner_model_name,
            temperature=0.1,
            max_tokens=512
        )
        gpt_response = GPTResponse(response)
        self.update_token_counters(gpt_response.input_tokens, gpt_response.output_tokens)
        return self.get_response_text_json(gpt_response)['strategies']

    def generate_strategies_unorthodox(self) -> List[str]:
        prompt = (
            UNORTHODOX_PLANNER_PROMPT
            .replace("{{$problem_definition}}", PROBLEM_FORMULATION.get(self.dataset_name))
            .replace("{{$effective_strategies}}", "\n".join(self.effective_strategies))
            .replace("{{$ineffective_strategies}}", "\n".join(self.ineffective_strategies))
            .replace("{{$strategy_num}}", str(self.ns))
        )
        response = run_completion(
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            completion_model_name=self.planner_model_name,
            temperature=0.7,
            max_tokens=512
        )
        gpt_response = GPTResponse(response)
        self.update_token_counters(gpt_response.input_tokens, gpt_response.output_tokens)
        return self.get_response_text_json(gpt_response)['strategies']

    def generate_strategies(self) -> List[str]:
        # The epsilon serves as the exploration ratio
        if self.patience == 0:
            self.eps = self.initial_eps
        else:
            self.eps = min(1.0, self.eps * 2)  # if the validation score

        if np.random.rand() < self.eps:
            return self.generate_strategies_unorthodox()
        else:
            return self.generate_strategies_great_planner()

    @lru_cache(maxsize=None)
    def generate_passages_for_strategy(self, strategy):
        prompt = PASSAGE_GENERATOR_PROMPT.replace("{{$query}}", strategy)
        response = run_completion(
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            completion_model_name=self.passage_gen_model_name,
            temperature=0.0,
            max_tokens=1024
        )
        gpt_response = GPTResponse(response)
        self.update_token_counters(gpt_response.input_tokens, gpt_response.output_tokens)
        return gpt_response.response_text

    def get_passages_from_response(self, passage_response: str) -> List[str]:
        # generate passages for given strategies
        passge_set = set()
        # post process the passages
        # remove " from the beginning and end
        # remove number from the beginning (e.g., "1. Thank you for that!" -> "Thank you for that!")
        for passage in passage_response.split('\n'):
            passage_text = passage[2:].strip()
            if passage_text.startswith('"'):
                passage_text = passage_text[1:]
            if passage_text.endswith('"'):
                passage_text = passage_text[:-1]
            passage_text = passage_text.strip()
            if passage_text:
                passge_set.add(passage_text)

        return list(passge_set)

    def generate_passages(self, new_strategies):
        for strategy in new_strategies:
            passage_response = self.generate_passages_for_strategy(strategy)
            self.passages[strategy] = self.get_passages_from_response(passage_response)

    def compute_features(self, df, set_name, dataset_name, only_effective_strategies=False):
        if 'utterance_emb' not in df.columns:
            df['utterance_emb'] = self.get_embeddings(list(df['utterance'].values), emb_model_name=self.emb_model_name)
            df.to_csv(f'dataset/{dataset_name}/{dataset_name}_{set_name}.gzip', compression='gzip', index=False)

        utterance_embeddings = list(df['utterance_emb'].values)
        for strategy, passages in self.passages.items():
            if (strategy in df.columns) or (only_effective_strategies and strategy not in self.effective_strategies):
                continue

            passage_embeddings = self.get_embeddings(passages, emb_model_name=self.emb_model_name)
            sims = cosine_similarity(passage_embeddings, utterance_embeddings)
            df[strategy] = np.array(sims.sum(axis=0))

        return df

    def evaluate_and_select_strategies(self, df_train, df_val, new_strategies) -> float:
        if self.model is None:
            # make a first time fit
            X_train, y_train = df_train[self.effective_strategies].values, df_train['label'].values
            self.model = LogisticRegression(max_iter=self.model_max_iter, C=self.model_c, penalty='l2')
            self.model.fit(X_train, y_train)

            X_val, y_val = df_val[self.effective_strategies].values, df_val['label'].values
            baseline_score = self.model.score(X_val, y_val)
        else:
            baseline_score = self.curr_eval_score

        selected_strategies = []
        for i, strategy in enumerate(new_strategies):
            X_train, y_train = df_train[self.effective_strategies + [strategy]].values, df_train['label'].values
            X_val, y_val = df_val[self.effective_strategies + [strategy]].values, df_val['label'].values
            score = LogisticRegression(max_iter=self.model_max_iter, C=self.model_c, penalty='l2').fit(X_train, y_train).score(X_val,
                                                                                                                 y_val)
            if score > baseline_score:
                selected_strategies.append(strategy)

        # Train a model based on a combination of all selected features
        self.model = LogisticRegression(max_iter=self.model_max_iter, C=self.model_c, penalty='l2')
        combined_strategies = self.effective_strategies + selected_strategies
        X_train, y_train = df_train[combined_strategies].values, df_train['label'].values
        X_val, y_val = df_val[combined_strategies].values, df_val['label'].values
        self.model.fit(X_train, y_train)

        # Select only top_k features if needed to avoid strategy floating
        if len(combined_strategies) >= self.top_k:
            # find the top_k features and update effective vs ineffective strategies
            summed_coefs = np.sum(np.abs(self.model.coef_), axis=0)
            top_k_indices = np.argsort(summed_coefs)[-self.top_k:][::-1]
            self.effective_strategies = [combined_strategies[i] for i in top_k_indices]
            self.ineffective_strategies = list(set(self.passages.keys()) - set(self.effective_strategies))

            # Train another model based on selected features
            X_train, y_train = df_train[self.effective_strategies].values, df_train['label'].values
            X_val, y_val = df_val[self.effective_strategies].values, df_val['label'].values
            self.model = LogisticRegression(max_iter=self.model_max_iter, C=self.model_c, penalty='l2')
            self.model.fit(X_train, y_train)
        else:
            self.effective_strategies = combined_strategies
            self.ineffective_strategies = list(set(self.passages.keys()) - set(self.effective_strategies))

        print("New effective strategies: ", self.effective_strategies)
        print("New ineffective strategies: ", self.ineffective_strategies)
        return self.model.score(X_val, y_val)

    def infer(self, df_test):
        df_test = self.compute_features(df_test, 'test', dataset_name=self.dataset_name, only_effective_strategies=True)
        X_test = df_test[self.effective_strategies].values
        return self.model.predict(X_test)

    def save_model(self):
        model_dir = f"models/{self.dataset_name}"
        os.makedirs(model_dir, exist_ok=True)
        model_atts = dict(self.__dict__)
        model_atts.pop('model')
        with open(f"{model_dir}/praise_atts.json", 'w') as f:
            json.dump(model_atts, f)
        joblib.dump(self.model, f"{model_dir}/best_model.joblib")

    def run(self, data, num_iterations=50, early_stopping=5):
        print(f"Running PRAISE on {self.dataset_name}")

        # extract dfs from data
        df_train, df_val, df_test = data['train'], data['valid'], data['test']

        # generate the passages and embeddings for the initial strategies
        self.generate_passages(self.effective_strategies)
        df_train = self.compute_features(df_train, 'train', self.dataset_name)
        df_val = self.compute_features(df_val, 'valid', self.dataset_name)

        while (self.iteration < num_iterations) and (self.patience < early_stopping):
            print(f"Iteration: {self.iteration}")
            strategies = set(self.generate_strategies())
            new_strategies = list(strategies - set(self.passages.keys()))
            print("New strategies: ", new_strategies)
            self.generate_passages(new_strategies)
            df_train = self.compute_features(df_train, 'train', self.dataset_name)
            df_val = self.compute_features(df_val, 'valid', self.dataset_name)
            self.curr_eval_score = self.evaluate_and_select_strategies(df_train, df_val, new_strategies)
            if self.curr_eval_score > self.best_validation_score:
                self.best_validation_score = self.curr_eval_score
                self.patience = 0
                self.save_model()
            else:
                self.patience += 1
            self.iteration += 1

        # run inference
        df_test['pred'] = self.infer(df_test)
        eval_results = sat_evaluation(df_test['pred'].values, df_test['label'].values, sat_num=3)

        # Open the CSV file in append mode
        with open(self.overall_output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.dataset_name] + eval_results)

        print(f'Test results for {self.dataset_name}: ', eval_results)


def load_praise_from_disk_if_exists(dataset_name, use_checkpoint=True, **kwargs):
    best_model_path = f"models/{dataset_name}"
    if use_checkpoint and os.path.exists(f'{best_model_path}/praise_atts.json'):
        with open(f'{best_model_path}/praise_atts.json', 'r') as f:
            attrs = json.load(f)
        result = PRAISE(**attrs)
        result.model = joblib.load(f'{best_model_path}/best_model.joblib')
    else:
        result = PRAISE(dataset_name=dataset_name, **kwargs)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run PRAISE model training and evaluation.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Which dataset to use. Can be either: mwoz, redial or sgd.")
    parser.add_argument("--overall_output_file", type=str, required=True, default="results.csv",
                        help="Path to the overall output CSV file.")
    parser.add_argument("--num_iterations", type=int, required=True, default=50,
                        help="Number of iterations to run the algorithm.")
    parser.add_argument("--early_stopping", type=int, required=True, default=5,
                        help="Value for early stopping.")
    parser.add_argument("--emb_model_name", type=str, required=True, default="text-embedding-3-large",
                        help="Model name to use for embedding.")
    parser.add_argument("--passage_gen_model_name", type=str, required=True, default="gpt-3.5-turbo-0125",
                        help="Model name to use for passage generation.")
    parser.add_argument("--planner_model_name", type=str, required=True, default="gpt-4-1106-preview",
                        help="Model name to use for planners.")
    parser.add_argument("--use_checkpoint", action="store_true",
                        help="Flag to indicate whether to use checkpoint if exists.")
    parser.add_argument("--model_max_iter", type=int, required=True, default=700,
                        help="model_max_iter value used for regression. in original paper, recommended to use"
                             " 500 for mwoz, and 700 for sgd and redial.")
    parser.add_argument("--model_c", type=int, required=True, default=100,
                        help="model_c value used for the LR model. in original paper, they used 100.")
    parser.add_argument("--ns", type=int, required=True, default=5,
                        help="Number of new strategies to produce with every iteration.")
    parser.add_argument("--top_k", type=int, required=True, default=5,
                        help="Number of maximal features to use for the LR model.")
    parser.add_argument("--eps", type=int, required=True, default=5,
                        help="Epsilon value used to decide when to use great planner vs unorthodox planner.")
    args = parser.parse_args()

    dataset = args.dataset_name
    praise = load_praise_from_disk_if_exists(dataset_name=dataset, **vars(args))
    praise.run(load_data(dataset), num_iterations=args.args.num_iterations, early_stopping=args.early_stopping)
