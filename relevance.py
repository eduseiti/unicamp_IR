import argparse as ap

import re
import pandas as pd
import time
import json
import sys
import pickle
import os

import openai


class LLMAccess():

    LLM_EVALUATION_KEY_FORMAT = "{}_{}"

    def __init__(self, 
                 LLM_interface, 
                 input_1k_token_cost, 
                 output_1k_token_cost,
                 initial_prompt, 
                 query_passage_format, 
                 output_regex, 
                 examples,
                 example_format,
                 evaluations_cache=None,
                 verbose=False):
        
        self.LLM_interface = LLM_interface
        self.input_1k_token_cost = input_1k_token_cost
        self.output_1k_token_cost = output_1k_token_cost

        self.initial_prompt = initial_prompt
        self.query_passage_format = query_passage_format
        self.output_regex = output_regex
        self.examples = examples
        self.example_format = example_format

        self.verbose=verbose

        if evaluations_cache is not None:
            self.LLM_relevance_evaluations_cache=evaluations_cache
        else:
            self.LLM_relevance_evaluations_cache={}



    def compute_cost(self, token_usage):

        """Shall be implemented with the given LLM cost computation.
        """



    def initialize_LLM(self, api_key):

        """Shall be implemented with the given LLM initiialization steps.

            One example is API Key initialization.
        """



    def get_evaluation_cache(self):
        return self.LLM_relevance_evaluations_cache



    def passage_relevance_evaluation(self, 
                                     query, 
                                     passage, 
                                     number_of_completions, 
                                     verbose=None):

        """Shall be implemented with the given LLM passage relevance evaluation implementation.

           Output value shall be a dictionary in one of the following 2 format of single or multiple completions:

           {
            'reasoning': <string>,
            'score': <int>}, 
            'usage': {'prompt_tokens': <int>,
                      'completion_tokens': <int>,
                      'total_tokens': <int>},
            'cost': <float>,
            'duration': <float>
           }

           or

           {
            'LLM_responses':[
                             {'reasoning': <string>,
                              'score': <int>}, 
                              ...
                            ],
            'usage': {'prompt_tokens': <int>,
                      'completion_tokens': <int>,
                      'total_tokens': <int>},
            'cost': <float>,
            'duration': <float>
           }
        """



    def query_passage_evaluation(self,
                                 query_passages_df, 
                                 output_file=None, 
                                 number_of_completions=1,
                                 use_evaluation_cache=True,
                                 verbose=None):

        if verbose is None:
            verbose = self.verbose

        LLM_evaluations = {}    # This dictionary holds all the passages evaluated in this round

        if use_evaluation_cache:
            LLM_evaluations_cache = self.LLM_relevance_evaluations_cache
        else:
            # If not using the evaluation cache, just make sure the same document is not 
            # evaluated twice

            LLM_evaluations_cache = LLM_evaluations

        
        for i, row in query_passages_df.iterrows():
            
            document_key = LLMAccess.LLM_EVALUATION_KEY_FORMAT.format(row['query'], row['passage_id'])

            if verbose:
                print("Query/Passage evaluation {}; document_key={}...".format(i, document_key))

            if document_key not in LLM_evaluations_cache:
                try:
                    relevance_results = self.passage_relevance_evaluation(query=row['query'], 
                                                                          passage=row['passage'], 
                                                                          number_of_completions=number_of_completions,
                                                                          verbose=verbose)
                except Exception as e:
                    print(e)

                    if number_of_completions > 1:
                        time.sleep(60)
                    else:
                        time.sleep(30)

                    relevance_results = self.passage_relevance_evaluation(query=row['query'], 
                                                                          passage=row['passage'], 
                                                                          number_of_completions=number_of_completions,
                                                                          verbose=verbose)

                if (number_of_completions > 1) and ('score' not in relevance_results):
                    
                    rounded_score = 0
                    
                    for LLM_response in relevance_results['LLM_responses']:
                        rounded_score += LLM_response['score']

                    relevance_results['score'] = int(round(rounded_score / number_of_completions))
                    

                    if verbose:
                        print(">> Rounded score: {}\n\n\n".format(relevance_results['score']))
                
                relevance_results['saved_cost'] = 0.0

                if use_evaluation_cache:
                    # Save the new passage evaluation in the cache, if using it

                    LLM_evaluations_cache[document_key] = relevance_results
            else:

                # The query/passage tuple has already been evaluated

                if verbose:
                    print("-- LLM already evaluated document {}...\n".format(document_key))

                if use_evaluation_cache:

                    relevance_results = LLM_evaluations_cache[document_key].copy()

                    # Indicate the previous cost has now been saved

                    relevance_results['saved_cost'] = relevance_results['cost']
                    relevance_results['cost'] = 0.0
                else:
                    relevance_results = LLM_evaluations_cache[document_key]
                    

            LLM_evaluations[document_key] = relevance_results

            
            validation_results_df = pd.concat([query_passages_df.reset_index(drop=True), 
                                               pd.DataFrame.from_dict(LLM_evaluations, orient='index').reset_index(drop=True)], axis=1)
            

            if (output_file is not None) or (output_file != ""):
                validation_results_df.to_csv(output_file, sep='\t', index=False)
        
        return validation_results_df



class GPT4Access(LLMAccess):

    MODEL_NAME="gpt-4"
    EVALUATION_MAX_TOKENS_RESPONSE=500



    def __init__(self, 
                 initial_prompt, 
                 query_passage_format, 
                 output_regex, 
                 examples, 
                 example_format,
                 evaluations_cache=None,
                 verbose=True):
        
        super().__init__(openai, 
                         0.03, 
                         0.06, 
                         initial_prompt, 
                         query_passage_format, 
                         output_regex, 
                         examples, 
                         example_format=example_format,
                         evaluations_cache=evaluations_cache,
                         verbose=verbose)


    
    def initialize_LLM(self, api_key):
        self.LLM_interface.api_key = api_key



    def compute_cost(self, token_usage):

        cost =  token_usage['prompt_tokens'] / 1000 * self.input_1k_token_cost + \
                token_usage['completion_tokens'] / 1000 * self.output_1k_token_cost
        
        return cost
    


    def passage_relevance_evaluation(self, 
                                     query, 
                                     passage, 
                                     number_of_completions, 
                                     verbose=None):

        if verbose is None:
            verbose = self.verbose
        

        start_time = time.time()

        query_passage_to_evaluate=self.query_passage_format.format(passage, query)

        if verbose:
            print("++++++++++++++++++++++++++")
            print(query_passage_to_evaluate)
            print("++++++++++++++++++++++++++")


        if number_of_completions > 1:
            temperature = 1
        else:
            temperature = 0

        messages_to_send = [self.initial_prompt]

        if self.examples is not None:
            for i, example in enumerate(self.examples):
                for example_role in example:
                    if example_role['role'] == "user":
                        messages_to_send.append({'role': "user", 
                                                 'content': self.example_format.format(i + 1, example_role['content'])})
                    else:
                        messages_to_send.append(example_role)

        messages_to_send.append({'role': "user", 'content': query_passage_to_evaluate})

    
        if verbose:
            print("\n")
            print(messages_to_send)
        
        response = openai.ChatCompletion.create(model=GPT4Access.MODEL_NAME,
                                                messages=messages_to_send,
                                                temperature=temperature,
                                                n=number_of_completions,
                                                max_tokens=GPT4Access.EVALUATION_MAX_TOKENS_RESPONSE)



        if verbose:
            print("\n")
            print(response['choices'])

        LLM_responses = []

        for i in range(number_of_completions):
            response_text = response['choices'][i]['message']['content']

            m = re.match(self.output_regex, response_text)

            if len(m.groups()) == 2:
                score = int(round(float(m.group(2))))
                reasoning = m.group(1)
            else:
                score = None
                reasoning = None

            LLM_responses.append({'score': score,
                                  'reasoning': reasoning})

        final_time = time.time()
        final_cost = self.compute_cost(response['usage'])

        if verbose:
            print("\nLLM query relevance evaluation duration: {}; cost: {}; usage{}\n\n".format(final_time - start_time, 
                                                                                                final_cost, 
                                                                                                response['usage']))

        if number_of_completions > 1:
            final_result = {'LLM_responses': LLM_responses,
                            'usage': response['usage'].copy(),
                            'cost': final_cost,
                            'duration': final_time - start_time}
        else:
            final_result = {'score': LLM_responses[0]['score'],
                            'reasoning': LLM_responses[0]['reasoning'],
                            'usage': response['usage'].copy(),
                            'cost': final_cost,
                            'duration': final_time - start_time}

        return final_result




if __name__ == '__main__':

    parser = ap.ArgumentParser()

    parser.add_argument('--model', default='gpt4')
    parser.add_argument('--api_keys', default=None, help="JSON file with LLM API keys storage.")
    parser.add_argument('--api_key_to_use', default=None, help="String identifying the API key to use from the API keys storage.")
    parser.add_argument('--initial_prompt', default=None, help="JSON file with the main prompt.")
    parser.add_argument('--query_passage_format', default=None, help="String format to combine query and passages to send to LLM")
    parser.add_argument('--examples', default=None, help="JSON file wiht the examples to be added in the prompt.")
    parser.add_argument('--example_format', default=None, help="String format for each example to be added in the prompt.")
    parser.add_argument('--result_regex', default=None, help="REGEX string to parse the LLM result.")
    parser.add_argument('--query_passages', default=None, help="Query/passages .tsv file.")
    parser.add_argument('--completions', default=1, type=int, help="Number of completions to request from LLM.")
    parser.add_argument('--output', default=None, help="Passage evaluations output file")
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--history', default='None', help="History file to reload execution parameters and evaluation cache")
    parser.add_argument('--config', default=None, help="Optional JSON file to hold execution configuration parameters.")

    args = parser.parse_args()


    # Try loading the history, if it already exist

    if (args.history is not None) and (os.path.exists(args.history)):
        with open(args.history, "rb") as input_file:
            history_data = pickle.load(input_file)

        evaluation_cache = history_data['evaluation_cache']
    else:
        evaluation_cache = None


    # Load the initial prompt and examples from the provided JSON files, if any.

    if args.initial_prompt is not None:
        args.initial_prompt = json.load(open(args.initial_prompt))

    if args.examples is not None:
        args.examples = json.load(open(args.examples))


    if args.config is not None:
        # Check which parameters have been provided through the single configuration file.

        config_from_file = {}

        with open(args.config) as input_file:
            for line in input_file:
                # print(line)

                m = re.match("\s*[\"\'](.+)[\"\']\s*:\s*[\"\'](.+)[\"\']\s*,?\s*[\n\r]", line)

                if (m is not None) and (len(m.groups()) == 2):
                    config_from_file[m.group(1)] = m.group(2)
                # else:
                #     print("no match")

        # print(config_from_file)

        for config_parameter in ['initial_prompt', 'query_passage_format', 'examples', 'example_format', 'result_regex', 'completions', 'output', 'verbose', 'api_key_to_use']:
            if (getattr(args, config_parameter) is None) and (config_parameter in config_from_file):
                setattr(args, config_parameter, config_from_file[config_parameter])


    if args.verbose:
        print("Provided args:\n\n")

        for key, item in vars(args).items():
            print("{}={}\n".format(key, item))


    # Check mandatory fields

    for config_parameter in ['initial_prompt', 'query_passage_format', 'result_regex', 'query_passages']:
        if (getattr(args, config_parameter) is None):
            raise ValueError("\"{}\" needs to be defined...".format(config_parameter))


    if args.model == 'gpt4':

        if args.api_keys is None:
            raise ValueError("Needs valid API key to access GPT4...".format(config_parameter))

        LLMInterface = GPT4Access(initial_prompt=args.initial_prompt,
                                  query_passage_format=args.query_passage_format,
                                  output_regex=args.result_regex, 
                                  examples=args.examples,
                                  example_format=args.example_format,
                                  evaluations_cache=evaluation_cache,
                                  verbose=args.verbose)
        
        LLMInterface.initialize_LLM(api_key=json.load(open(args.api_keys))[args.api_key_to_use])

        query_passages_df = pd.read_csv(args.query_passages, sep="\t")

        if args.verbose:
            print("query_passages_df.shape={}".format(query_passages_df.shape))

        validation_result_df = LLMInterface.query_passage_evaluation(query_passages_df=query_passages_df,
                                                                     output_file=args.output,
                                                                     number_of_completions=args.completions,
                                                                     verbose=args.verbose)
        


        if args.verbose:
            print("Final LLM usage cost: {}\nCost saved through already analyzed query/passage tuples: {}".format(validation_result_df['cost'].sum(),
                                                                                                                  validation_result_df['saved_cost'].sum()))



        # Save the history, if provided

        if (args.history is not None):
            with open(args.history, "wb") as output_file:
                pickle.dump({'args': args, 
                             'evaluation_cache': LLMInterface.get_evaluation_cache()}, output_file, pickle.HIGHEST_PROTOCOL)


        # If there is no output file, just send the result through the stout

        if args.output is None:
            validation_result_df.to_csv(sys.stdout, sep="\t", index=False)
