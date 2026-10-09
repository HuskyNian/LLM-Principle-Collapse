import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import re
import copy
from tqdm import tqdm
# Set your OpenAI API key
import random
import torch
from functools import partial
import os
from algorithm.base import ItemFactory,HistoryBuffer
from rdkit import Chem
import json
import time
from model.util import *
from algorithm import PromptTemplate
import importlib
import pickle
from model.LLM import LLM
from algorithm.web_search import WebSearchMemory


def format_log_value(value, precision=4, sci_threshold=1e-3, max_fixed=1e5):
    """Compact numeric formatting for console logs without changing saved values."""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not np.isfinite(value):
            return str(value)
        if value == 0:
            return "0"
        abs_value = abs(value)
        if abs_value < sci_threshold or abs_value >= max_fixed:
            return f"{value:.{precision}e}"
        text = f"{value:.{precision}f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(value)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MOO:
    def __init__(self, reward_system, llm,property_list,config,seed):
        self.reward_system = reward_system
        self.config = config
        self.seed = seed
        self.llm = llm
        #self.au_llm = LLM(model = self.config.get('model.name2'),config = config)
        self.history = HistoryBuffer()
        self.item_factory = ItemFactory(property_list)
        self.property_list = property_list
        self.pop_size = self.config.get('optimization.pop_size')
        self.budget = self.config.get('optimization.eval_budget')
        self.so_selection_ratio = self.config.get('so_selection_ratio', default=0.5)
        self.use_au = self.config.get('use_au')
        self.save_dir = os.path.join(self.config.get('save_dir'),self.config.get('model.name'))
        self.prompt_module = getattr(PromptTemplate ,self.config.get('model.prompt_module',default='Prompt'))
        self.history_moles = []
        self.mol_buffer = [] # same as all_mols but with orders for computing auc
        self.main_mol_buffer = []
        self.au_mol_buffer = []
        self.results_dict = {'results':[]}
        self.main_results_dict = {'results':[]}
        self.au_results_dict = {'results':[]}
        self.history_experience = []
        self.repeat_num = 0
        self.failed_num = 0
        self.generated_num = 0
        self.llm_calls = 0
        self.start = True
        self.patience = 0
        self.old_score = 0
        self.early_stopping = False
        self.record_dict = {}
        for i in ['main','au']:
            for j in ['all_num','failed_num','repeat_num']:
                self.record_dict[i+'_'+j] = 0
        self.record_dict['main_history_smiles'] = []
        self.record_dict['au_history_smiles'] = []
        self.time_step = 0
        self.start_time = time.time()
        self.num_offspring = self.config.get('num_offspring',default=2)
        self.exp_input_tokens = 0
        self.exp_output_tokens = 0
        self.web_search_memory = None
        self.web_search_records = []

    def budget_used(self):
        """How much of eval_budget is consumed.

        budget_metric:
          'valid'    (default; original behaviour) -- unique valid candidates,
                     i.e. len(mol_buffer).
          'attempts' -- every candidate generated and sent to the evaluator,
                     failures included. Failed attempts still cost evaluation
                     time and still carry signal about the model, so for
                     expensive evaluators this is the honest unit of budget.
        """
        if self.config.get('budget_metric', default='valid') == 'attempts':
            return self.generated_num
        return len(self.mol_buffer)

    def generate_initial_population(self, n):
        module_path = self.config.get('evalutor_path')  # e.g., "molecules"
        module = importlib.import_module(module_path)
        _generate = getattr(module, "generate_initial_population")
        strings = _generate(self.config,self.seed)
        if len(strings) == 0:
            print('generate init population return empty list, use llm to generate init pops')
            return []
        if isinstance(strings[0],str):
            items = [self.item_factory.create(i) for i in strings]
        else:
            self.store_history_moles(strings)
            items = strings
        for it in items:
            it.operator = 'seed'          # provided by the problem, not the LLM
            it.generation = 0
        return items
    def llm_init(self,generated):        
        prompt = self.prompt_generator.get_prompt('empty',None,None)
        log_gap = 10
        i=0
        while len(generated) <= self.pop_size:
            response = self.llm.chat(prompt)
            new_smiles = extract_smiles_from_string(response)
            new_smiles = [self.item_factory.create(smile) for smile in new_smiles]
            for it in new_smiles:
                it.operator = 'llm_init'   # unconditioned generation at t=0
                it.generation = 0
            # Under the attempts budget these candidates cost evaluations like
            # any other, so they must count. Gated on budget_metric so the
            # molecular runs' reported Uniqueness/Validity stay byte-identical.
            if self.config.get('budget_metric', default='valid') == 'attempts':
                self.generated_num += len(new_smiles)
            generated.extend(new_smiles)
            i+= 1
            if i%log_gap==0:
                print(f'{i} llm calls, init pop size [{len(generated)}/{self.pop_size}]')
        return generated
            
        
    def _error_feedback_text(self):
        # opt-in via config `error_feedback`; evaluators may expose
        # get_error_feedback() returning a short "recent failures" block.
        # Default off -> existing problems' prompts are byte-identical.
        if not self.config.get('error_feedback', default=False):
            return ''
        getter = getattr(self.reward_system, 'get_error_feedback', None)
        return getter() if getter else ''

    def mutation(self, parent_list):
        prompt = self.prompt_generator.get_prompt('mutation',parent_list,self.history_moles)
        prompt += self._error_feedback_text()
        #print('mutation prompt:',prompt)
        #assert False
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        return [self.item_factory.create(smile) for smile in new_smiles],prompt,response

    def crossover(self, parent_list):
        prompt = self.prompt_generator.get_prompt('crossover',parent_list,self.history_moles)
        prompt += self._error_feedback_text()
        #print('crossover prompt:',prompt)
        #assert False
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)

        return [self.item_factory.create(smile) for smile in new_smiles],prompt,response

    def mutation2(self, parent_list):
        prompt = self.prompt_generator.get_prompt('mutation',parent_list,self.history_moles)
        #print('mutation prompt:',prompt)
        #assert False
        response = self.au_llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        return [self.item_factory.create(smile) for smile in new_smiles],prompt,response

    def crossover2(self, parent_list):
        prompt = self.prompt_generator.get_prompt('crossover',parent_list,self.history_moles)
        #print('crossover prompt:',prompt)
        #assert False
        response = self.au_llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        
        return [self.item_factory.create(smile) for smile in new_smiles],prompt,response

    def evaluate(self,pops_origin):
        '''############
        pops = [i for i in pops_origin if i.total is None]
        already_evaluated = [i for i in pops_origin if i.total is not None]
        print('len pops: ',len(pops),'len already_evaluated: ',len(already_evaluated))
        if len(pops) == 0:
            return pops_origin
        pops, log_dict = self.reward_system.evaluate(pops,self.mol_buffer)
        self.failed_num += log_dict['invalid_num']
        self.repeat_num += log_dict['repeated_num']
        pops.extend(already_evaluated)
        pops = self.store_history_moles(pops)
        '''
        pops = pops_origin
        pops, log_dict = self.reward_system.evaluate(pops,self.mol_buffer,main_history=True)
        self.failed_num += log_dict['invalid_num']
        self.repeat_num += log_dict['repeated_num']
        pops = self.store_history_moles(pops)
        return pops


    
    def store_history_moles(self,pops):
        unique_pop = []
        for i in pops:
            if i.value not in self.history_moles:
                self.history_moles.append(i.value)
                self.mol_buffer.append([i, len(self.mol_buffer)+1])
                unique_pop.append(i)
            else:
                self.repeat_num += 1
        return unique_pop
        
    def explore(self, parent_list=None):
        """Structural-jump operator: propose candidates without conditioning on
        specific parents.

        Was previously a no-op stub (`def explore(self): pass`) with a signature
        that could not even be called by mating(), so explore_prob had to be 0 in
        every config and the only mechanism able to leave the current region was
        effectively disabled. The prompt path in PromptTemplate existed but was
        unreachable, and expected Item objects while MOO passed a list of raw
        strings (history_moles), so it would have raised on x.total.

        Here we pass real Items from mol_buffer. The exploration prompt shows a
        random sample of the elite pool plus a slice of the worst candidates and
        asks for novel proposals unlike either, so it is genuinely unconditioned
        on a chosen parent pair while still being informed by what has been tried.
        """
        evaluated = [entry[0] for entry in self.mol_buffer if entry[0].total is not None]
        if len(evaluated) < 20:
            # Too little history for the elite/worst split the prompt needs;
            # fall back to mutation this round rather than emitting a weak prompt.
            if not parent_list:
                return [], '', ''
            items, prompt, response = self.mutation(parent_list)
            for it in items:              # label honestly: this was a mutation
                it.operator = 'mutation'
            return items, prompt, response
        prompt = self.prompt_generator.get_prompt('explore', None, evaluated)
        prompt += self._error_feedback_text()
        response = self.llm.chat(prompt)
        new_smiles = extract_smiles_from_string(response)
        return [self.item_factory.create(smile) for smile in new_smiles], prompt, response

    def log_results(self, 
                mol_buffer: list = None, 
                buffer_type: str = "default", 
                finish: bool = False) -> None:
        """
        Logs performance metrics (AUCs, top scores, diversity, hypervolume, etc.) 
        for the given molecular buffer, and saves them to JSON.

        Parameters:
        - mol_buffer (list): The molecular buffer to evaluate, list of (Molecule, Info) tuples.
        - buffer_type (str): The name of the buffer type ('main', 'au', or 'default' for mol_buffer).
        - finish (bool): Whether this is the final evaluation.
        """
        if mol_buffer is None:
            mol_buffer = self.mol_buffer
        if not mol_buffer:
            # Zero valid candidates so far (possible on hard problems where even
            # the seed fails evaluation). Log a stub row instead of crashing on
            # top10[0] below; the run continues and may recover.
            self.results_dict['results'].append({
                'all_unique_moles': 0, 'llm_calls': self.llm_calls,
                'Uniqueness': 1.0, 'Validity': 0.0, 'Training_step': self.time_step,
                'avg_top1': 0, 'avg_top10': 0, 'avg_top100': 0,
                'top1_auc': 0, 'top10_auc': 0, 'top100_auc': 0,
                'hypervolume': 0, 'div': 0,
                'input_tokens': self.llm.input_tokens, 'output_tokens': self.llm.output_tokens,
                'input_tokens_exp': self.exp_input_tokens, 'output_tokens_exp': self.exp_output_tokens,
                'generated_num': self.generated_num,
                'running_time[s]': time.time() - self.start_time,
            })
            print(f'{buffer_type}: 0/{self.budget} generated: {self.generated_num} | no valid candidates yet')
            self.time_step += 1
            return
        auc1 = top_auc(mol_buffer, 1, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc10 = top_auc(mol_buffer, 10, finish=finish, freq_log=100, max_oracle_calls=self.budget)
        auc100 = top_auc(mol_buffer, 100, finish=finish, freq_log=100, max_oracle_calls=self.budget)

        top100 = sorted(mol_buffer, key=lambda item: item[0].total, reverse=True)[:100]
        top100_mols = [i[0] for i in top100]
        top10 = top100_mols[:10]

        avg_top10 = np.mean([i.total for i in top10])
        avg_top100 = np.mean([i.total for i in top100_mols])

        if self.config.get('cal_div',default=False):
            from tdc import Evaluator
            div_evaluator = Evaluator(name = 'Diversity')
            diversity_top100 = div_evaluator([i.value for i in top100_mols])
        else:
            diversity_top100 = 0
        
        ###
        # A run can legitimately have zero valid candidates so far (e.g. the
        # seed itself fails evaluation on a slow problem); logging must not
        # crash the whole run in that case.
        if top10 and top10[0].property and 'l_delta_b' in top10[0].property and 'aspect_ratio' in top10[0].property:
            all_mols = [item[0] for item in mol_buffer]
            all_mols = [i for i in all_mols if i.property['feasibility']<0.01]
            print('average top 10feasibility: ',np.mean([i.property['feasibility'] for i in top10]))
            if len(all_mols)>0:
                scores = np.array([[-i.property['l_delta_b'],i.property['aspect_ratio']] for i in all_mols])
                volume = cal_fusion_hv(scores)
            else:
                volume = 0
        else:
            scores = np.array([i.scores for i in top100_mols])
            volume = cal_hv(scores) ###

        if buffer_type == "default":
            uniqueness = 1 - self.repeat_num / (self.generated_num + 1e-6)
            validity = 1 - self.failed_num / (self.generated_num + 1e-6)
        else:
            uniqueness = 1 - self.record_dict[f'{buffer_type}_repeat_num'] / (self.record_dict[f'{buffer_type}_all_num'] + 1e-6)
            validity = 1 - self.record_dict[f'{buffer_type}_failed_num'] / (self.record_dict[f'{buffer_type}_all_num'] + 1e-6)

        # Handle early stopping for default logging only
        if buffer_type == "default":
            new_score = avg_top10
            if new_score - self.old_score == 0:
                self.patience += 1
                patience_limit = self.config.get('early_stopping_patience', default=50)
                if self.config.get('early_stopping', default=True) and self.patience >= patience_limit:
                    print(f'convergence criteria met: avg_top10 unchanged for {self.patience} generations, abort ...... ')
                    self.early_stopping = True
            else:
                self.patience = 0
            self.old_score = new_score

        # Select results_dict and path
        if buffer_type == "default":
            results_dict = self.results_dict
            save_dir = os.path.join(self.save_dir, "results")
        elif buffer_type == "main":
            results_dict = self.main_results_dict
            save_dir = os.path.join(self.save_dir, "results_main")
        elif buffer_type == "au":
            results_dict = self.au_results_dict
            save_dir = os.path.join(self.save_dir, "results_au")
        else:
            raise ValueError(f"Unknown buffer_type: {buffer_type}")

        json_path = os.path.join(save_dir, '_'.join(self.property_list) + '_' +
                                self.config.get('save_suffix') + f'_{self.seed}.json')
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        results_dict['results'].append({
            'all_unique_moles': len(self.history_moles),
            'llm_calls': self.llm_calls,
            'Uniqueness': uniqueness,
            'Validity': validity,
            'Training_step': self.time_step,
            'avg_top1': top10[0].total,
            'avg_top10': avg_top10,
            'avg_top100': avg_top100,
            'top1_auc': auc1,
            'top10_auc': auc10,
            'top100_auc': auc100,
            'hypervolume': volume,
            'div': diversity_top100,
            'input_tokens': self.llm.input_tokens,
            'output_tokens': self.llm.output_tokens,
            'input_tokens_exp': self.exp_input_tokens,
            'output_tokens_exp': self.exp_output_tokens,
            'generated_num': self.generated_num,
            'running_time[s]': time.time()-self.start_time
        })

        # Only include config and history in default log
        if buffer_type == "default":
            print('================================================================')
            results_dict['params'] = self.config.to_string()
            if len(self.history_experience) > 1:
                results_dict['history_experience'] = [self.history_experience[0], self.history_experience[-1]]
            if self.web_search_memory and self.web_search_memory.enabled:
                results_dict['web_search_records'] = self.web_search_memory.records
                results_dict['external_principles'] = self.web_search_memory.external_principles

        with open(json_path, 'w',encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False,indent=4)
        
        running_hours = (time.time() - self.start_time) / 3600
        print(f'{buffer_type}: {len(self.history_moles)}/{self.budget} generated: {self.generated_num} | '
            f'mol_buffer: {len(mol_buffer)} | '
            f'Uniqueness: {format_log_value(uniqueness)} | '
            f'Validity: {format_log_value(validity)} | '
            f'llm_calls: {self.llm_calls} | '
            f'Training step: {self.time_step} | '
            f'avg_top1: {format_log_value(top10[0].total)} | '
            f'avg_top10: {format_log_value(avg_top10)} | '
            f'avg_top100: {format_log_value(avg_top100)} | '
            f'top1_auc: {format_log_value(auc1)} | '
            f'top10_auc: {format_log_value(auc10)} | '
            f'top100_auc: {format_log_value(auc100)} | '
            f'hv: {format_log_value(volume)} | '
            f'unique top 100: {len(np.unique([i.value for i in top100_mols]))} | '
            f'input_tokens: {self.llm.input_tokens} | '
            f'output_tokens: {self.llm.output_tokens} | '
            f'input_tokens_exp: {self.exp_input_tokens} | '
            f'output_tokens_exp: {self.exp_output_tokens} | '
            f'running_time: {format_log_value(running_hours)} h | '
            f'div: {format_log_value(diversity_top100)}'
            )


    def update_experience(self):
        prompt,best_moles_prompt,bad_moles_prompt = self.prompt_generator.make_experience_prompt(self.mol_buffer)
        start_input_tokens = self.llm.input_tokens
        start_output_tokens = self.llm.output_tokens
        response = self.llm.chat(prompt)
        self.exp_input_tokens += self.llm.input_tokens - start_input_tokens
        self.exp_output_tokens += self.llm.output_tokens - start_output_tokens
        #self.prompt_generator.experience = (f"I already have some experience, take advantage of them :{response}"
        #                                    )
        self.prompt_generator.pure_experience = response
        # domain-neutral noun; defaults keep existing problems' prompts byte-identical
        candidate_noun = self.config.get('candidate_noun', default='molecules')
        self.prompt_generator.experience = (f"I already have some experience: <experience> {response} </experience>"
                                            f"You can take advantage of these experience and try to propose better {candidate_noun} according to the objectives.\n")
        #self.prompt_generator.experience = (f"I already have some experience and some good and bad moleculles, the experience is: <experience> {response}" 
        #                                    f"good example molecules are: {best_moles_prompt},\n"
        #                                    f"and bad example molecules that you need to avoid molecules like them: {bad_moles_prompt}. </experience>\n"
        #                                    f"You can take advantage of them and try to propose better molecules according to the objectives.\n"
        #                                    )
        
        self.history_experience.append(self.prompt_generator.experience) 
        print('length exp:',len(self.prompt_generator.experience))

    def setup_web_search(self):
        try:
            self.web_search_memory = WebSearchMemory(
                config=self.config,
                llm=self.llm,
                seed=self.seed,
                property_list=self.property_list,
            )
        except Exception as e:   # 附属功能，任何初始化失败都不应让 run 崩掉
            print(f"web_search setup failed ({e}); continuing without web search")
            self.web_search_memory = None
            return
        if self.web_search_memory.enabled:
            print('web_search enabled for molecule optimization')

    def maybe_update_web_search(self):
        if not self.web_search_memory or not self.web_search_memory.enabled:
            return
        start_input_tokens = self.llm.input_tokens
        start_output_tokens = self.llm.output_tokens
        record = self.web_search_memory.maybe_update(
            mol_buffer=self.mol_buffer,
            budget_used=self.budget_used(),
            budget=self.budget,
        )
        self.exp_input_tokens += self.llm.input_tokens - start_input_tokens
        self.exp_output_tokens += self.llm.output_tokens - start_output_tokens
        if record:
            self.llm_calls += int(record.get('llm_calls_used', 0))
            self.web_search_records = self.web_search_memory.records
            self.prompt_generator.external_principles = self.web_search_memory.external_principles
            print('web_search external principles length:', len(self.prompt_generator.external_principles or ''))

    def checkpoint_data(self, population, init_pops, start_time):
        data = {
                'history':self.history,
                'init_pops':init_pops,
                'final_pops':population,
                'all_mols':self.mol_buffer,
                'properties':self.property_list,
                'evaluation': self.results_dict['results'],
                'running_time':f'{(time.time()-start_time)/3600:.2f} hours',
                'experience': self.history_experience,
            }
        if self.web_search_memory and self.web_search_memory.enabled:
            data['web_search_records'] = self.web_search_memory.records
            data['external_principles'] = self.web_search_memory.external_principles
        return data

    
    def run(self):
        store_path = os.path.join(self.save_dir,'mols','_'.join(self.property_list) + '_' + self.config.get('save_suffix') + f'_{self.seed}' +'.pkl')
        self.prompt_generator = self.prompt_module(self.config)
        self.setup_web_search()
        
        if not os.path.exists(os.path.dirname(store_path)):
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
            
        ''' initialize genetic gfn model'''
        
        if self.use_au:
            from mcce.genetic_gfn.multi_objective.genetic_gfn.run import Genetic_GFN_Optimizer
            from genetic_gfn.multi_objective.run import prepare_optimization_inputs
            args, config_default, oracle = prepare_optimization_inputs()
            self.au_model = Genetic_GFN_Optimizer(args=args)
            self.au_model.setup_model(oracle, config_default)

        """High level logic"""
        print('exper_name',self.config.get('exper_name'))
        set_seed(self.seed)
        start_time = time.time()
        
        #initialization 
        if self.config.get('inject_per_generation'):
            module_path = self.config.get('evalutor_path')  # e.g., "molecules"
            module = importlib.import_module(module_path)
            _get = getattr(module, "get_database")
            database = _get(self.config,n_sample=200)
        if self.config.get('resume') or self.branch_enabled():
            population,init_pops = self.load_ckpt(store_path)
            if self.delay_hint_enabled():
                self.log_results()
        else:
            population = self.generate_initial_population(n=self.pop_size)
            if len(population)<self.pop_size:
                population = self.llm_init(population)
            population = self.evaluate(population) # including removing invalid and repeated candidates
            self.log_results()
            self.maybe_update_web_search()
            init_pops = copy.deepcopy(population)
        data = self.checkpoint_data(population, init_pops, start_time)
        with open(store_path, 'wb') as f:
            pickle.dump(data, f)    
        
        self.num_gen = 0
        
        while True:
            if self.config.get('inject_per_generation'):
                print('inject!')
                population.extend(random.sample(database,self.config.get('inject_per_generation')))
            offspring_times = max(min(self.pop_size //self.num_offspring, (self.budget - self.budget_used()) //self.num_offspring),1)
            offspring = self.generate_offspring(population, offspring_times)
            population = self.select_next_population(self.pop_size)
            self.log_results()
            self.maybe_update_web_search()
            # experience_start: how many evaluated candidates before summaries begin.
            # Default 100 keeps existing problems unchanged; kernel budgets (~200)
            # need a much lower threshold or summaries barely exist.
            if self.config.get('model.experience_prob')>0 and len(self.mol_buffer)>self.config.get('experience_start', default=100):
                self.update_experience()
            if self.budget_used() >= self.budget or self.early_stopping:
                self.log_results(finish=True)
                if self.use_au or self.au_llm:
                    self.log_results(self.main_mol_buffer,buffer_type="main", finish=True)
                    self.log_results(self.au_mol_buffer,buffer_type="au", finish=True)
                # Save before breaking: the per-generation save below is only
                # reached on non-final generations, so without this the last
                # generation's candidates exist in mol_buffer but never land in
                # the checkpoint (observed: pilot pkl had 14 of 16 candidates).
                final_data = self.checkpoint_data(population, [obj[0] for obj in self.mol_buffer[:self.pop_size]], start_time)
                with open(store_path, 'wb') as f:
                    pickle.dump(final_data, f)
                break
            self.num_gen+=1
            data = self.checkpoint_data(population, [obj[0] for obj in self.mol_buffer[:self.pop_size]], start_time)
            with open(store_path, 'wb') as f:
                pickle.dump(data, f)
            if self.num_gen%10==0:
                print(f"Data saved to {store_path}")
        print(f'=======> total running time { (time.time()-start_time)/3600 :.2f} hours <=======')
        
        return init_pops,population  # 计算效率

    def record(self, tmp_offspring: list, buffer_type: str) -> list:
        """
        Records valid and unique molecules from the given temporary offspring list.

        Parameters:
        - tmp_offspring (list): List of generated candidate molecules (Item objects).
        - buffer_type (str): Type identifier for tracking stats (e.g., 'main', 'au').

        Returns:
        - offspring (list): Filtered list of valid and unique Item objects.
        """
        offspring = []
        smiles_this_gen = []
        self.record_dict[buffer_type + '_all_num'] += len(tmp_offspring)

        for child in tmp_offspring:
            mol = Chem.MolFromSmiles(child.value)
            if mol is None:
                self.record_dict[buffer_type + '_failed_num'] += 1
            else:
                child.value = Chem.MolToSmiles(mol)
                if (child.value in self.record_dict[buffer_type + '_history_smiles'] or
                    child.value in smiles_this_gen):
                    self.record_dict[buffer_type + '_repeat_num'] += 1
                else:
                    self.record_dict[buffer_type + '_history_smiles'].append(child.value)
                    smiles_this_gen.append(child.value)
                    offspring.append(child)
        return offspring

    def mating(self, parent_list: list, au: bool = False) -> tuple:
        """
        Applies genetic operators (crossover, mutation, or exploration) to parent list.

        Parameters:
        - parent_list (list): A list of parent Item objects.
        - au (bool): Whether this is for auxiliary model (default: False).

        Returns:
        - tuple: (list of offspring Items, str prompt, str response)
        """
        crossover_prob = self.config.get('model.crossover_prob')
        mutation_prob = self.config.get('model.mutation_prob')
        explore_prob = self.config.get('model.explore_prob')
        function = np.random.choice([self.crossover,self.mutation,self.explore],p=[crossover_prob,
                                                                                   mutation_prob,
                                                                                   explore_prob])
        # Provenance: which operator produced a candidate, and from which
        # parents, is needed to analyse where principle switches come from
        # (e.g. did explore produce the escapes?). Recorded on the offspring
        # below; without it the trajectory cannot answer that question.
        _op_name = getattr(function, '__name__', str(function))
        items,prompt,response = function(parent_list)
        parent_values = [getattr(p, 'value', None) for p in (parent_list or [])]
        for _it in items:
            # explore() may fall back to mutation on a short history and tags
            # itself; don't overwrite a label the operator already set.
            if not getattr(_it, 'operator', None):
                _it.operator = _op_name
            _it.parent_values = parent_values
            _it.generation = getattr(self, 'num_gen', 0)
        return items,prompt,response

    def mating2(self, parent_list: list, au: bool = False) -> tuple:
        crossover_prob = self.config.get('model.crossover_prob')
        mutation_prob = self.config.get('model.mutation_prob')
        explore_prob = self.config.get('model.explore_prob')
        function = np.random.choice([self.crossover2,self.mutation2,self.explore],p=[crossover_prob,
                                                                                   mutation_prob,
                                                                                   explore_prob])
        items,prompt,response = function(parent_list)
        return items,prompt,response
    
    def generate_offspring(self, population: list, offspring_times: int) -> list:
        """
        Generates new offspring from the population using LLM-driven operations, and evaluates them.

        Parameters:
        - population (list): List of Item objects representing the current population.
        - offspring_times (int): Number of offspring pairs to generate.

        Returns:
        - list: Evaluated and recorded offspring.
        """
        parents = [random.sample(population, min(self.num_offspring, len(population))) for i in range(offspring_times)]  # degenerate small populations must not crash
        # 一代里全部 LLM 调用都超时（网关抽风）时 results 为空，下面的 zip 不会执行，
        # children 就没定义 → UnboundLocalError 直接把整条 run 打死（2026-09-09 012×Qwen L1start）。
        # 预先置空：这一代产 0 个子代，循环照常进入下一代。
        children, prompts, responses = [], [], []
        parallel = True
        if parallel:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.mating, parent_list=parent_list) for parent_list in parents]
                #children, prompts, responses = zip(*results) #[[item,item],[item,item]] # ['who are you value 1', 'who are you value 2'] # ['yes, 'no']   
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=180)  # 最多等待 120 秒
                        results.append(result)
                    except concurrent.futures.TimeoutError:
                        print("Warning: A task timed out after 180 seconds.")
                        continue  # 或者 results.append(default_value)
                if results:
                    children, prompts, responses = zip(*results)
                    self.llm_calls += len(results)
                else:
                    print("No results collected due to timeouts.")
        else:
            children,prompts,responses = [],[],[]
            for parent_list in tqdm(parents):
                child,prompt,response = self.mating(parent_list)
                children.append(child)
                prompts.append(prompt)
                responses.append(response)
                self.llm_calls += 1
        tmp_offspring = []
        if self.start:
            self.start=False
        for child_pair in children:
            self.generated_num += len(child_pair)
            tmp_offspring.extend(child_pair)
        
        if self.use_au:
            self.record(tmp_offspring,'main')
            self.save_log_mols(tmp_offspring,buffer_type='main')
            
            au_smiles = self.au_model.sample_n_smiles(32,self.mol_buffer)
            ### au_smiles = self.generate_offspring_au(population,offspring_times=20)
            au_smiles = [self.item_factory.create(smiles) for smiles in au_smiles]
            self.generated_num += len(au_smiles)
            self.record(au_smiles,'au')
            self.save_log_mols(au_smiles,buffer_type='au')
            tmp_offspring.extend(au_smiles)

        if self.au_llm:
            self.record(tmp_offspring,'main')
            self.save_log_mols(tmp_offspring,buffer_type='main')

            parents = [random.sample(population, min(self.num_offspring, len(population))) for i in range(offspring_times)]  # degenerate small populations must not crash
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.mating2, parent_list=parent_list) for parent_list in parents]
                #children, prompts, responses = zip(*results) #[[item,item],[item,item]] # ['who are you value 1', 'who are you value 2'] # ['yes, 'no']   
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=180)  # 最多等待 120 秒
                        results.append(result)
                    except concurrent.futures.TimeoutError:
                        print("Warning: A task timed out after 180 seconds.")
                        continue  # 或者 results.append(default_value)
                if results:
                    au_children, au_prompts, au_responses = zip(*results)
                    self.llm_calls += len(results)
                else:
                    print("No results collected due to timeouts.")
            au_smiles = []
            for au_child_pair in au_children:
                au_smiles.extend(au_child_pair)
            self.generated_num += len(au_smiles)
            self.record(au_smiles,'au')
            self.save_log_mols(au_smiles,buffer_type='au')
            
            tmp_offspring.extend(au_smiles)
            
        #offspring = self.sanitize(tmp_offspring,record=True)
        offspring = tmp_offspring
        
        if len(offspring) == 0:
            return []

        offspring = self.evaluate(offspring)
        self.history.push(prompts,children,responses) 
        return offspring

    def save_log_mols(self, mols: list, buffer_type: str) -> None:
        """
        Evaluates molecules, trains AU model if applicable, stores in buffer, and logs metrics.

        Parameters:
        - mols (list): List of Item objects to save and evaluate.
        - buffer_type (str): Indicates which buffer ('main' or 'au') to update.
        """
        self.time_step += 1
        mols,_ = self.reward_system.evaluate(mols,self.mol_buffer)
        if buffer_type=='main':
            mol_buffer = self.main_mol_buffer
            # #######self.au_model.train_on_smiles([i.value for i in mols],[i.total for i in mols],loop=4,time_step=self.time_step,mol_buffer=mol_buffer)
        elif buffer_type=='au':
            mol_buffer = self.au_mol_buffer
            # #######self.au_model.train_on_smiles([i.value for i in mols],[i.total for i in mols],loop=4,time_step=self.time_step,mol_buffer=mol_buffer)
        # print('oracle length: ',len(self.mol_buffer))

        self.mol_buffer_store(mol_buffer,mols)
        # 这里加了重复的 
        self.log_results(mol_buffer,buffer_type, finish=False)

    def mol_buffer_store(self, mol_buffer: list, mols: list) -> list:
        """
        Stores non-duplicate molecules into a specified buffer.

        Parameters:
        - mol_buffer (list): List buffer to store Item objects.
        - mols (list): List of Item objects to attempt storing.

        Returns:
        - list: Updated mol_buffer with new unique molecules.
        """
        all_smiles = [i[0].value for i in mol_buffer]
        for child in mols:
            mol = Chem.MolFromSmiles(child.value) 
            if mol is None: # check if valid
                pass
            else:
                child.value = Chem.MolToSmiles(mol,canonical=True)
                # check if repeated
                if child.value in all_smiles:
                    pass
                else:
                    all_smiles.append(child.value)
                    mol_buffer.append([child,len(mol_buffer)+1])
        return mol_buffer

    def select_next_population(self,pop_size):
        whole_population = [i[0] for i in self.mol_buffer]
        if self.config.get('selector') == 'nsga2':
            return nsga2_selection(whole_population, pop_size)
        elif self.config.get('selector') == 'so':
            return so_selection(whole_population, pop_size)
        else:
            if len(self.property_list)>1:
                return nsga2_so_selection(whole_population, pop_size, self.so_selection_ratio)
            else:
                return so_selection(whole_population,pop_size)

    def generate_offspring_au(self, population, offspring_times=20):
        parents = [random.sample(population, min(2, len(population))) for i in range(offspring_times)]  # degenerate small populations must not crash
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.mating, parent_list=parent_list,au=True) for parent_list in parents]
            results = [future.result() for future in futures]
            children, prompts, responses = zip(*results) #[[item,item],[item,item]] # ['who are you value 1', 'who are you value 2'] # ['yes, 'no']
            self.llm_calls += len(results)
        tmp_offspring = []
        for child_pair in children:
            self.generated_num += len(child_pair)
            tmp_offspring.extend(child_pair)
        return tmp_offspring

    def delay_hint_enabled(self):
        return bool(self.config.get('use_hint', default=False)) and bool(self.config.get('delay_hint', default=False))

    # --- generic trajectory branching -------------------------------------
    # Replaying a finished run from an arbitrary point is needed for any
    # mid-trajectory intervention, not just the delayed principle hint:
    # entropy-triggered hints, operator swaps, backbone swaps, or simply
    # continuing a run that hit its budget. `branch_from_suffix` + `branch_at`
    # express that directly; delay_hint remains a special case of it.
    def branch_enabled(self):
        return bool(self.config.get('branch_from_suffix', default=None)) or self.delay_hint_enabled()

    def branch_source_suffix(self):
        return (self.config.get('branch_from_suffix', default=None)
                or self.config.get('delay_hint_source_suffix', default='test'))

    def branch_cutoff(self):
        """Branch point, in units of *valid* candidates (mol_buffer order).

        `branch_at: null` (and no delay-hint fraction) means "take the whole
        source trajectory", i.e. continue where it stopped.
        """
        explicit = self.config.get('branch_at', default=None)
        if explicit is not None:
            return int(explicit)
        if self.delay_hint_enabled():
            return self.delay_hint_cutoff()
        return 10 ** 9  # effectively no truncation

    def artifact_paths_for_suffix(self, suffix):
        stem = '_'.join(self.property_list) + '_' + suffix + f'_{self.seed}'
        pkl_path = os.path.join(self.save_dir, 'mols', stem + '.pkl')
        json_path = os.path.join(self.save_dir, 'results', stem + '.json')
        return pkl_path, json_path

    def delay_hint_cutoff(self):
        explicit_cutoff = self.config.get('delay_hint_budget', default=None)
        if explicit_cutoff is not None:
            return int(explicit_cutoff)
        fraction = float(self.config.get('delay_hint_fraction', default=0.4))
        return int(round(self.budget * fraction))

    def truncate_buffer_at_cutoff(self, mol_buffer, cutoff):
        truncated = []
        for idx, record in enumerate(mol_buffer, start=1):
            order = record[1] if isinstance(record, (list, tuple)) and len(record) > 1 else idx
            if int(order) <= cutoff:
                truncated.append(record)
        if not truncated:
            truncated = mol_buffer[:cutoff]
        return truncated

    def closest_result_to_cutoff(self, results, cutoff):
        if not results:
            return 0, {}
        return min(
            enumerate(results),
            key=lambda item: abs(int(item[1].get('all_unique_moles', 0)) - cutoff)
        )

    def extract_pure_experience(self, experience):
        match = re.search(r"<experience>(.*?)</experience>", str(experience), flags=re.S)
        return match.group(1).strip() if match else str(experience)

    def restore_experience_from_ckpt(self, ckpt, experience_count=None):
        experiences = ckpt.get('experience', []) or []
        if experience_count is not None:
            experiences = experiences[:experience_count]
        self.history_experience = experiences
        if self.history_experience:
            last_experience = self.history_experience[-1]
            self.prompt_generator.experience = last_experience
            self.prompt_generator.pure_experience = self.extract_pure_experience(last_experience)
	    
    def load_ckpt(self,store_path):
        print('resume training')
        delay_hint = self.branch_enabled() and not self.config.get('resume')
        if delay_hint :
            source_suffix = self.branch_source_suffix()  # delay_hint_source_suffix 是其默认值，冉念的用法行为不变
            source_store_path, json_path = self.artifact_paths_for_suffix(source_suffix)
            cutoff = self.branch_cutoff()
            print(f'branch resume: source_suffix={source_suffix}, cutoff={cutoff}')
        else:
            source_store_path = store_path
            save_dir = os.path.join(self.save_dir, "results")
            json_path = os.path.join(save_dir, '_'.join(self.property_list) + '_' +
                                self.config.get('save_suffix') + f'_{self.seed}.json')
        with open(source_store_path,'rb') as f:
            ckpt = pickle.load(f)
        with open(json_path,'rb') as f:
            result_ckpt = json.load(f)
        if delay_hint:
            self.mol_buffer = self.truncate_buffer_at_cutoff(ckpt['all_mols'], cutoff)
        else:
            self.mol_buffer = ckpt['all_mols']
        population = self.select_next_population(self.pop_size) 
        init_pops = [i[0] for i in self.mol_buffer[:self.pop_size]]#ckpt['init_pops']
        self.history = ckpt['history']
        
        self.history_moles = [i[0].value for i in self.mol_buffer]
        if self.web_search_memory and self.web_search_memory.enabled:
            restored_web_records = [
                record for record in ckpt.get('web_search_records', [])
                if int(record.get('budget_used', 0)) <= len(self.mol_buffer)
            ]
            restored_external_principles = (
                restored_web_records[-1].get('external_principles', '')
                if restored_web_records else ''
            )
            self.web_search_memory.restore(
                restored_web_records,
                restored_external_principles,
            )
            self.web_search_records = self.web_search_memory.records
            self.prompt_generator.external_principles = self.web_search_memory.external_principles
        if delay_hint:
            start_result_idx, start_result = self.closest_result_to_cutoff(result_ckpt['results'], len(self.mol_buffer))
            self.results_dict['results'] = []
            if self.config.get('restore_experience', default=False):
                self.restore_experience_from_ckpt(ckpt, experience_count=start_result_idx)
        else:
            start_result_idx = len(result_ckpt['results']) - 1
            start_result = result_ckpt['results'][-1]
            self.results_dict['results'] = ckpt['evaluation']
            if self.config.get('restore_experience', default=False):
                self.restore_experience_from_ckpt(ckpt)
        self.generated_num = start_result.get('generated_num', 0)
        self.llm_calls = start_result.get('llm_calls', 0)
        self.repeat_num = int((1-start_result.get('Uniqueness', 1)) * self.generated_num)
        self.failed_num = int((1-start_result.get('Validity', 1)) * self.generated_num)
        return population, init_pops

    
    
 
