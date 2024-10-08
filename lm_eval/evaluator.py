import itertools
import json
import logging
import random
import time
import os
from datetime import date
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Union
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import faiss
import faiss.contrib.torch_utils

import lm_eval.api.metrics
import lm_eval.api.registry
import lm_eval.api.task
import lm_eval.models
from lm_eval.caching.cache import delete_cache
from lm_eval.evaluator_utils import (
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
    print_writeout,
    run_task_tests,
)
from lm_eval.loggers import EvaluationTracker
from lm_eval.loggers.utils import add_env_info, add_tokenizer_info, get_git_commit_hash
from lm_eval.tasks import (
    TaskManager,
    get_task_dict,
)
from lm_eval.utils import (
    eval_logger,
    handle_non_serializable,
    hash_string,
    positional_deprecated,
    simple_parse_args_string,
)

from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from lm_eval.api.model import LM
    from lm_eval.api.task import Task

from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# Ensure you have the necessary NLTK data
nltk.download('punkt')
def generate_ngrams(text, n):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    return list(ngrams(tokens, n))

def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

@positional_deprecated
def simple_evaluate(
    model,
    model_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    evaluation_tracker: Optional[EvaluationTracker] = None,
    system_instruction: Optional[str] = None,
    apply_chat_template: Union[bool, str] = False,
    fewshot_as_multiturn: bool = False,
    gen_kwargs: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param cache_requests: bool, optional
        Speed up evaluation by caching the building of dataset requests. `None` if not caching.
    :param rewrite_requests_cache: bool, optional
        Rewrites all of the request cache if set to `True`. `None` if not desired.
    :param delete_requests_cache: bool, optional
        Deletes all of the request cache if set to `True`. `None` if not desired.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderrs. set to 0 for no stderr calculations to be performed.
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: Union[bool, str]
        Specifies whether to apply a chat template to the prompt.
        - If set to True, the default chat template is applied.
        - If set to a string, applies the specified chat template by name.
        Defaults to False (no chat template applied).
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.
    :param fewshot_random_seed: int
        Random seed for fewshot sampler random generator. If set to None, the seed of generator will be set to None.

    :return
        Dictionary of results
    """
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))
    start_date = time.time()
    torch.set_default_device(device)

    if delete_requests_cache:
        eval_logger.info("Deleting requests cache...")
        delete_cache()

    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    if tasks is None:
        tasks = []
    if len(tasks) == 0:
        raise ValueError(
            "No tasks specified, or no tasks found. Please verify the task names."
        )

    if gen_kwargs is not None:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            "generation_kwargs specified through cli, these settings will update set parameters in yaml tasks. "
            "Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if gen_kwargs == "":
            gen_kwargs = None

    if isinstance(model, str):
        if model_args is None:
            eval_logger.warning("model_args not specified. Using defaults.")
            model_args = ""

        if isinstance(model_args, dict):
            eval_logger.info(
                f"Initializing {model} model, with arguments: {model_args}"
            )
            lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )

        else:
            eval_logger.info(
                f"Initializing {model} model, with arguments: {simple_parse_args_string(model_args)}"
            )
            lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )
    else:
        if not isinstance(model, lm_eval.api.model.LM):
            raise TypeError(
                f"The value of `model` passed to simple_evaluate() was of type {type(model)}, but is required to be a subclass of lm_eval.api.model.LM . This may be because you are passing an initialized Hugging Face PreTrainedModel without having wrapped it in `lm_eval.models.huggingface.HFLM(pretrained=my_model)` first."
            )
        eval_logger.info("Using pre-initialized model")
        lm = model

    if use_cache is not None:
        eval_logger.info(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank"
            + str(lm.rank)
            + ".db",
        )

    if task_manager is None:
        task_manager = TaskManager(verbosity)

    task_dict = get_task_dict(tasks, task_manager)

    # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                if task_obj.get_config("output_type") == "generate_until":
                    if gen_kwargs is not None:
                        task_obj.set_config(
                            key="generation_kwargs", value=gen_kwargs, update=True
                        )

                if predict_only:
                    eval_logger.info(
                        f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                    )
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        eval_logger.info(
                            f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                        )
                    else:
                        eval_logger.warning(
                            f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                        )
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (
                        default_num_fewshot := task_obj.get_config("num_fewshot")
                    ) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                eval_logger.info(
                    f"Setting fewshot random generator seed to {fewshot_random_seed}"
                )

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    if check_integrity:
        run_task_tests(task_list=tasks)

    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=model,
            model_args=model_args,
            system_instruction=system_instruction,
            chat_template=lm.chat_template(apply_chat_template),
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        bootstrap_iters=bootstrap_iters,
        write_out=write_out,
        log_samples=True if predict_only else log_samples,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        verbosity=verbosity,
    )

    if lm.rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
        }
        # add more detailed model info if available
        if isinstance(lm, lm_eval.models.huggingface.HFLM):
            results["config"].update(lm.get_model_info())
        # add info about execution
        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (
                    list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []
                ),
                "device": device,
                "use_cache": use_cache,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
                "fewshot_seed": fewshot_random_seed,
            }
        )
        results["git_hash"] = get_git_commit_hash()
        results["date"] = start_date
        add_env_info(results)  # additional environment info to results
        add_tokenizer_info(results, lm)  # additional info about tokenizer
        return results
    else:
        return None


@positional_deprecated
def evaluate(
    lm: "LM",
    task_dict,
    limit: Optional[int] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: Optional[int] = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    system_instruction: Optional[str] = None,
    apply_chat_template: Union[bool, str] = False,
    fewshot_as_multiturn: bool = False,
    verbosity: str = "INFO",
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderr. Set to 0 for skipping all stderr calculations.
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: Union[bool, str]
        Specifies whether to apply a chat template to the prompt.
        - If set to True, the default chat template is applied.
        - If set to a string, applies the specified chat template by name.
        Defaults to False (no chat template applied).
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :return
        Dictionary of results
    """

    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    # tracks all Instances/requests a model must generate output on.
    requests = defaultdict(list)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = defaultdict(int)

    # get lists of group hierarchy and each type of request
    eval_tasks = get_task_list(task_dict)
    if not log_samples:
        if not all(
            "bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys()
            for task_output in eval_tasks
        ):
            raise ValueError("log_samples must be True for 'bypass' metric-only tasks")
    for task_output in eval_tasks:
        task: Task = task_output.task
        limit = get_sample_size(task, limit)
        task.build_all_requests(
            limit=limit,
            rank=lm.rank,
            world_size=lm.world_size,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=bool(apply_chat_template),
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=getattr(lm, "apply_chat_template")
            if apply_chat_template
            else None,
            tokenizer_name=getattr(lm, "tokenizer_name", "")
            if apply_chat_template
            else "",
        )
        eval_logger.debug(
            f"Task: {task_output.task_name}; number of requests on this rank: {len(task.instances)}"
        )
        if write_out:
            print_writeout(task)
        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if lm.world_size > 1:
            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = (
                lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            )
            # "multiple_choice" task types dispatch (several) "loglikelihood" request types
            reqtype = (
                "loglikelihood"
                if task.OUTPUT_TYPE == "multiple_choice"
                else task.OUTPUT_TYPE
            )
            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            # todo: may not account for padding in cases like SquadV2 which has multiple req types
            padding_requests[reqtype] += numpad

    ### Run LM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        eval_logger.info(f"Running {reqtype} requests")
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)

        # run requests through model
        resps = getattr(lm, reqtype)(cloned_reqs)

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()

    RANK = lm.rank
    WORLD_SIZE = lm.world_size
    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_output in eval_tasks:
        start_time = time.time()
        task = task_output.task
        to_save = []
        task.apply_filters()

        ### Collect values of metrics on all datapoints ###
        # # unpack results and sort back in order and return control to Task
        # TODO: make it possible to use a different metric per filter
        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps.keys():
            doc_iterator = task.doc_iterator(
                rank=RANK, limit=limit, world_size=WORLD_SIZE
            )
            
            i = 0
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                metrics = task.process_results(
                    doc, [req.filtered_resps[filter_key] for req in requests]
                )
                doc["answers"] = [req.filtered_resps[filter_key][:-1] for req in requests]
            
                if (task_output.task.task_name == "piqa") or (task_output.task.task_name == "hellaswag") or (task_output.task.task_name == "boolq"):
                    doc["dist_of_correct"] = [req.filtered_resps[filter_key] for req in requests][int(doc["label"])][-1]
                elif (task_output.task.task_name == "social_iqa") or (task_output.task.task_name == "copa"):
                    doc["dist_of_correct"] = [req.filtered_resps[filter_key] for req in requests][int(doc["label"])-1][-1]
                elif (task_output.task.task_name == "pubmedqa"):
                    doc["dist_of_correct"] = [req.filtered_resps[filter_key] for req in requests][["yes", "no", "maybe"].index(doc["final_decision"])][-1]
                elif (task_output.task.task_name == "lambada_openai"):
                    doc["dist_of_correct"] = [req.filtered_resps[filter_key] for req in requests][-1][-1]
                elif (task_output.task.task_name == "truthfulqa_mc1"):
                    doc["dist_of_correct"] = [req.filtered_resps[filter_key] for req in requests][doc["mc1_targets"]["labels"].index(1)][-1]
                elif ("mmlu" in task_output.task.task_name):
                    doc["dist_of_correct"] = [req.filtered_resps[filter_key] for req in requests][doc["answer"]][-1]
                elif (task_output.task.task_name == "mathqa"):
                    doc["dist_of_correct"] = [req.filtered_resps[filter_key] for req in requests][["a", "b", "c", "d", "e"].index(doc["correct"])][-1]
                elif (task_output.task.task_name == "logiqa"):
                    doc["dist_of_correct"] = [req.filtered_resps[filter_key] for req in requests][["a", "b", "c", "d"].index(doc["label"])][-1]
                elif (task_output.task.task_name == "arc_challenge") or (task_output.task.task_name == "arc_easy") or (task_output.task.task_name == "arc") or (task_output.task.task_name == "openbookqa") or (task_output.task.task_name == "commonsense_qa"):
                    doc["correct_answer"] = doc["choices"]["text"][doc["choices"]["label"].index(doc["answerKey"])]
                    doc["dist_of_correct"] = [req.filtered_resps[filter_key] for req in requests][doc["choices"]["label"].index(doc["answerKey"])][-1]
                elif (task_output.task.task_name == "winogrande"):
                    if int(doc["answer"]) == 2:
                        doc["correct_answer"] = doc["option2"]
                    elif int(doc["answer"]) == 1:
                        doc["correct_answer"] = doc["option1"]
                    doc["question"] = doc["sentence"]
                    doc["dist_of_correct"] = [req.filtered_resps[filter_key] for req in requests][int(doc["answer"])-1][-1]

                doc["dist_of_correct"] = torch.tensor(doc["dist_of_correct"]).to(lm.device)
                to_save.append(doc)
                
                if log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [
                            req.filtered_resps[filter_key] for req in requests
                        ],
                        "doc_hash": hash_string(
                            json.dumps(
                                requests[0].doc,
                                indent=2,
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                        ),
                        "prompt_hash": hash_string(requests[0].arguments[0]),
                        "target_hash": hash_string(str(target)),
                    }
                    example.update(metrics)
                    task_output.logged_samples.append(example)
            
                for metric, value in metrics.items():
                    to_save[i][metric] = value
                    task_output.sample_metrics[(metric, filter_key)].append(value)
                i += 1

        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        
        questions = []
        for k in range(len(to_save)):
            if task_output.task.task_name == "piqa":
                question_k = to_save[k]["goal"]
            elif (task_output.task.task_name == "logiqa") or ("mmlu" in task_output.task.task_name) or (task_output.task.task_name == "truthfulqa_mc1") or (task_output.task.task_name == "boolq") or (task_output.task.task_name == "winogrande") or (task_output.task.task_name == "arc_easy") or (task_output.task.task_name == "arc_challenge") or (task_output.task.task_name == "social_iqa") or (task_output.task.task_name == "commonsense_qa"):
                question_k = to_save[k]["question"]
            elif (task_output.task.task_name == "openbookqa"):
                question_k = to_save[k]["question_stem"]
            elif (task_output.task.task_name == "hellaswag"):
                question_k = to_save[k]["ctx"]
            elif (task_output.task.task_name == "mathqa"):
                question_k = to_save[k]["Problem"]
            elif (task_output.task.task_name == "pubmedqa"):
                question_k = to_save[k]["QUESTION"]
            elif (task_output.task.task_name == "copa"):
                question_k = to_save[k]["premise"]
            elif (task_output.task.task_name == "lambada_openai"):
                question_k = to_save[k]["text"]
            questions.append(question_k)
        
        question_embeddings = st_model.encode(questions)
        print("ENCODING QUESTIONS", flush=True)
        for k in tqdm(range(len(question_embeddings))):
            to_save[k]["question_embedding"] = question_embeddings[k]

        sanity_check = {}
        if lm.revision == "main":
            directory = "results/"+task_output.task.task_name+"/"+lm._config.name_or_path.split("/")[-1]+"/"
            os.makedirs(directory, exist_ok=True)
            f = open("results/"+task_output.task.task_name+"/"+lm._config.name_or_path.split("/")[-1]+"/"+"sanity_check.json", "w")
        else:
            directory = "results/"+task_output.task.task_name+"/"+lm._config.name_or_path.split("/")[-1]+"-"+lm.revision+"/"
            os.makedirs(directory, exist_ok=True)
            f = open("results/"+task_output.task.task_name+"/"+lm._config.name_or_path.split("/")[-1]+"-"+lm.revision+"/"+"sanity_check.json", "w")
        
        all_question_embeddings = [to_save[i]["question_embedding"] for i in range(len(to_save))]
        question_similarity_matrix = util.pytorch_cos_sim(all_question_embeddings, all_question_embeddings)
        
        all_logits = torch.stack([torch.tensor(to_save[j]["dist_of_correct"]) for j in range(len(to_save))]).to(lm.device)
        all_logits = all_logits.reshape((all_logits.shape[0], all_logits.shape[2]))
        logits_similarity_matrix = util.pytorch_cos_sim(all_logits, all_logits)

        result_matrix1 = ((logits_similarity_matrix > 0.95) & (question_similarity_matrix > 0.6)).float()
        result_matrix2 = ((logits_similarity_matrix > 0.9) & (question_similarity_matrix > 0.5)).float()
        result_matrix3 = ((logits_similarity_matrix > 0.85) & (question_similarity_matrix > 0.4)).float()
        result_matrix4 = ((logits_similarity_matrix > 0.8) & (question_similarity_matrix > 0.35)).float()
        result_matrix5 = ((logits_similarity_matrix > 0.75) & (question_similarity_matrix > 0.3)).float()
        
        indices_to_remove1, indices_to_remove2, indices_to_remove3, indices_to_remove4, indices_to_remove5 = [], [], [], [], []
        indices_to_remove1_emb, indices_to_remove2_emb, indices_to_remove3_emb, indices_to_remove4_emb, indices_to_remove5_emb = [], [], [], [], []
        indices_to_remove1_log, indices_to_remove2_log, indices_to_remove3_log, indices_to_remove4_log, indices_to_remove5_log = [], [], [], [], []

        for k in tqdm(range(len(to_save))):
            sanity_check[questions[k]] = {"f1_le":[], "f2_le":[], "f3_le":[], "f4_le":[], "f5_le":[]}
            for j in range(k+1, len(to_save)):
                if (len(indices_to_remove1) <= int(len(to_save)*0.1)) and (k not in indices_to_remove1) and (j not in indices_to_remove1) and (result_matrix1[k, j] == 1):
                    indices_to_remove1.append(j)
                    sanity_check[questions[k]]["f1_le"].append(questions[j]) 
                
                if (len(indices_to_remove2) <= int(len(to_save)*0.25)) and (k not in indices_to_remove2) and (j not in indices_to_remove2) and (result_matrix2[k, j] == 1):
                    indices_to_remove2.append(j)
                    sanity_check[questions[k]]["f2_le"].append(questions[j]) 

                if (len(indices_to_remove3) <= int(len(to_save)*0.5)) and (k not in indices_to_remove3) and (j not in indices_to_remove3) and (result_matrix3[k, j] == 1):
                    indices_to_remove3.append(j)
                    sanity_check[questions[k]]["f3_le"].append(questions[j]) 

                if (len(indices_to_remove4) <= int(len(to_save)*0.75)) and (k not in indices_to_remove4) and (j not in indices_to_remove4) and (result_matrix4[k, j] == 1):
                    indices_to_remove4.append(j)
                    sanity_check[questions[k]]["f4_le"].append(questions[j]) 

                if (len(indices_to_remove5) <= int(len(to_save)*0.9)) and (k not in indices_to_remove5) and (j not in indices_to_remove5) and (result_matrix5[k, j] == 1):
                    indices_to_remove5.append(j)
                    sanity_check[questions[k]]["f5_le"].append(questions[j]) 

        for k in tqdm(range(len(to_save))):
            sanity_check[questions[k]]["f1_e"], sanity_check[questions[k]]["f2_e"], sanity_check[questions[k]]["f3_e"], sanity_check[questions[k]]["f4_e"], sanity_check[questions[k]]["f5_e"],  = [], [], [], [], []
            for j in range(k+1, len(to_save)):
                if (len(indices_to_remove1_emb) <= int(len(to_save)*0.1)) and (k not in indices_to_remove1_emb) and (j not in indices_to_remove1_emb) and (question_similarity_matrix[k, j] > 0.6):
                    indices_to_remove1_emb.append(j)
                    sanity_check[questions[k]]["f1_e"].append(questions[j])

                if (len(indices_to_remove2_emb) <= int(len(to_save)*0.25)) and (k not in indices_to_remove2_emb) and (j not in indices_to_remove2_emb) and (question_similarity_matrix[k, j] > 0.5):
                    indices_to_remove2_emb.append(j)
                    sanity_check[questions[k]]["f2_e"].append(questions[j]) 

                if (len(indices_to_remove3_emb) <= int(len(to_save)*0.5)) and (k not in indices_to_remove3_emb) and (j not in indices_to_remove3_emb) and (question_similarity_matrix[k, j] > 0.4):
                    indices_to_remove3_emb.append(j)
                    sanity_check[questions[k]]["f3_e"].append(questions[j])

                if (len(indices_to_remove4_emb) <= int(len(to_save)*0.75)) and (k not in indices_to_remove4_emb) and (j not in indices_to_remove4_emb) and (question_similarity_matrix[k, j] > 0.35):
                    indices_to_remove4_emb.append(j)
                    sanity_check[questions[k]]["f4_e"].append(questions[j])
                
                if (len(indices_to_remove5_emb) <= int(len(to_save)*0.9)) and (k not in indices_to_remove5_emb) and (j not in indices_to_remove5_emb) and (question_similarity_matrix[k, j] > 0.3):
                    indices_to_remove5_emb.append(j)
                    sanity_check[questions[k]]["f5_e"].append(questions[j])

        for k in tqdm(range(len(to_save))):
            sanity_check[questions[k]]["f1_l"], sanity_check[questions[k]]["f2_l"], sanity_check[questions[k]]["f3_l"], sanity_check[questions[k]]["f4_l"], sanity_check[questions[k]]["f5_l"],  = [], [], [], [], []
            for j in range(k+1, len(to_save)):
                if (len(indices_to_remove1_log) <= int(len(to_save)*0.1)) and (k not in indices_to_remove1_log) and (j not in indices_to_remove1_log) and (logits_similarity_matrix[k, j] > 0.95):
                    indices_to_remove1_log.append(j)
                    sanity_check[questions[k]]["f1_l"].append(questions[j])

                if (len(indices_to_remove2_log) <= int(len(to_save)*0.25)) and (k not in indices_to_remove2_log) and (j not in indices_to_remove2_log) and (logits_similarity_matrix[k, j] > 0.9):
                    indices_to_remove2_log.append(j)
                    sanity_check[questions[k]]["f2_l"].append(questions[j]) 

                if (len(indices_to_remove3_log) <= int(len(to_save)*0.5)) and (k not in indices_to_remove3_log) and (j not in indices_to_remove3_log) and (logits_similarity_matrix[k, j] > 0.85):
                    indices_to_remove3_log.append(j)
                    sanity_check[questions[k]]["f3_l"].append(questions[j])

                if (len(indices_to_remove4_log) <= int(len(to_save)*0.75)) and (k not in indices_to_remove4_log) and (j not in indices_to_remove4_log) and (logits_similarity_matrix[k, j] > 0.8):
                    indices_to_remove4_log.append(j)
                    sanity_check[questions[k]]["f4_l"].append(questions[j])
                
                if (len(indices_to_remove5_log) <= int(len(to_save)*0.9)) and (k not in indices_to_remove5_log) and (j not in indices_to_remove5_log) and (logits_similarity_matrix[k, j] > 0.75):
                    indices_to_remove5_log.append(j)
                    sanity_check[questions[k]]["f5_l"].append(questions[j])

        indices_to_remove1, indices_to_remove2, indices_to_remove3, indices_to_remove4, indices_to_remove5 = list(set(indices_to_remove1)), list(set(indices_to_remove2)), list(set(indices_to_remove3)), list(set(indices_to_remove4)), list(set(indices_to_remove5))
        total_score_acc1, total_score_acc2, total_score_acc3, total_score_acc4, total_score_acc5 = sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove1]), sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove2]), sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove3]), sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove4]), sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove5])

        indices_to_remove1_emb, indices_to_remove2_emb, indices_to_remove3_emb, indices_to_remove4_emb, indices_to_remove5_emb = list(set(indices_to_remove1_emb)), list(set(indices_to_remove2_emb)), list(set(indices_to_remove3_emb)), list(set(indices_to_remove4_emb)), list(set(indices_to_remove5_emb))
        total_score_acc1_emb, total_score_acc2_emb, total_score_acc3_emb, total_score_acc4_emb, total_score_acc5_emb = sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove1_emb]), sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove2_emb]), sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove3_emb]), sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove4_emb]), sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove5_emb])

        indices_to_remove1_log, indices_to_remove2_log, indices_to_remove3_log, indices_to_remove4_log, indices_to_remove5_log = list(set(indices_to_remove1_log)), list(set(indices_to_remove2_log)), list(set(indices_to_remove3_log)), list(set(indices_to_remove4_log)), list(set(indices_to_remove5_log))
        total_score_acc1_log, total_score_acc2_log, total_score_acc3_log, total_score_acc4_log, total_score_acc5_log = sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove1_log]), sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove2_log]), sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove3_log]), sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove4_log]), sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove5_log])

        total_score_acc1_rand, total_score_acc2_rand, total_score_acc3_rand, total_score_acc4_rand, total_score_acc5_rand = 0, 0, 0, 0, 0
        for _ in range(5):
            indices_to_remove1_rand, indices_to_remove2_rand, indices_to_remove3_rand, indices_to_remove4_rand, indices_to_remove5_rand = random.sample(range(len(to_save)), int(len(to_save) * 0.10)), random.sample(range(len(to_save)), int(len(to_save) * 0.25)), random.sample(range(len(to_save)), int(len(to_save) * 0.5)), random.sample(range(len(to_save)), int(len(to_save) * 0.75)), random.sample(range(len(to_save)), int(len(to_save) * 0.9))
            total_score_acc1_rand += sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove1_rand]) 
            total_score_acc2_rand += sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove2_rand])
            total_score_acc3_rand += sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove3_rand])
            total_score_acc4_rand += sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove4_rand])
            total_score_acc5_rand += sum([to_save[k]["acc"] for k in range(len(to_save)) if k not in indices_to_remove5_rand])
        
        total_score_acc1_rand /= 5
        total_score_acc2_rand /= 5
        total_score_acc3_rand /= 5
        total_score_acc4_rand /= 5
        total_score_acc5_rand /= 5

        if lm.revision == "main":
            print("MODEL: {}".format(lm._config.name_or_path))
        else:
            print("MODEL: {}".format(lm._config.name_or_path.split("/")[-1]+"-"+lm.revision))
        
        print("TASK: {}".format(task_output.task.task_name))
        
        print("Filter Ratio LE (1,2,3,4,5):" + str(len(indices_to_remove1)/len(to_save)), str(len(indices_to_remove2)/len(to_save)), str(len(indices_to_remove3)/len(to_save)), str(len(indices_to_remove4)/len(to_save)), str(len(indices_to_remove5)/len(to_save)))
        print("Filtered Accuracy LE (1,2,3,4,5):" + str(float(total_score_acc1/(len(to_save)-len(indices_to_remove1)))), str(float(total_score_acc2/(len(to_save)-len(indices_to_remove2)))), str(float(total_score_acc3/(len(to_save)-len(indices_to_remove3)))), str(float(total_score_acc4/(len(to_save)-len(indices_to_remove4)))), str(float(total_score_acc5/(len(to_save)-len(indices_to_remove5)))))
        
        print("Filter Ratio E (1,2,3,4,5):" + str(len(indices_to_remove1_emb)/len(to_save)), str(len(indices_to_remove2_emb)/len(to_save)), str(len(indices_to_remove3_emb)/len(to_save)), str(len(indices_to_remove4_emb)/len(to_save)), str(len(indices_to_remove5_emb)/len(to_save)))
        print("Filtered Accuracy E (1,2,3,4,5):" + str(float(total_score_acc1_emb/(len(to_save)-len(indices_to_remove1_emb)))), str(float(total_score_acc2_emb/(len(to_save)-len(indices_to_remove2_emb)))), str(float(total_score_acc3_emb/(len(to_save)-len(indices_to_remove3_emb)))), str(float(total_score_acc4_emb/(len(to_save)-len(indices_to_remove4_emb)))), str(float(total_score_acc5_emb/(len(to_save)-len(indices_to_remove5_emb)))))

        print("Filter Ratio L (1,2,3,4,5):" + str(len(indices_to_remove1_log)/len(to_save)), str(len(indices_to_remove2_log)/len(to_save)), str(len(indices_to_remove3_log)/len(to_save)), str(len(indices_to_remove4_log)/len(to_save)), str(len(indices_to_remove5_log)/len(to_save)))
        print("Filtered Accuracy L (1,2,3,4,5):" + str(float(total_score_acc1_log/(len(to_save)-len(indices_to_remove1_log)))), str(float(total_score_acc2_log/(len(to_save)-len(indices_to_remove2_log)))), str(float(total_score_acc3_log/(len(to_save)-len(indices_to_remove3_log)))), str(float(total_score_acc4_log/(len(to_save)-len(indices_to_remove4_log)))), str(float(total_score_acc5_log/(len(to_save)-len(indices_to_remove5_log)))))
       

        print("Filter Ratio R (1,2,3,4,5):" + str(len(indices_to_remove1_rand)/len(to_save)), str(len(indices_to_remove2_rand)/len(to_save)), str(len(indices_to_remove3_rand)/len(to_save)), str(len(indices_to_remove4_rand)/len(to_save)), str(len(indices_to_remove5_rand)/len(to_save)))
        print("Filtered Accuracy R (1,2,3,4,5):" + str(float(total_score_acc1_rand/(len(to_save)-len(indices_to_remove1_rand)))), str(float(total_score_acc2_rand/(len(to_save)-len(indices_to_remove2_rand)))), str(float(total_score_acc3_rand/(len(to_save)-len(indices_to_remove3_rand)))), str(float(total_score_acc4_rand/(len(to_save)-len(indices_to_remove4_rand)))), str(float(total_score_acc5_rand/(len(to_save)-len(indices_to_remove5_rand)))))

        json.dump(sanity_check, f, indent=2, ensure_ascii=False)
        all_embeddings = [(questions[i], to_save[i]["question_embedding"]) for i in range(len(to_save))]
        all_embeddings_f1 = [(questions[i], to_save[i]["question_embedding"]) for i in range(len(to_save)) if i not in indices_to_remove1]
        all_embeddings_f2 = [(questions[i], to_save[i]["question_embedding"]) for i in range(len(to_save)) if i not in indices_to_remove2]
        all_embeddings_f3 = [(questions[i], to_save[i]["question_embedding"]) for i in range(len(to_save)) if i not in indices_to_remove3]
        all_embeddings_f4 = [(questions[i], to_save[i]["question_embedding"]) for i in range(len(to_save)) if i not in indices_to_remove4]
        all_embeddings_f5 = [(questions[i], to_save[i]["question_embedding"]) for i in range(len(to_save)) if i not in indices_to_remove5]
        
        counter = 0 
        for emb in [all_embeddings, all_embeddings_f1, all_embeddings_f2, all_embeddings_f3, all_embeddings_f4, all_embeddings_f5]:
            try:
                if counter in [1, 2, 4]:
                    counter += 1
                    continue
                
                if len(emb) > 30:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
                else:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=len(emb), n_iter=1000)
                
                embeddings = [e[1] for e in emb]
                questions = [e[0] for e in emb]
                emb = torch.tensor(embeddings).cpu()
                data_2d = tsne.fit_transform(emb)
                k = 5
                n_clusters = int(len(emb) / k) + 1
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(data_2d.astype(np.float64))
                
                labels = kmeans.labels_ 
                cluster_indices = {}
                for index, label in enumerate(labels):
                    if label not in cluster_indices.keys():
                        cluster_indices[float(label)] = []
                    cluster_indices[float(label)].append(questions[index])
                
                x_min, x_max = data_2d[:, 0].min() - 1, data_2d[:, 0].max() + 1
                y_min, y_max = data_2d[:, 1].min() - 1, data_2d[:, 1].max() + 1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

                Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()].astype(np.float64))  # Ensure float64 type
                Z = Z.reshape(xx.shape)
                plt.figure(figsize=(10, 8))
                plt.contourf(xx, yy, Z, alpha=0.5, cmap='viridis')  # Background color according to clustering

                scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, edgecolor='k', cmap='viridis', s=5)
                plt.title('t-SNE with k-Means Clustering and Background')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.colorbar(scatter, ticks=range(n_clusters), label='Cluster Label')  # Color bar for cluster labels
                if lm.revision == "main":
                    plt.savefig("results/"+task_output.task.task_name+"/"+lm._config.name_or_path.split("/")[-1]+"/"+str(counter)+"clustering.png", dpi=300, bbox_inches='tight')
                    f = open("results/"+task_output.task.task_name+"/"+lm._config.name_or_path.split("/")[-1]+"/"+str(counter)+"clustering.json", "w")
                    json.dump(cluster_indices, f, ensure_ascii=False, indent=2)
                else:
                    plt.savefig("results/"+task_output.task.task_name+"/"+lm._config.name_or_path.split("/")[-1]+"-"+lm.revision+"/"+str(counter)+"clustering.png", dpi=300, bbox_inches='tight')
                    f = open("results/"+task_output.task.task_name+"/"+lm._config.name_or_path.split("/")[-1]+"-"+str(counter)+lm.revision+"/"+"clustering.json", "w")
                    json.dump(cluster_indices, f, ensure_ascii=False, indent=2)
                counter += 1
            except ValueError:
                counter += 1
                pass

        end_time = time.time()
        execution_time = end_time - start_time

        if lm.revision == "main":
            print(f'Execution time for {task_output.task.task_name} + {lm._config.name_or_path.split("/")[-1]}: {execution_time:.6f} seconds', flush=True)
        else:
            print(f'Execution time for {task_output.task.task_name} + {lm._config.name_or_path.split("/")[-1]+"-"+lm.revision}: {execution_time:.6f} seconds', flush=True)
            
        print("="*50)

    if WORLD_SIZE > 1:
        # if multigpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            if log_samples:
                # for task_name, task_samples in list(samples.items()):
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.logged_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )

                if RANK == 0:
                    task_output.logged_samples = list(
                        itertools.chain.from_iterable(full_samples)
                    )

            # then collect metrics across all ranks
            for metrics in task_output.sample_metrics:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.sample_metrics[metrics],
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(
                        itertools.chain.from_iterable(metric_list)
                    )

    if RANK == 0:
        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
        (
            results,
            samples,
            configs,
            versions,
            num_fewshot,
            higher_is_better,
        ) = consolidate_results(eval_tasks)

        ### Calculate group metrics ###
        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(
                results, versions, task_dict
            )

        results_agg, group_agg = prepare_print_tasks(task_dict, results)
        subtask_list = get_subtask_list(task_dict)

        # collect all higher_is_better values for metrics
        # in the group's subtasks.
        # TODO: clean this up ; unify with the below metric_list loop?
        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if (
                len(task_list) != 0
            ):  # subtask list will list "task_name": [] for solo tasks
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better.keys():
                            _higher_is_better[m] = h

                        if (
                            m in _higher_is_better
                            and _higher_is_better[m] is not None
                            and _higher_is_better[m] != h
                        ):
                            eval_logger.warning(
                                f"Higher_is_better values for metric {m} in group {group} are not consistent. Defaulting to None."
                            )
                            _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better

        results_dict = {
            "results": dict(results_agg.items()),
            **(
                {"groups": dict(group_agg.items())}
                if (bool(group_agg) & show_group_table)
                else {}
            ),
            "group_subtasks": dict(reversed(subtask_list.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": {
                task_output.task_name: {
                    "original": len(task_output.task.eval_docs),
                    "effective": min(
                        limit if limit else len(task_output.task.eval_docs),
                        len(task_output.task.eval_docs),
                    ),
                }
                for task_output in eval_tasks
            },
        }
        if log_samples:
            results_dict["samples"] = dict(samples)

        return results_dict

    else:
        return None


def request_caching_arg_to_dict(cache_requests: str) -> dict:
    request_caching_args = {
        "cache_requests": cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": cache_requests == "refresh",
        "delete_requests_cache": cache_requests == "delete",
    }

    return request_caching_args