import logging
import os
import re
from argparse import ArgumentParser, Namespace

import pandas as pd
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Load a HF model and generate evaluation responses.",
    )

    parser.add_argument("--model_name_or_path", type=str, help="HF model name or path.")
    parser.add_argument("--hf_ds_path", type=str, help="HF dataset path.")
    parser.add_argument("--hf_ds_split", type=str, help="HF dataset split name.")
    parser.add_argument("--batch_size", type=int, help="Max batch size.")
    parser.add_argument(
        "--pad_to_multiple_of", type=int, help="Pads batch to multiples of a value."
    )
    parser.add_argument("--max_new_tokens", type=int, help="Max new tokens.")
    parser.add_argument(
        "--max_token_count_message", type=int, help="Max number of tokens in message."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path should be in form 'username/dataset_name'",
    )
    return parser.parse_args()


def get_data(hf_ds_path: str, hf_ds_split: str):
    ds = load_dataset(
        path=hf_ds_path,
        split=hf_ds_split,
        use_auth_token=os.environ.get("HF_READ_TOKEN"),
    )
    df = ds.to_pandas()
    messages = df["messages"].tolist()
    _ids = df["_id"].tolist()
    labels = df["LABEL"].tolist()
    inputs_list = list(zip(labels, _ids, messages))

    new_messages = [[data[2][0]] for data in inputs_list]

    return new_messages, inputs_list


def get_pattern_from_text(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.DOTALL)
    if match:
        value = match.group(1)
    else:
        value = "ERROR"

    return value


def get_scores_from_generated_text(text: str):
    pattern = r'"SCORE":\s*"(\w+)"'
    return get_pattern_from_text(text=text, pattern=pattern)


def get_reasoning_from_generated_text(text: str):
    pattern = r'"REASONING":\s*\[(.*?)\]'
    return get_pattern_from_text(text=text, pattern=pattern)


def get_accuracy_and_correct_match(df: pd.DataFrame) -> tuple[float, float]:
    n_correct = sum(df["LABEL"] == df["score"])
    accuracy = n_correct / df.shape[0]
    logger.info(f"Accuracy: {accuracy * 100:.2f}%")

    return n_correct, accuracy


def calculate_accuracies(df: pd.DataFrame):
    new_df = df[df["score"] != "TOO LONG CONTEXT"].copy()
    pass_df = new_df[new_df["LABEL"] == "PASS"]
    fail_df = new_df[new_df["LABEL"] == "FAIL"]
    all_rows = new_df.shape[0]
    all_pass_rows = pass_df.shape[0]
    all_fail_rows = fail_df.shape[0]
    if not all_rows:
        logger.info("DataFrame is empty. ")
    if all_rows > 0:
        n_correct, accuracy = get_accuracy_and_correct_match(new_df)
        logger.info(f"All rows: {all_rows}")
        logger.info(f"Correct examples: {n_correct}   Accuracy: {accuracy}")
        if not all_pass_rows:
            logger.info("PASS DataFrame is empty.")
        if not all_fail_rows:
            logger.info("FAIL DataFrame is empty.")

        if all_pass_rows > 0:
            n_pass_correct, pass_accuracy = get_accuracy_and_correct_match(pass_df)
            logger.info(f"All PASS rows: {all_pass_rows}")
            logger.info(
                f"Correct PASS examples: {n_pass_correct}   PASS Accuracy: {pass_accuracy}"
            )
        if all_fail_rows > 0:
            n_fail_correct, fail_accuracy = get_accuracy_and_correct_match(fail_df)
            logger.info(f"All FAIL rows: {all_fail_rows}")
            logger.info(
                f"Correct FAIL examples: {n_fail_correct}   FAIL Accuracy: {fail_accuracy}"
            )


def main():
    logger.info("Parse inputs.")
    args = parse_args()

    distributed_state = PartialState()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=distributed_state.device,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    messages, inputs_list = get_data(
        hf_ds_path=args.hf_ds_path, hf_ds_split=args.hf_ds_split
    )

    def count_tokens(text):
        return len(tokenizer.encode(text))

    # Limit messages to those below a certain token count
    messages_count = [
        count_tokens(m[0]["content"]) < args.max_token_count_message for m in messages
    ]
    inputs_list = [il for i, il in enumerate(inputs_list) if messages_count[i]]
    messages = [m for i, m in enumerate(messages) if messages_count[i]]

    # Split into batches
    formatted_message_batches = [
        messages[i : i + args.batch_size]
        for i in range(0, len(messages), args.batch_size)
    ]

    # Apply padding on the left since we are doing generation
    padding_side_default = tokenizer.padding_side
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_messages_batches = [
        tokenizer.apply_chat_template(
            formatted_message_batch,
            padding=True,
            pad_to_multiple_of=args.pad_to_multiple_of,
            return_tensors="pt",
            return_dict=True,
        )
        for formatted_message_batch in formatted_message_batches
    ]

    # Put back the original padding behavior
    tokenizer.padding_side = padding_side_default

    completions_per_process = []
    with distributed_state.split_between_processes(
        tokenized_messages_batches, apply_padding=True
    ) as batched_messages:
        for batch in batched_messages:
            # Move the batch to the device
            batch = batch.to(distributed_state.device)
            # We generate the text, decode it and add it to the list completions_per_process
            outputs = model.generate(**batch, max_new_tokens=args.max_new_tokens)
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            completions_per_process.extend(generated_text)

            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    # We are gathering string, so we need to use gather_object.
    # If you need to gather tensors, you can use gather from accelerate.utils
    completions_gather = gather_object(completions_per_process)

    # Drop duplicates produced by apply_padding in split_between_processes
    completions = completions_gather[: len(messages)]

    if distributed_state.is_last_process:
        split_completions_by = '"SCORE":\n{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}'
        completions = [c.split(split_completions_by)[1] for c in completions]
        scores = [get_scores_from_generated_text(c) for c in completions]
        reasoning = [get_reasoning_from_generated_text(c) for c in completions]

        df = pd.DataFrame(inputs_list)
        df.columns = [
            "LABEL",
            "_id",
            "messages",
        ]
        df["generated_text"] = completions
        df["reasoning"] = reasoning
        df["score"] = scores
        col_order = ["_id", "LABEL", "score", "messages", "reasoning", "generated_text"]

        logger.info("Last row of the dataframe:")
        logger.info(df.iloc[-1]._id)
        logger.info(df.iloc[-1].LABEL)
        logger.info(df.iloc[-1].score)
        logger.info(df.iloc[-1].messages)
        logger.info(df.iloc[-1].reasoning)
        logger.info(df.iloc[-1].generated_text)

        df = df[col_order]

        calculate_accuracies(df)

        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(f"{args.output_path}")
        logger.info(f"Saved dataset to HF Hub : {args.output_path}")


if __name__ == "__main__":
    main()
