from datasets import load_dataset

def preprocess_function(example):
    # Check if the 'input' field is present and is a string
    if 'input' not in example or not isinstance(example['input'], str):
        # Handle the case when 'input' is missing or not a string
        # You can choose to skip the example or provide a default value
        example['input'] = ''

    # Check if the 'label' field is present and is an integer
    if 'label' not in example or not isinstance(example['label'], int):
        # Handle the case when 'label' is missing or not an integer
        # You can choose to skip the example or provide a default value
        example['label'] = 0

    return example

# Load the dataset from the Hugging Face Hub
dataset_repo_id = "ruliad/math_value_net_experiment_data-2"

dataset = load_dataset(dataset_repo_id)

# Apply the preprocessing function to each example in the dataset
dataset = dataset.map(preprocess_function, num_proc=4)

# Save the preprocessed dataset in JSONL format
dataset.to_json("data/preprocessed_dataset.jsonl")