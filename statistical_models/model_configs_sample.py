# Using 2 model families with 4 datasets each = 16 total files
ALL_CONFIGS = [
    # Llama-3 70B
    {
        'model': 'llama_70b', 
        'dataset': 'disinfo', 
        'base_path': '../sample_data/llama_70b/original/disinfo_original_outputs.csv', 
        'ft_path': '../sample_data/llama_70b/warm/disinfo_warm_outputs.csv'
    },
    {
        'model': 'llama_70b', 
        'dataset': 'medqa', 
        'base_path': '../sample_data/llama_70b/original/medqa_original_outputs.csv', 
        'ft_path': '../sample_data/llama_70b/warm/medqa_warm_outputs.csv'
    },
    {
        'model': 'llama_70b', 
        'dataset': 'trivia', 
        'base_path': '../sample_data/llama_70b/original/trivia_original_outputs.csv', 
        'ft_path': '../sample_data/llama_70b/warm/trivia_warm_outputs.csv'
    },
    {
        'model': 'llama_70b', 
        'dataset': 'truthfulqa', 
        'base_path': '../sample_data/llama_70b/original/truthfulqa_original_outputs.csv', 
        'ft_path': '../sample_data/llama_70b/warm/truthfulqa_warm_outputs.csv'
    },
    
    # Qwen-32B
    {
        'model': 'qwen_32b', 
        'dataset': 'disinfo', 
        'base_path': '../sample_data/qwen_32b/original/disinfo_original_outputs.csv', 
        'ft_path': '../sample_data/qwen_32b/warm/disinfo_warm_outputs.csv'
    },
    {
        'model': 'qwen_32b', 
        'dataset': 'medqa', 
        'base_path': '../sample_data/qwen_32b/original/medqa_original_outputs.csv', 
        'ft_path': '../sample_data/qwen_32b/warm/medqa_warm_outputs.csv'
    },
    {
        'model': 'qwen_32b', 
        'dataset': 'trivia', 
        'base_path': '../sample_data/qwen_32b/original/trivia_original_outputs.csv', 
        'ft_path': '../sample_data/qwen_32b/warm/trivia_warm_outputs.csv'
    },
    {
        'model': 'qwen_32b', 
        'dataset': 'truthfulqa', 
        'base_path': '../sample_data/qwen_32b/original/truthfulqa_original_outputs.csv', 
        'ft_path': '../sample_data/qwen_32b/warm/truthfulqa_warm_outputs.csv'
    },
]