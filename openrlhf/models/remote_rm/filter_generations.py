import pandas as pd

def filter_generations(df):
    """
    Filter generations based on matching indices where both is_reasoning_complete 
    and correctness_math_verify are True. Takes only the first match if multiple exist.
    
    Args:
        df: pandas DataFrame containing the columns:
            - is_reasoning_complete: list of boolean values or single boolean
            - correctness_math_verify: list of boolean values or single boolean
            - generations: list of strings or single string
            
    Returns:
        DataFrame with filtered_output column containing the first matching generation
    """
    filtered_outputs = []
    
    # Process each row
    for idx in range(len(df)):
        reasoning = df['is_reasoning_complete'].iloc[idx]
        correctness = df['correctness_math_verify'].iloc[idx]
        generations = df['generations'].iloc[idx]
        
        # Handle single boolean case
        if isinstance(reasoning, bool):
            filtered = generations if reasoning and correctness else None
        else:
            # Handle list case
            matching_indices = [
                i for i in range(len(reasoning))
                if reasoning[i] and correctness[i]
            ]
            # Take only the first match if exists
            filtered = generations[matching_indices[0]] if matching_indices else None
        
        filtered_outputs.append(filtered)
    
    # Add filtered results as a new column
    df['filtered_output'] = filtered_outputs
    
    return df

# Example usage:
if __name__ == "__main__":
    # Sample data with both single and multiple matches
    data = {
        'is_reasoning_complete': [
            [True, True, False, True],  # multiple booleans
            True,                       # single boolean
            [True, False, True],        # multiple booleans
        ],
        'correctness_math_verify': [
            [True, False, True, True],  # multiple booleans
            True,                       # single boolean
            [True, False, False],       # multiple booleans
        ],
        'generations': [
            ["text1", "text2", "text3", "text4"],  # list of texts
            "single_text",                         # single text
            ["text5", "text6", "text7"],          # list of texts
        ]
    }
    
    df = pd.DataFrame(data)
    result = filter_generations(df)
    
    print("\nResults for each row:")
    for idx in range(len(result)):
        print(f"\nRow {idx + 1}:")
        print("is_reasoning_complete:", result['is_reasoning_complete'].iloc[idx])
        print("correctness_math_verify:", result['correctness_math_verify'].iloc[idx])
        print("generations:", result['generations'].iloc[idx])
        print("filtered_output:", result['filtered_output'].iloc[idx])
