from transformers import pipeline
import numpy as np

# User and market performance data (daily returns)
user_performance = [0.5, -0.2, 1.0, -0.3, 0.7]  # User's daily returns (%)
sp500_performance = [0.3, -0.1, 0.8, -0.2, 0.6]  # S&P 500 returns (%)

# Function to calculate alpha (difference between user's return and S&P 500 return)
def calculate_alpha(user_returns, market_returns):
    # Calculate daily alpha: user return - market return
    alpha = np.array(user_returns) - np.array(market_returns)
    return alpha.mean()  # Return average alpha

# Use Hugging Face's GPT-2 pipeline (or another model of your choice)
generator = pipeline('text-generation', model='gpt2')

# Function to answer user questions using Hugging Face's GPT-2
def get_trading_performance_answer(question):
    # Prepare the context for GPT-2 to generate an answer
    prompt = f"Given the following performance data:\n"
    prompt += f"User's daily returns: {user_performance}\n"
    prompt += f"S&P 500 daily returns: {sp500_performance}\n"
    prompt += f"Question: {question}\n"
    
    # Generate a response using Hugging Face's GPT-2 model
    response = generator(prompt, max_length=200, num_return_sequences=1)
    
    # Extract and return the answer
    return response[0]['generated_text'].strip()

# Example usage of the function
if __name__ == "__main__":
    # Calculate alpha for the user
    alpha = calculate_alpha(user_performance, sp500_performance)
    print(f"User's average alpha: {alpha:.2f}%\n")

    # Demo: Asking the GPT-2 model about trading performance
    question1 = "What was my alpha last week?"
    question2 = "How did I perform compared to the S&P 500?"
    
    print("Answer to Question 1:")
    print(get_trading_performance_answer(question1))

    print("\nAnswer to Question 2:")
    print(get_trading_performance_answer(question2))
