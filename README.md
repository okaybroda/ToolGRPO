# ToolGRPO

Trains language models to learn how to use tools through reinforcement learning. The project uses Group Relative Policy Optimization (GRPO) to fine-tune language models, teaching them to effectively utilize tools like code execution to solve complex problems.

The project builds upon the GRPO algorithm proposed in the paper "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" and extends it to support tool-augmented language models. The primary test case uses the AIME (American Invitational Mathematics Examination) dataset, where the model learns to solve mathematical problems by writing and executing Python code.

## Features

- Fine-tuning of language models using GRPO algorithm
- Support for tool usage and code generation
- Python code execution and evaluation capabilities
- Weights & Biases integration for experiment tracking
- Fast inference using vLLM
- LoRA fine-tuning support

## Installation

First, install Rye if you haven't already:

```bash
# Install Rye
curl -sSf https://rye.astral.sh/get | bash
```

Then, install and set up the project:

```bash
# Clone the repository
git clone https://github.com/yourusername/toolgrpo.git
cd toolgrpo

# Install dependencies and set up the environment
rye sync
```

## Usage

To run the tool GRPO trainer:

```bash
rye run python src/toolgrpo/tool_grpo.py
```

The main implementation in `tool_grpo.py` sets up a custom trainer with the following features:
- Uses Qwen/Qwen2.5-0.5B-Instruct as the base model
- Configures LoRA fine-tuning with rank 128
- Supports Python code execution and evaluation
- Integrates with Weights & Biases for experiment tracking

## Project Structure

- `src/toolgrpo/`: Main package directory
  - `tool_grpo.py`: Core implementation of the GRPO trainer with tool usage
  - `grpo_trainer.py`: Base GRPO trainer implementation
  - `python_interpreter.py`: Python code execution and evaluation utilities

## Training Issues and Challenges

For detailed information about current training challenges, limitations, and mitigation strategies, please see the [Training Issues](docs/training_issues.md) document

## Future Advancements

### Multi-Turn Tool Usage
The project aims to extend the current single-turn tool usage to support multi-turn interactions, where the language model can:
- Maintain context across multiple tool calls
- Learn to chain tool calls effectively
- Handle tool execution failures gracefully
- Make decisions about when to use which tool based on intermediate results

### Multi-Tool Training
Future development will focus on training models to use multiple tools simultaneously:
- Integration of diverse tools (e.g., web search, code execution, API calls)
- Training on heterogeneous datasets to improve tool selection
- Development of tool-specific reward functions
- Implementation of tool usage patterns and best practices

The goal is to create more versatile language models that can:
- Dynamically select appropriate tools for different tasks
- Combine multiple tools to solve complex problems
- Learn from tool interaction patterns
- Adapt to new tools with minimal additional training

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
