# Quiz Builder (LangChain + Google Gemini)

An interactive quiz generator built with LangChain and Google Gemini. It demonstrates both classic LangChain chains and the modern LangChain Expression Language (LCEL) to:

- Generate a beginner-level question for a given topic
- Provide an answer with a short explanation
- Optionally generate a subtle hint before revealing the answer
- Run an interactive Q&A flow in the console/notebook

## Project Contents

- `quiz_generator.ipynb` — the main notebook containing all examples:
  - Classic chains with `LLMChain` and `SequentialChain`
  - Modern LCEL pipeline using `|` composition
  - Advanced 3-step chain: question → hint → answer
  - `interactive_quiz()` function for a simple interactive experience

## Prerequisites

- Python 3.9+ recommended
- A Google Generative AI API key with access to Gemini models
- Jupyter environment (e.g., JupyterLab, VS Code, or any notebook runner)

## Setup

1) Clone/open this repository and go to the quiz builder folder:

```
TOPICS/ai_agents/quiz_builder
```

2) Create and activate a virtual environment (optional but recommended):

```
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows (PowerShell/CMD)
```

3) Install dependencies. The notebook installs packages directly, but you can also install them yourself first for faster starts:

```
pip install -U langchain langchain-community langchain-core langchain-google-genai python-dotenv pandas langchain-classic
```

## Configure Environment Variables

The notebook loads a `.env` file from the current working directory to read your API key. Create a file named `.env` alongside the notebook with:

```
GOOGLE_API_KEY=your_api_key_here
```

Notes:
- You can generate an API key from Google AI Studio or your Google Cloud project where Generative AI is enabled.
- The notebook uses the model ID `gemini-2.5-flash` by default and `temperature=0.7`.

## How to Run

1) Open `quiz_generator.ipynb` in Jupyter.
2) Run the cells in order. The first cell upgrades/installs required packages; you can skip it if your environment already has them.
3) Example flows included:
   - Basic quiz: topic → question → answer
   - LCEL version of the above
   - Advanced chain that adds a hint step
   - Interactive mode that asks you to enter a topic and type an answer

When the `interactive_quiz()` cell runs, you will be prompted in the notebook to:
- Enter a topic
- Type your answer
The notebook will then display the generated question, your answer, and the model’s answer with an explanation.

## Troubleshooting

- API key not found:
  - Ensure `.env` exists in your current working directory when running the notebook
  - Verify `GOOGLE_API_KEY` is set and has no surrounding quotes or spaces
  - Restart the kernel after creating/updating `.env`

- Import errors (e.g., `ModuleNotFoundError`):
  - Re-run the first cell that installs packages, or run the `pip install` command from Setup
  - Verify the active interpreter/virtual environment is the one where packages are installed

- Rate limits or quota errors:
  - Reduce the number of runs or try again later
  - Consider lowering temperature or switching to a more cost-efficient model if available

- Notebook input prompts not appearing:
  - Make sure you’re executing in an environment that supports `input()` in notebooks (Jupyter, VS Code Notebooks, etc.)

## Customization Tips

- Change the default topic by editing the corresponding cell values.
- Adjust prompt templates in the notebook for different phrasing or difficulty.
- Modify `temperature` and the `model` name in the `ChatGoogleGenerativeAI` initialization for different model behavior.

## License

This folder follows the repository’s overall license and usage terms. Ensure you comply with Google’s terms for Generative AI APIs.
