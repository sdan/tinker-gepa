# GEPA: Prompt Optimization via LLM Reflection

I spent the last week building GEPA within thinking machines' tinker-cookbook. GEPA basically does prompt optimization by sampling a bunch of times, having a larger LLM(teacher) reflect on failures and mutate the prompt within an environment we define. Environment creation is the harder part since you need to be able to let a teacher know how to reward/score prompts with a lot of Q/A pairs. I included GSM8K, HotpotQA, and AIME environments as an example.

This utilizes Tinker API's batch sampling to take all the prompts we generate on one rollout, score them, then run another rollout all in one batch sync. It doesn't update the models weights.

**Paper**: [arxiv.org/abs/2507.19457](https://arxiv.org/abs/2507.19457) | **Library**: [github.com/gepa-ai/gepa](https://github.com/gepa-ai/gepa)

## Installation

This recipe requires the [Tinker API](https://thinkingmachines.ai/) and [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook).

```bash
# Get API access at https://thinkingmachines.ai/
export TINKER_API_KEY=sk-...

# Install dependencies
pip install tinker
pip install git+https://github.com/thinking-machines-lab/tinker-cookbook.git

# Clone this repo
git clone https://github.com/sdan/tinker-gepa.git
cd tinker-gepa
```

## Running This Recipe

### GSM8K

```bash
python train.py \
    task_name=gsm8k \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    reflection_model="deepseek-ai/DeepSeek-V3.1" \
    max_metric_calls=50
```

After optimization, expect `final/best_score` around 0.91.

### HotpotQA

```bash
python train.py \
    task_name=hotpotqa \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    reflection_model="deepseek-ai/DeepSeek-V3.1" \
    max_metric_calls=100
```

### AIME

```bash
python train.py \
    task_name=aime \
    model_name="Qwen/Qwen3-4B-Instruct-2507" \
    reflection_model="deepseek-ai/DeepSeek-V3.1" \
    max_metric_calls=150 \
    eval_test=true
```

## Custom Tasks

Register via `TASK_REGISTRY`:

```python
from tasks import GEPATask, register_task

@register_task("my_benchmark")
class MyBenchmarkTask(GEPATask):
    @property
    def name(self) -> str:
        return "my_benchmark"

    @property
    def seed_prompt(self) -> str:
        return "You are a helpful assistant..."

    def load_data(self, seed: int = 0):
        return train, val, test  # GEPADataInstance lists

    def score(self, response: str, answer: str, metadata: dict | None = None) -> float:
        return 1.0 if answer.strip() in response else 0.0
```

Then: `python train.py task_name=my_benchmark`
