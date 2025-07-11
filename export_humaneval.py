import json
from human_eval.data import read_problems

# 1) Load all HumanEval tasks into a list
problems_dict = read_problems()            # returns { task_id: {...} }
problems = list(problems_dict.values())    # list of problem dicts

# 2) Write them out as JSONL
with open("programmer_data.jsonl", "w") as fout:
    for entry in problems:
        prompt   = entry["prompt"].strip()
        solution = entry["canonical_solution"].strip()
        obj = {
            "prompt": f"### Prompt:\n{prompt}\n"
                      f"### Completion:\n```python\n{solution}\n```"
        }
        fout.write(json.dumps(obj) + "\n")

print(f"Wrote {len(problems)} entries to programmer_data.jsonl")
