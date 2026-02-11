PLAN_PROMPT = """You are RALPH planner. Break the request into 2-5 safe, concise steps.\nRequest: {task_text}"""
ACT_PROMPT = """You are RALPH actor. Current plan:\n{plan}\nNext step: {step}\nReturn action result and progress."""
EVAL_PROMPT = """You are RALPH evaluator. Decide complete=true/false and confidence [0,1].\nTask: {task_text}\nLatest result: {latest_result}"""
