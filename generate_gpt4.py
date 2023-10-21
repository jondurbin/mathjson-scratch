import asyncio
from tqdm import tqdm
import numpy as np
import faiss
import requests
import json
import mathjson
import re
import torch
import os
import transformers
from smart_open import smart_open
from airoboros.embeddings import calculate_embeddings
from airoboros.self_instruct import SelfInstructor
from sentence_transformers import SentenceTransformer

# Instructor (just used for the generate_response method).
instructor = SelfInstructor()

# Embedding model, which we'll use to generate embeddings of known valid instructions for selecting closest in-context learning examples.
embedding_model = SentenceTransformer("thenlper/gte-large", device="cuda")
embedding_dimension = embedding_model.get_sentence_embedding_dimension()
embedding_tokenizer = transformers.AutoTokenizer.from_pretrained("thenlper/gte-large")
index = faiss.IndexFlatL2(embedding_dimension)

# Load items to generate.
to_generate = [json.loads(line) for line in smart_open('https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl').readlines()]

# Load the validated dataset.
validated = json.loads(smart_open('https://huggingface.co/datasets/jondurbin/mathjson-alpha/resolve/main/conversations.json').read())
validated = [{'question': item['conversations'][1]['value'], 'answer': item['conversations'][2]['value']} for item in validated]

# Generate embeddings of the instructions so we can get the most similar examples for in-context learning.
all_embeddings = []
for item in tqdm(validated):
    all_embeddings.append(
        np.array(
            [
                calculate_embeddings(
                    item['query'],
                    embedding_model,
                    embedding_tokenizer,
                )
            ]
        )
    )
    index.add(all_embeddings[-1])

async def get_solution(item):
    query = item['question']
    hint = item['answer']
    expected_answer = hint.split('### ')[-1].strip()
    if not re.match(r'^[0-9\.-]+$', expected_answer):
        return False, None, None
    expected_answer = float(expected_answer)
  
    # Find the closest examples.
    query_emb = np.array(
        [
            calculate_embeddings(query, embedding_model, embedding_tokenizer),
        ]
    )
    _, indices = index.search(query_emb, k=7)
    indices = indices[0].tolist()
    prompt = """MathJSON is a format used to represent solutions to problems as JSON that can be evaluated.
Available functions:
- Add
- Subtract
- Negate
- Multiply
- Divide
- Power
- Root
- Sqrt
- Square
- Exp
- Ln
- Lb
- Lg
- LogOnePlus
- Abs
- Ceil
- Chop
- Floor
- Round
- BaseForm
- Clamp
- Max
- Rational
- Gamma
- LogGamma
- SignGamma
- Sum
- Product
- Sin
- Arcsin
- Sinh
- Arsinh
- Cos
- Arccos
- Cosh
- Arcosh
- Tan
- Arctan
- Arctan2
- Tanh
- Artanh
- Cot
- Acot
- Coth
- Arcoth
- Sec
- Asec
- Sech
- Asech
- Csc
- Acsc
- Csch
- Acsch
- FromPolarCoordinates
- ToPolarCoordinates
- Hypot
- Haversine
- InverseHaversine

Trigonometry functions (Cos, Tan, etc.) require radians, so be sure if values are provided in degrees to multiply the value by Pi/180, e.g. cosine of 50 degrees would be represented as: ["Cos", ["Multiply", 50, ["Divide", "Pi", 180]]]

Be sure to pay attention to wording regarding rounding and/or whole numbers.  Some values cannot be fractional, e.g. "minimum number of buses" must always be an integer, so make use of functions like "Round", "Ceil", "Floor", "Clamp", etc. but only when appropriate/asked for a whole number of items.

Pay close attention to percentages vs ratios.

If the solution cannot be written as a MathJSON formula, respond "SKIP".

Available constants:
- ExponentialE: Eulerʼs number (https://www.wikidata.org/wiki/Q82435)
- MachineEpsilon: The difference between 1 and the next larger floating point number.
- CatalanConstant: Catalanʼs Constant on Wikipedia (https://en.wikipedia.org/wiki/Catalan%27s_constant)
- GoldenRatio: Golden Ratio on Wikipedia (https://en.wikipedia.org/wiki/Golden_ratio)
- EulerGamma: Euler-Mascheroni Constant on Wikipedia (https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant)
- Degrees: Pi / 180 =~ 0.01745329251994329
- Pi: The numeric value of Pi =~ 3.14159265358979323

"""
    for idx in indices:
        prompt += "\n".join([
            'Question: ' + validated[idx]['question'],
            '\n',
            validated[idx]['answer'],
            '\n\n',
        ])
    prompt += "\n" + "\n".join([
        'Question: ' + query,
        'Hint: ' + hint,
        '\n',
        "Here's what we know:",
    ])

    # Get the answer.
    response = await instructor.generate_response(prompt, filter_response=False, temperature=0.01)
    if not response:
        return False, None, expected_answer

    data = {"response": response}

    # Extract and evaluate the mathjson
    re_match = re.search('<mathjson>(.*?)</mathjson>', response, re.DOTALL)
    if not re_match:
        print('Could not find mathjson output.')
        return False, None, expected_answer
    solution_str = re_match.group(1)
    try:
        solution = json.loads(solution_str)
    except:
        print(f'Invalid JSON in output: {solution_str}')
        return False, None, expected_answer
    answer = None
    try:
        answer = mathjson.evaluate(solution)
    except Exception as exc:
        print(f'Evaluation fail: {exc}')
        return False, None, expected_answer
    data["answer"] = answer
    data["expected_answer"] = expected_answer
    data["original"] = item
    try:
        if round(expected_answer, 5) == round(answer, 5):
            return True, data, expected_answer
    except:
        ...
    return False, data, expected_answer

# Evaluate.
async def run():
    total = 0
    correct = 0
    for aitem in to_generate:
        is_correct, data, expected_answer = await get_solution(aitem)        
        total += 1
        answer = (data or {}).get('answer')
        if is_correct:
            correct += 1
        print(f'{item["question"]}')
        print(f'  expected: {expected_answer}')
        print(f'  inferred: {answer}')
        if is_correct:
            print('  Correct!')
        else:
            print(f'  Wrong!')
        print(f'---\nRunning total: {correct} / {total} = {correct / total}')

asyncio.run(run())
