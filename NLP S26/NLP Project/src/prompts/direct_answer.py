_DA_TEMPLATE = (
    "Read the following context carefully and answer the multiple choice "
    "question.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "A. {choice_a}\n"
    "B. {choice_b}\n"
    "C. {choice_c}\n"
    "D. {choice_d}\n\n"
    "Answer with a single letter: A, B, C, or D."
)


def build_da_prompt(instance: dict) -> str:
    return _DA_TEMPLATE.format(
        context=instance["input"],
        question=instance["question"],
        choice_a=instance["choices"][0],
        choice_b=instance["choices"][1],
        choice_c=instance["choices"][2],
        choice_d=instance["choices"][3],
    )
