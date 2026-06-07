import random

WORDS = [
    "BOSS","BYE","CAMERA","COFFEE","COLLEGE","COME","COMPUTER","DISLIKE","DOOR","DOWN",
    "FRIEND","GIVE","GO","GOOD","GREAT","HE","HELLO","HUNGRY","LEFT","LOVE","MAYBE","ME",
    "MINE","MOVE","NO","PHONE","ROOM","SAD","SCARED","SEE","SIT","SLEEP","SORRY","STAND",
    "STUDENT","TEA","THINK","THIRSTY","TODAY","TOMORROW","TRY","UP","VEGETABLE","WAIT",
    "WALK","WATER","WHICH","WHO","WHY","WORK","YESTERDAY","YOU"
]

SUBJECTS = ["ME", "YOU", "HE"]
VERBS = ["GO", "COME", "EAT", "DRINK", "WALK", "WORK", "SEE", "TRY", "MOVE"]
OBJECTS = ["COFFEE", "TEA", "WATER", "VEGETABLE", "ROOM", "COLLEGE", "COMPUTER", "PHONE"]
FEELINGS = ["HUNGRY", "THIRSTY", "SAD", "SCARED", "GOOD", "GREAT"]

TIME_WORDS = ["YESTERDAY", "TODAY", "TOMORROW"]

QUESTIONS = ["WHO", "WHY", "WHICH"]


def generate_simple():
    s = random.choice(SUBJECTS)
    v = random.choice(VERBS)
    o = random.choice(OBJECTS)
    return f"{s} {v} {o}"


def generate_with_time():
    t = random.choice(TIME_WORDS)
    s = random.choice(SUBJECTS)
    v = random.choice(VERBS)
    return f"{t} {s} {v}"


def generate_feeling():
    s = random.choice(SUBJECTS)
    f = random.choice(FEELINGS)
    return f"{s} {f}"


def generate_question():
    q = random.choice(QUESTIONS)
    s = random.choice(SUBJECTS)
    return f"{q} {s}"


def generate_multi():
    return f"{generate_simple()} {generate_question()}"


def generate_tests(n=100):
    tests = []

    for _ in range(n):
        choice = random.choice([
            generate_simple,
            generate_with_time,
            generate_feeling,
            generate_question,
            generate_multi
        ])
        tests.append(choice())

    return tests


if __name__ == "__main__":
    tests = generate_tests(200)

    with open("test_sentences.txt", "w") as f:
        for t in tests:
            f.write(t + "\n")

    print("✅ Generated test_sentences.txt with 200 samples")