QUESTION_WORDS = ["who", "what", "where", "when", "why", "how"]

SUBJECT_MAP = {
    "me": "I",
    "i": "I",
    "he": "he",
    "she": "she",
    "you": "you",
    "they": "they"
}

BE_WORDS = ["sad", "happy", "hungry", "thirsty", "good", "bad", "great", "scared"]

VERB_PAST = {
    "go": "went",
    "eat": "ate",
    "see": "saw",
    "come": "came",
    "drink": "drank",
    "work": "worked",
    "try": "tried",
    "move": "moved",
    "walk": "walked"
}


# -------------------------
# FAST TEXT CLEANER
# -------------------------
def light_clean(text: str) -> str:
    text = text.strip()

    fixes = {
        " iam ": " I am ",
        " i ": " I ",
        " dont ": " don't ",
        " cant ": " can't ",
        " wont ": " won't ",
    }

    for k, v in fixes.items():
        text = text.replace(k, v)

    return text


# -------------------------
# TENSE DETECTION
# -------------------------
def detect_tense(words):
    if "yesterday" in words:
        return "past"
    elif "tomorrow" in words:
        return "future"
    return "present"


# -------------------------
# STATEMENT BUILDER
# -------------------------
def build_statement(words):
    words = [w.lower() for w in words]

    subject = SUBJECT_MAP.get(words[0], words[0])

    time_words = [w for w in words if w in ["yesterday", "tomorrow", "today"]]
    words = [w for w in words if w not in ["yesterday", "tomorrow", "today"]]

    if len(words) < 2:
        return subject

    verb = words[1]
    rest = " ".join(words[2:]) if len(words) > 2 else ""

    tense = detect_tense(words)

    # BE sentence
    if verb in BE_WORDS:
        be = "am" if subject == "I" else "is" if subject in ["he", "she"] else "are"
        sentence = f"{subject} {be} {verb}"

    elif tense == "future":
        sentence = f"{subject} will {verb} {rest}".strip()

    elif tense == "past":
        sentence = f"{subject} {VERB_PAST.get(verb, verb + 'ed')} {rest}".strip()

    else:
        sentence = f"{subject} {verb} {rest}".strip()

    if time_words:
        return f"{time_words[0].capitalize()} {sentence}"

    return sentence


# -------------------------
# QUESTION BUILDER
# -------------------------
def build_question(words):
    words = [w.lower() for w in words]

    if len(words) < 2:
        return " ".join(words) + "?"

    q = words[0]
    subject = SUBJECT_MAP.get(words[1], words[1])

    be = "am" if subject == "I" else "is" if subject in ["he", "she"] else "are"

    if q == "who":
        return f"Who {be} {subject}?"

    if q == "why":
        return f"Why {be} {subject}?"

    if q == "how":
        return f"How {be} {subject}?"

    if q == "what":
        return f"What {be} {subject}?"

    return " ".join(words) + "?"


# -------------------------
# SMART RESTRUCTURE
# -------------------------
def smart_restructure(text):
    words = text.lower().split()

    sentences = []
    current = []

    for w in words:
        if w in QUESTION_WORDS:
            if current:
                sentences.append(build_statement(current))
                current = []
            current.append(w)
        else:
            current.append(w)

    if current:
        if current[0] in QUESTION_WORDS:
            sentences.append(build_question(current))
        else:
            sentences.append(build_statement(current))

    return ". ".join(s.capitalize() for s in sentences)


# -------------------------
# MAIN API (REPLACEMENT FOR YOUR OLD ONE)
# -------------------------
def build_sentence(text):
    text = light_clean(text)
    text = smart_restructure(text)
    return text


# -------------------------
# CLI TEST
# -------------------------
if __name__ == "__main__":
    import sys
    input_text = " ".join(sys.argv[1:])
    print(build_sentence(input_text))