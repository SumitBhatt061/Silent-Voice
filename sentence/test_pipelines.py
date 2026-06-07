from sentence.sentence_builder import correct_sentence

INPUT_FILE = "sentence/test_sentences.txt"
OUTPUT_FILE = "sentence/results.txt"


def run_tests():
    with open(INPUT_FILE, "r") as f:
        lines = f.readlines()

    results = []

    for line in lines:
        inp = line.strip()
        out = correct_sentence(inp)

        results.append(f"INPUT: {inp}")
        results.append(f"OUTPUT: {out}")
        results.append("-" * 40)

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(results))

    print("✅ Testing complete → results.txt generated")


if __name__ == "__main__":
    run_tests()