import os
import sys
import time
import psutil

GAP_PENALTY = 30
MATCH_SCORE = {
    ('A','A'):0,   ('A','C'):110, ('A','G'):48,  ('A','T'):94,
    ('C','A'):110, ('C','C'):0,   ('C','G'):118, ('C','T'):48,
    ('G','A'):48,  ('G','C'):118, ('G','G'):0,   ('G','T'):110,
    ('T','A'):94,  ('T','C'):48,  ('T','G'):110, ('T','T'):0
}

def read_input(file_path: str) -> tuple[str, list[int], str, list[int]]:
    """
    Reads the input file and returns the base strings and their indices for generation.
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    idx = 0
    base_X = lines[idx]
    idx += 1

    indices_X = []
    while idx < len(lines) and lines[idx].isdigit():
        indices_X.append(int(lines[idx]))
        idx += 1

    base_Y = lines[idx]
    idx += 1

    indices_Y = []
    while idx < len(lines):
        indices_Y.append(int(lines[idx]))
        idx += 1

    return base_X, indices_X, base_Y, indices_Y


def build_sequence(base: str, indices: list[int]) -> str:
    """
    Builds the full string from the base string using the indices.
    """
    sequence = base
    for index in indices:
        if index < 0 or index >= len(sequence):
            if index == len(sequence) - 1:
                sequence = sequence + sequence
            else:
                raise ValueError("Invalid insertion index")
        else:
            pos = index + 1
            sequence = sequence[:pos] + sequence + sequence[pos:]
    return sequence


def compute_score(str1: str, str2: str) -> list[int]:
    """
    Computes the alignment score using space-efficient DP.
    """
    m, n = len(str1), len(str2)
    prev_row = [j * GAP_PENALTY for j in range(n + 1)]

    for i in range(1, m + 1):
        curr_row = [0] * (n + 1)
        curr_row[0] = i * GAP_PENALTY
        char1 = str1[i - 1]
        for j in range(1, n + 1):
            cost_diag = prev_row[j - 1] + MATCH_SCORE[(char1, str2[j - 1])]
            cost_up   = prev_row[j] + GAP_PENALTY
            cost_left = curr_row[j - 1] + GAP_PENALTY
            curr_row[j] = min(cost_diag, cost_up, cost_left)
        prev_row = curr_row
    return prev_row


def align_small(str1: str, str2: str) -> tuple[str, str]:
    """
    Performs DP alignment for small strings.
    """
    m, n = len(str1), len(str2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        dp[i][0] = dp[i-1][0] + GAP_PENALTY
    for j in range(1, n+1):
        dp[0][j] = dp[0][j-1] + GAP_PENALTY

    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = min(
                dp[i-1][j-1] + MATCH_SCORE[(str1[i-1], str2[j-1])],
                dp[i-1][j] + GAP_PENALTY,
                dp[i][j-1] + GAP_PENALTY
            )

    # traceback
    i, j = m, n
    aligned1, aligned2 = [], []
    while i > 0 or j > 0:
        if i>0 and j>0 and dp[i][j] == dp[i-1][j-1] + MATCH_SCORE[(str1[i-1], str2[j-1])]:
            aligned1.append(str1[i-1])
            aligned2.append(str2[j-1])
            i -= 1
            j -= 1
        elif i>0 and dp[i][j] == dp[i-1][j] + GAP_PENALTY:
            aligned1.append(str1[i-1])
            aligned2.append('_')
            i -= 1
        else:
            aligned1.append('_')
            aligned2.append(str2[j-1])
            j -= 1

    aligned1.reverse()
    aligned2.reverse()
    return "".join(aligned1), "".join(aligned2)


def divide_and_align(str1: str, str2: str) -> tuple[str, str]:
    """
    Performs divide and conquer alignment for larger strings.
    """
    m, n = len(str1), len(str2)

    if m == 0:
        return '_'*n, str2
    if n == 0:
        return str1, '_'*m
    if m == 1 or n == 1:
        return align_small(str1, str2)

    mid = m // 2

    score_left = compute_score(str1[:mid], str2)
    score_right = compute_score(str1[mid:][::-1], str2[::-1])

    best_cut = 0
    best_val = None
    for k in range(n+1):
        val = score_left[k] + score_right[n-k]
        if best_val is None or val < best_val:
            best_val = val
            best_cut = k

    left1, left2 = divide_and_align(str1[:mid], str2[:best_cut])
    right1, right2 = divide_and_align(str1[mid:], str2[best_cut:])

    return left1 + right1, left2 + right2


def get_memory_usage() -> int:
    """
    Returns memory usage of the current process in KB.
    """
    process = psutil.Process()
    return int(process.memory_info().rss / 1024)  # KB


def execute(input_file: str, output_file: str, answer_file: str = None) -> bool:
    """
    Main execution function to align sequences and write output.
    """
    try:
        base_X, indices_X, base_Y, indices_Y = read_input(input_file)
    except FileNotFoundError:
        return False

    seq_X = build_sequence(base_X, indices_X)
    seq_Y = build_sequence(base_Y, indices_Y)

    mem_before = get_memory_usage()
    start_time = time.perf_counter()

    aligned_X, aligned_Y = divide_and_align(seq_X, seq_Y)

    end_time = time.perf_counter()
    mem_after = get_memory_usage()

    total_cost = 0
    for a, b in zip(aligned_X, aligned_Y):
        if a == '_' or b == '_':
            total_cost += GAP_PENALTY
        else:
            total_cost += MATCH_SCORE[(a, b)]

    try:
        with open(output_file, "w") as f:
            f.write(f"{total_cost}\n")
            f.write(f"{aligned_X}\n")
            f.write(f"{aligned_Y}\n")
            f.write(f"{(end_time - start_time) * 1000}\n")
            f.write(f"{mem_after-mem_before}")
    except Exception as e:
        print("Error writing output file")

    return True


if __name__ == "__main__":
    inp = sys.argv[1]
    out = sys.argv[2]

    execute(input_file=inp, output_file=out)
    sys.exit(0)
