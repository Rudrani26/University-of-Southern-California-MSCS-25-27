#!/usr/bin/env python3

import sys
import time
import psutil

# -----------------------------
# Scoring parameters (fixed)
# -----------------------------

GAP_PENALTY = 30

ALPHA = {
    'A': {'A': 0,   'C': 110, 'G': 48,  'T': 94},
    'C': {'A': 110, 'C': 0,   'G': 118, 'T': 48},
    'G': {'A': 48,  'C': 118, 'G': 0,   'T': 110},
    'T': {'A': 94,  'C': 48,  'G': 110, 'T': 0}
}


# -----------------------------
# Utility: process memory in KB
# -----------------------------

def process_memory_kb():
    process = psutil.Process()
    mem_info = process.memory_info()
    return int(mem_info.rss / 1024)


# -----------------------------
# String generation from input
# -----------------------------

def generate_string(base, indices):
    s = base
    for idx_str in indices:
        i = int(idx_str)
        # Insert a copy of the whole string s just after index i (0-based)
        s = s[:i + 1] + s + s[i + 1:]
    return s


def parse_input_file(input_path):
    with open(input_path, 'r') as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    # First base string
    s0 = raw_lines[0]

    # Collect indices for s until we hit a non-numeric line
    s_indices = []
    pos = 1
    while pos < len(raw_lines) and raw_lines[pos].isdigit():
        s_indices.append(raw_lines[pos])
        pos += 1

    # Now this line is the second base string
    t0 = raw_lines[pos]
    pos += 1

    # Remaining lines are indices for t
    t_indices = []
    while pos < len(raw_lines):
        if raw_lines[pos].isdigit():
            t_indices.append(raw_lines[pos])
        pos += 1

    s = generate_string(s0, s_indices)
    t = generate_string(t0, t_indices)

    return s, t


# -----------------------------
# Basic DP alignment
# -----------------------------

def sequence_alignment_dp(X, Y):
    m = len(X)
    n = len(Y)

    # Full DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize first row and column
    for i in range(1, m + 1):
        dp[i][0] = i * GAP_PENALTY
    for j in range(1, n + 1):
        dp[0][j] = j * GAP_PENALTY

    # Fill DP table
    for i in range(1, m + 1):
        xi = X[i - 1]
        for j in range(1, n + 1):
            yj = Y[j - 1]
            cost_match = ALPHA[xi][yj]

            cost_diag = dp[i - 1][j - 1] + cost_match      # align xi with yj
            cost_up   = dp[i - 1][j] + GAP_PENALTY         # xi with gap
            cost_left = dp[i][j - 1] + GAP_PENALTY         # gap with yj

            dp[i][j] = min(cost_diag, cost_up, cost_left)

    # Backtrack to get aligned strings
    i, j = m, n
    aligned_X = []
    aligned_Y = []

    while i > 0 or j > 0:
        # If both i and j > 0, we can consider diagonal
        if i > 0 and j > 0:
            xi = X[i - 1]
            yj = Y[j - 1]
            cost_match = ALPHA[xi][yj]
            if dp[i][j] == dp[i - 1][j - 1] + cost_match:
                aligned_X.append(xi)
                aligned_Y.append(yj)
                i -= 1
                j -= 1
                continue

        # Check if coming from top (gap in Y)
        if i > 0 and dp[i][j] == dp[i - 1][j] + GAP_PENALTY:
            aligned_X.append(X[i - 1])
            aligned_Y.append('_')
            i -= 1
        # Else from left (gap in X)
        else:
            aligned_X.append('_')
            aligned_Y.append(Y[j - 1])
            j -= 1

    # Reverse to get correct order
    aligned_X.reverse()
    aligned_Y.reverse()

    total_cost = dp[m][n]
    return total_cost, ''.join(aligned_X), ''.join(aligned_Y)


# -----------------------------
# Main entry point
# -----------------------------

def main():
    if len(sys.argv) != 3:
        # Per project spec, no extra printing; just exit silently on wrong usage
        return

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Generate the two strings from input file
    s, t = parse_input_file(input_path)

    # Measure memory and time around the DP algorithm
    before_mem = process_memory_kb()
    start_time = time.time()

    cost, aligned_s, aligned_t = sequence_alignment_dp(s, t)

    end_time = time.time()
    after_mem = process_memory_kb()

    time_ms = (end_time - start_time) * 1000.0
    mem_kb = float(after_mem - before_mem)

    # Write output: 5 lines as required
    with open(output_path, 'w') as out:
        out.write(str(cost) + '\n')
        out.write(aligned_s + '\n')
        out.write(aligned_t + '\n')
        out.write(str(time_ms) + '\n')
        out.write(str(mem_kb) + '\n')


if __name__ == "__main__":
    main()
