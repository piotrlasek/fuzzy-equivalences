def generate_idx(t, k):
    if k == 0:
        raise ValueError("k must be greater than 0")

    index_step = len(t) / k  # Calculate the step between each index based on the table size and k
    return [int(index_step * i) for i in range(k)]


# Example usage:
t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
k = 5
result = generate_table(t, k)
print(result)
