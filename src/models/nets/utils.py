def calculate_num_filters(num_strided_layers, bw_factor):
    return 2 * 3 * (2 ** (2 * num_strided_layers)) // bw_factor
