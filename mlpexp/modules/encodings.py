import torch


def tanh_positional_encoding(X, res=10, sharpness_values=[1,2,4,8]):
    num_sharpness_values = len(sharpness_values)
    encoded = torch.zeros((X.size(0), X.size(1) * 2 * res * num_sharpness_values))
    frequencies = 2 ** torch.linspace(0, res - 1, res)
    for i in range(X.size(1)):
        for j, freq in enumerate(frequencies):
            for s, sharpness in enumerate(sharpness_values):
                offset = s * 2
                index = i * 2 * res * num_sharpness_values + j * 2 * num_sharpness_values + offset
                encoded[:, index] = torch.tanh(sharpness * torch.sin(X[:, i] * freq))
                encoded[:, index + 1] = torch.tanh(sharpness * torch.cos(X[:, i] * freq))
    return encoded