import re

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    list_of_all_chars = list(text)
    unique_chars = list(set(list_of_all_chars))
    # sort the characters
    unique_chars.sort()
    indices = range(len(unique_chars))
    # Create the two dictionaries with indices
    char_to_idx = dict(zip(unique_chars, indices))
    idx_to_char = dict(zip(indices, unique_chars))
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    n_removed = 0
    text_clean = text
    for c in chars_to_remove:
        n_removed += text.count(c)
        text_clean = text_clean.replace(c, "")
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    N = len(text)
    D = len(char_to_idx)
    result = torch.zeros([N, D], dtype=torch.int8)
    for idx, c in enumerate(text):
        result[idx, char_to_idx[c]] = 1
    # ========================
    return result

def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    result = ''.join([idx_to_char[one_hot.tolist().index(1)] for one_hot in embedded_text])
    # ========================
    return result

def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
                                #(len(text), V) -> (len(text)/seq_len, seq_len, V)
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======

    # instructions are not clear. what to do with remaining chars? that is, (len(text) % seq_len > 0)
    """    pad_last_string = False
    if pad_last_string:
        a = list(embed_text[:-1, :].split(seq_len, dim=0))
        S, V = a[0].size()
        K, _ = a[-1].size()
        pad = torch.zeros(S-K, V)
        a[-1] = torch.cat((a[-1], pad), dim=0)
        samples = torch.stack(a, dim=0)
    else: # ignore remaining letters
    """

    """
    An easier explanation for what we get in samples and labels is the following:
    samples is the text embedded to one-hot vectors, and then split to groups of seq_len (we get N full groups).
    labels is the same as samples, but shifted by one char to the right (we get N full groups).
    But in compare to samples, the embedding is not represented by one-hot vectors, but by indices.
    Reminder: one-hot it a vector of zeros, where there is '1' in the index of the indices of the char. 
    in labels instead of the one-hot vector we just write the indices number.
    That is, the labels is a tensor of shape (N, S) instead of (N, S, V).
    """

    # samples tensor shape
    S = seq_len
    N = (len(text) - 1) // S
    V = len(char_to_idx)

    # Embed the given text
    embed_text = chars_to_onehot(text[:N * S+1], char_to_idx)

    # sample first N*S chars starting from pos 0
    samples = embed_text[:-1, :]

    # split to N groups of size S
    samples = torch.reshape(samples, shape=(N, S, V)).to(device)

    # represent text as indices
    labels = numpy.array([char_to_idx[c] for c in text[1:N*S+1]])

    # numpy array to tensor
    labels = torch.tensor(labels, dtype=torch.int8)

    # split to N chunks of size S
    labels = torch.reshape(labels, shape=(N, S)).to(device)

    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======

    exponents = torch.exp((y-torch.max(y))/temperature)
    # Divide each element by the sum of all elements
    result = exponents / torch.sum(exponents, axis=dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======

    with torch.no_grad():
        input_to_model = start_sequence
        hidden_state = None

        while len(out_text) < n_chars:
            # Embed the given text
            embed_input = chars_to_onehot(input_to_model, char_to_idx).to(dtype=torch.float, device=device)
            # Add a batch dimension of 1
            embed_input = embed_input.unsqueeze(0)
            # Feed the start_sequence into the model
            output, hidden_state = model(embed_input, hidden_state)
            # Removing first batch dimension (equal to 1) and taking last char of output
            # After the first loop, the output will always be of len 1 because we input one char
            output = output.squeeze(0)[-1]
            # Calculate the probabilities of all chars to be sampled
            probabilities = hot_softmax(output, temperature=T)
            # Sample one char index from probabilities
            new_char_idx = torch.multinomial(probabilities, num_samples=1).item()
            # Convert the idx of char to a char
            new_char = idx_to_char[new_char_idx]
            # Append the new char to the output
            out_text += new_char
            # Feed the new char into the model in the next loop
            input_to_model = new_char

    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======

        indices = range(0, len(self.dataset) - (len(self.dataset) % self.batch_size))
        indices = np.array_split(indices, self.batch_size)
        idx = np.transpose(indices).flatten()

        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======

        # TODO: check this
        cur_in_dim = in_dim
        dropout_layer = nn.Dropout(dropout)
        for k in range(n_layers):
            # Create layer weights
            # Only need bias once so it is disabled for "x" matrices and enabled for h.
            W_xz = nn.Linear(cur_in_dim, self.h_dim, bias=False)
            W_xr = nn.Linear(cur_in_dim, self.h_dim, bias=False)
            W_xg = nn.Linear(cur_in_dim, self.h_dim, bias=False)

            W_hz = nn.Linear(self.h_dim, self.h_dim, bias=True)
            W_hr = nn.Linear(self.h_dim, self.h_dim, bias=True)
            W_hg = nn.Linear(self.h_dim, self.h_dim, bias=True)

            # Register modules
            self.add_module(f'layer_{k}: W_xz', W_xz)
            self.add_module(f'layer_{k}: W_xr', W_xr)
            self.add_module(f'layer_{k}: W_xg', W_xg)
            self.add_module(f'layer_{k}: W_hz', W_hz)
            self.add_module(f'layer_{k}: W_hr', W_hr)
            self.add_module(f'layer_{k}: W_hg', W_hg)

            self.add_module(f'layer_{k}: dropout', dropout_layer)

            # Add params to layer params
            self.layer_params += [W_xz, W_xr, W_xg, W_hz, W_hr, W_hg, dropout_layer]

            cur_in_dim = h_dim
        self.W_hy = nn.Linear(cur_in_dim, self.out_dim, bias=True)
        self.add_module(f'W_hy', self.W_hy)
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        '''
        z = FF(x,h,bias)
        r = FF(x,h,bias)
        g = tanh(x, r*h, bias)
        h = lerp(h, g, z) = z*h + (1-z)*g

        run flow: bottom to top 
        '''
        sigmoid = nn.Sigmoid()
        tanh = nn.Tanh()
        out_y_list = []
        for t in range(seq_len):
            X = input[:, t, :]
            for i in range(0, len(self.layer_params), 7):
                # Get the 7 layer params for each layer
                W_xz, W_xr, W_xg, W_hz, W_hr, W_hg, dropout_layer = tuple(self.layer_params[i:i + 7])

                # Calculate current layer from i
                k = i // 7

                # Get hidden layer
                h_in = layer_states[k]

                # Calculate Gru pass
                z = sigmoid(W_xz(X) + W_hz(h_in))
                r = sigmoid(W_xr(X) + W_hr(h_in))
                g = tanh(W_xg(X) + W_hg(r * h_in))
                h_out = z * h_in + (1 - z) * g

                X = dropout_layer(h_out)
                # Save hidden state
                layer_states[k] = h_out

                # Calculate next layer input
            out_y_list.append(self.W_hy(X))
        layer_output = torch.stack(out_y_list, dim=1)
        hidden_state = torch.stack(layer_states, dim=1)
        # ========================
        return layer_output, hidden_state