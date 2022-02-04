import numpy as np
import tensorflow as tf


def get_bidirectional_states(states, layer_num, layer_type, state=None):
  """Helper function that can retrieve the forward and reverse states for a given
  layer in a bidrectional RNN. 

  Args:
      states (List of Tensors): Holds returned output states from Bidirectional
                                RNN model call.
      layer_num (int): What layers state is being retrieved.
      layer_type (str): gru, rnn, or lstm. lstm need special prcessing for context
                        and hidden state.
      state (str, optional): if lstm, then 'hidden' or 'context' must be provided
                            to identify which state to return

  Raises:
      ValueError: If lstm is provided without a state.
      ValueError: if rnn or gru are provided with a state.

  Returns:
      List of Tensors: Retrieves the forward and reverse states for given layer.
  """
  start = 2*(layer_num-1)
  if layer_type == "lstm":
      if state == "hidden":
          return [states[start][0], states[start+1][0]]
      elif state == "context":
          return [states[start][1], states[start+1][1]]
      else:
          raise ValueError(f"[layer_type == lstm but state = {state}]: \
                            state must be 'hidden' or 'context' for lstm")
  elif layer_type in ["rnn", "gru"]:
      if state:
          raise ValueError(f"[layer_type == {layer_type} but state = {state}]: \
                              state must be None for {layer_type}")
      
      return [states[start], states[start+1]]



class Encoder(tf.keras.layers.Layer):
  """
  Custom Encoder class that extends Tensorflow layer.
  Constructs an encoder of the provided architecture
  """
  def __init__(self, input_vocab_size, enc_units, layer_type="rnn", layers=1,
                isBidirectional=False):
    """[summary]

    Args:
        input_vocab_size (int): size of input vocabulary.
        enc_units (int): number of encoder units.
        layer_type (str, optional): gru, rnn, or lstm. Defaults to "rnn".
        layers (int, optional): Number of RNN layers. Defaults to 1.
        isBidirectional (bool, optional): Use Bidirectionality. Defaults to False.

    Raises:
        ValueError: For invalid layer type.
    """
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.input_vocab_size = input_vocab_size
    self.layer_type=layer_type
    self.layers = layers
    self.isBidirectional = isBidirectional



    if not layer_type in {"gru", "lstm", "rnn"}:
      raise ValueError(f"[layer_type == {layer_type}]: layer_type must be one of:\
                          [gru, lstm, rnn]")    

    cells = []
    if self.layer_type == "gru": 
      for _ in range(self.layers):
        cells.append(tf.keras.layers.GRUCell(units=enc_units,
                                              recurrent_initializer='glorot_uniform',
                                              recurrent_dropout=.2))
    elif self.layer_type == "lstm":
      for _ in range(self.layers):
        cells.append(tf.keras.layers.LSTMCell(units=enc_units,
                                                recurrent_initializer='glorot_uniform',
                                                recurrent_dropout=.2))
    else:
      for _ in range(self.layers):
        cells.append(tf.keras.layers.SimpleRNNCell(units=enc_units,
                                                    recurrent_initializer='glorot_uniform',
                                                    recurrent_dropout=.2))

    if self.isBidirectional:
      self.rnn = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(cells,
                                                                  return_sequences=True,
                                                                  return_state=True,
                                                                  ))
    else:
      self.rnn = tf.keras.layers.RNN(cells,
                                      return_sequences=True,
                                      return_state=True,
                                      )

  def call(self, tokens, state=None):
    
    oh_input = tf.one_hot(tokens, depth=self.input_vocab_size)

    if self.isBidirectional:
      if self.layer_type == "lstm":
        # lstm need to concatinate both context and hidden states.
        output, *s = self.rnn(oh_input, initial_state=state)
        state = []
        for i in range(self.layers):
          layer_hidden_states = get_bidirectional_states(s, layer_num=i+1, 
                                                     layer_type=self.layer_type, 
                                                     state="hidden")
          concat_hidden_states = tf.keras.layers.Concatenate()(layer_hidden_states)

          layer_context_states = get_bidirectional_states(s, layer_num=i+1, 
                                                      layer_type=self.layer_type, 
                                                      state="context")
          concat_context_states = tf.keras.layers.Concatenate()(layer_context_states)

          state.append([concat_hidden_states, concat_context_states])
      else:
        # is GRU or RNN
        output, *s = self.rnn(oh_input, initial_state=state)

        state = []
        for i in range(self.layers):
          layerStates = get_bidirectional_states(s,
                                               layer_num=i+1,
                                               layer_type=self.layer_type)
          state.append(tf.keras.layers.Concatenate()(layerStates))
    else:
      #Not Bidirectional
      output, *state = self.rnn(oh_input, initial_state=state)

    # In single layer networks state is either one tensor or a tuple of tensors
    # in n layer networks state is a list of of n states for each layer n.
    # Shapes:
    # For GRU:  output (batch, max_input_len, dims)
    #           state (1, dims)
    # For LSTM: output (batch, max_input_len, dims)
    #           state ( h(1, dims), c(1, dims) )
    # For RNN:  output (batch, max_input_len, dims)
    #           state (1, dims)
    # If n layers then the number of states *n
    # if bidirectional dims -> 2*dims because of concatination.
    return output, state
  

class luong_like_attention(tf.keras.layers.Layer):
  """
  Implimentation of Attention layer with tensorflows implimenation of
  multiplicative attention.
  """
  def __init__(self, units):
    super().__init__()
    
    self.W1 = tf.keras.layers.Dense(units, use_bias=False)


    self.attention = tf.keras.layers.Attention()

  def call(self, query, query_mask, value, value_mask):

    w1_query = self.W1(query)


    context_vector, attention_weights = self.attention(
        inputs = [w1_query, value],
        mask=[query_mask, value_mask],
        return_attention_scores = True,
    )


    return context_vector, attention_weights