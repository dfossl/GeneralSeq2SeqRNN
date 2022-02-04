import numpy as np
import tensorflow as tf


def getBidirectionalStates(states, layerNum, layertype, state=None):
  """Helper function that can retrieve the forward and reverse states for a given
  layer in a bidrectional RNN. 

  Args:
      states (List of Tensors): Holds returned output states from Bidirectional
                                RNN model call.
      layerNum (int): What layers state is being retrieved.
      layertype (str): gru, rnn, or lstm. lstm need special prcessing for context
                        and hidden state.
      state (str, optional): if lstm, then 'hidden' or 'context' must be provided
                            to identify which state to return

  Raises:
      ValueError: If lstm is provided without a state.
      ValueError: if rnn or gru are provided with a state.

  Returns:
      List of Tensors: Retrieves the forward and reverse states for given layer.
  """
  start = 2*(layerNum-1)
  if layertype == "lstm":
      if state == "hidden":
          return [states[start][0], states[start+1][0]]
      elif state == "context":
          return [states[start][1], states[start+1][1]]
      else:
          raise ValueError(f"[layertype == lstm but state = {state}]: \
                            state must be 'hidden' or 'context' for lstm")
  elif layertype in ["rnn", "gru"]:
      if state:
          raise ValueError(f"[layertype == {layertype} but state = {state}]: \
                              state must be None for {layertype}")
      
      return [states[start], states[start+1]]



class Encoder(tf.keras.layers.Layer):
  """
  Custom Encoder class that extends Tensorflow layer.
  Constructs an encoder of the provided architecture
  """
  def __init__(self, input_vocab_size, enc_units, layertype="rnn", layers=1,
                isBidirectional=False):
    """[summary]

    Args:
        input_vocab_size (int): size of input vocabulary.
        enc_units (int): number of encoder units.
        layertype (str, optional): gru, rnn, or lstm. Defaults to "rnn".
        layers (int, optional): Number of RNN layers. Defaults to 1.
        isBidirectional (bool, optional): Use Bidirectionality. Defaults to False.

    Raises:
        ValueError: For invalid layer type.
    """
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.input_vocab_size = input_vocab_size
    self.layertype=layertype
    self.layers = layers
    self.isBidirectional = isBidirectional



    if not layertype in {"gru", "lstm", "rnn"}:
      raise ValueError(f"[layertype == {layertype}]: layertype must be one of:\
                          [gru, lstm, rnn]")    

    cells = []
    if self.layertype == "gru": 
      for _ in range(self.layers):
        cells.append(tf.keras.layers.GRUCell(units=enc_units,
                                              recurrent_initializer='glorot_uniform',
                                              recurrent_dropout=.2))
    elif self.layertype == "lstm":
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
      if self.layertype == "lstm":
        # lstm need to concatinate both context and hidden states.
        output, *s = self.rnn(oh_input, initial_state=state)
        state = []
        for i in range(self.layers):
          layerHiddenStates = getBidirectionalStates(s, layerNum=i+1, 
                                                     layertype=self.layertype, 
                                                     state="hidden")
          concatHiddenStates = tf.keras.layers.Concatenate()(layerHiddenStates)

          layerContextStates = getBidirectionalStates(s, layerNum=i+1, 
                                                      layertype=self.layertype, 
                                                      state="context")
          concatContextStates = tf.keras.layers.Concatenate()(layerContextStates)

          state.append([concatHiddenStates, concatContextStates])
      else:
        # is GRU or RNN
        output, *s = self.rnn(oh_input, initial_state=state)

        state = []
        for i in range(self.layers):
          layerStates = getBidirectionalStates(s,
                                               layerNum=i+1,
                                               layertype=self.layertype)
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