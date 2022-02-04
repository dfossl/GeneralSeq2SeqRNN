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


class Decoder(tf.keras.layers.Layer):
  """
  Custom decoder class that impliments tensor flow layer.
  """
  def __init__(self, output_vocab_size, dec_units, layer_type="rnn", layers=1, useAttention=True):
    """
    Given vocab size, units, layer_type, number of layers, and attention and constructs a decoder
    of those properties.
    """
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.output_vocab_size = output_vocab_size

    self.layer_type=layer_type
    self.layers = layers
    self.useAttention = useAttention

    if not layer_type in {"gru", "lstm", "rnn"}:
      raise ValueError(f"[layer_type == {layer_type}]: layer_type must be one of: [gru, lstm, rnn]")    

    cells = []
    if self.layer_type == "gru": 
      for _ in range(self.layers):
        cells.append(tf.keras.layers.GRUCell(units=dec_units,
                                             recurrent_initializer='glorot_uniform',
                                             dropout=.3))
    elif self.layer_type == "lstm":
      for _ in range(self.layers):

        cells.append(tf.keras.layers.LSTMCell(units=dec_units,
                                             recurrent_initializer='glorot_uniform',
                                             dropout=.3))
    else:
      # rnn
      for _ in range(self.layers):
        cells.append(tf.keras.layers.SimpleRNNCell(units=dec_units,
                                             recurrent_initializer='glorot_uniform',
                                             dropout=.3))

    self.rnn = tf.keras.layers.RNN(cells, return_sequences=True, return_state=True)
    
    if self.useAttention:
      self.attention = luong_like_attention(self.dec_units)

      # This weighted matrix is for applying the context vector to decoder
      # output
      self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                      use_bias=False)

    self.fc = tf.keras.layers.Dense(self.output_vocab_size)

  def call(self, inputs, state=None):

      vectors = tf.one_hot(inputs["input_tokens"], depth=self.output_vocab_size)


      rnn_output, *state = self.rnn(vectors, initial_state=state)


      if self.useAttention:
        context_vector, attention_weights = self.attention(
            query=rnn_output, query_mask = inputs["dec_mask"], value=inputs["enc_output"], value_mask=inputs["enc_mask"])

        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        last_vector = self.Wc(context_and_rnn_output)

        
      else:
        attention_weights = None
        last_vector = rnn_output

      
      logits = self.fc(last_vector)

      return {"logits":logits, "attention_weights":attention_weights}, state


class MaskedLoss(tf.keras.losses.Loss):
  """
  Extenstion of Tensorflow lost class for masking padding values.
  """
  def __init__(self):
    self.name = 'masked_loss'
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="sum")

  def __call__(self, y_true, y_pred):


    # Mask off the losses on padding.
    mask = y_true != 0
    mask = tf.cast(mask, dtype=tf.int64)

    loss = self.loss(y_true, y_pred, sample_weight=mask)



    # We divide loss by reduce sum of mask as this allows us
    # to divide by the number of unmasked values.
    return loss / tf.reduce_sum(tf.cast(mask, tf.float32))


class MaskedTotalAccuracy(tf.keras.metrics.Metric):

  def __init__(self, name="masked_tot_acc", **kwargs):
    super(MaskedTotalAccuracy, self).__init__(name=name, **kwargs)
    self.sum_batch_acc = self.add_weight(name="Sum of Average Prediction Accuracies", initializer="zeros")
    self.num_batches = self.add_weight(name="Number of Batches called", initializer="zeros")
    

  def update_state(self, y_true, y_pred, sample_weight=None):

    # holds 1 for correct guess and 0 for wrong guess
    equality_true_and_pred = tf.math.equal(tf.cast(y_true, dtype=tf.int32), 
                                              tf.cast(tf.math.argmax(y_pred, axis=-1), dtype=tf.int32))


    if sample_weight is not None:

      # Negate mask to make all masked values 1
      n_sample_weight = tf.logical_not(tf.cast(sample_weight, dtype=tf.bool))

      # Since all masked values are 1 the logical or forces all padding guesses to be true
      # For total acuracy this is fine because the percentage right isn't being messaged
      # only the totality of correctness. In this way guesses can only be wrong in
      # non-padding regions.
      masked_results = tf.logical_or(n_sample_weight, equality_true_and_pred)
    else:
      masked_results = equality_true_and_pred


    collapsed_results = tf.reduce_all(masked_results, axis=1, keepdims=True)


    batch_number_true_positives = tf.reduce_sum(tf.cast(collapsed_results, dtype=tf.float32))

    batch_acc = batch_number_true_positives / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
    self.sum_batch_acc.assign_add(batch_acc)

    self.num_batches.assign_add(1.)


  def result(self):
    # Return the current average batch accruacy
    return self.sum_batch_acc/self.num_batches
  
  def reset_state(self):
    self.sum_batch_acc.assign(0.)
    self.num_batches.assign(0.)


class Seq2SeqModelConstructor(tf.keras.Model):
  """
  Seq2SeqModelConstructor extends Tensorflow model and handles the creation of the full model with
  encoder and decoder and impliments the custom training and evaluation steps.
  """
  def __init__(self,
               units,
               input_rxn_processor,
               output_rxn_processor,
               output_vocab_size,
               n_layers=1,
               layer_type="rnn",
               isBidirectional=False,
               useAttention=True):
    
    super().__init__()

    self.input_rxn_processor = input_rxn_processor
    self.output_rxn_processor = output_rxn_processor
    self.output_vocab_size = output_vocab_size
    self.n_layers = n_layers
    self.layer_type = layer_type
    self.isBidirectional = isBidirectional
    self.useAttention = useAttention


    self.encoder = Encoder(input_vocab_size=input_rxn_processor.vocabulary_size(),
                        enc_units=units,
                        layer_type=self.layer_type,
                        layers=self.n_layers,
                        isBidirectional=self.isBidirectional)
    

    if self.isBidirectional:
      dec_units = 2*units
    else:
      dec_units = units
    

    self.decoder = Decoder(output_vocab_size=output_rxn_processor.vocabulary_size(),
                                  dec_units=dec_units,
                                  layer_type=self.layer_type,
                                  layers=self.n_layers,
                                  useAttention=self.useAttention)
    

    self.seq_acc_1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    self.seq_acc_2 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)
    self.seq_acc_3 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)
    self.seq_acc_4 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=4)
    self.seq_acc_5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)
    self.tot_acc = MaskedTotalAccuracy()

  
  @property
  def metrics(self):
    return [self.seq_acc_1,self.seq_acc_2,self.seq_acc_3,self.seq_acc_4,self.seq_acc_5,self.tot_acc]
  

  def top_k_acc_loop(self, y_true, y_pred):
    """
    This only is needed because the Keras topKaccruacy seems to have issues with
    Batches?
    Here is my open issue:
    https://github.com/keras-team/keras/issues/15939
    """
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    
    mask = tf.cast(mask, dtype=tf.int32)

    for i in tf.range(tf.shape(y_true)[0]):
      self.seq_acc_1.update_state(y_true[i,:], y_pred[i,:,:], sample_weight=mask[i,:])
      self.seq_acc_2.update_state(y_true[i,:], y_pred[i,:,:], sample_weight=mask[i,:])
      self.seq_acc_3.update_state(y_true[i,:], y_pred[i,:,:], sample_weight=mask[i,:])
      self.seq_acc_4.update_state(y_true[i,:], y_pred[i,:,:], sample_weight=mask[i,:])
      self.seq_acc_5.update_state(y_true[i,:], y_pred[i,:,:], sample_weight=mask[i,:])

  def preprocess(self, input_text, target_text):
    # Convert the text to token IDs
    input_tokens = self.input_rxn_processor(input_text)
    target_tokens = self.output_rxn_processor(target_text)


    # Convert IDs to masks.
    input_mask = input_tokens != 0


    target_mask = target_tokens != 0


    return input_tokens, input_mask, target_tokens, target_mask

  @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                              tf.TensorSpec(dtype=tf.string, shape=[None])]])
  def train_step(self, inputs):
    input_text, target_text = inputs


    (input_tokens, input_mask,
    target_tokens, target_mask) = self._preprocess(input_text, target_text)


    with tf.GradientTape() as tape:
      enc_output, enc_state = self.encoder(input_tokens)

      dec_state = enc_state


      decoder_input = {"input_tokens":target_tokens[:, :-1],
                                  "dec_mask":target_mask[:, :-1],
                                  "enc_output":enc_output,
                                  "enc_mask":input_mask}

      dec_result, dec_state = self.decoder(decoder_input, state=dec_state)


      y = target_tokens[:,1:]
      y_pred = dec_result["logits"]
      average_loss = self.loss(y, y_pred)
      self.tot_acc.update_state(y, y_pred, target_mask[:,1:])


      self.top_k_acc_loop(y, y_pred)


      acc_top1 = self.seq_acc_1.result()
      acc_top2 = self.seq_acc_2.result()
      acc_top3 = self.seq_acc_3.result()
      acc_top4 = self.seq_acc_4.result()
      acc_top5 = self.seq_acc_5.result()
      totA = self.tot_acc.result()

      
        
    variables = self.trainable_variables 
    gradients = tape.gradient(average_loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))


    return {'batch_loss': average_loss,
            "acc_1":acc_top1,
            "acc_2":acc_top2,
            "acc_3":acc_top3,
            "acc_4":acc_top4,
            "acc_5":acc_top5,
            "totA":totA}


  @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]),
                                tf.TensorSpec(dtype=tf.string, shape=[None])]])
  def test_step(self, inputs):
    input_text, target_text = inputs


    (input_tokens, input_mask,
    target_tokens, target_mask) = self._preprocess(input_text, target_text)

    enc_output, enc_state = self.encoder(input_tokens)

    dec_state = enc_state


    decoder_input = {"input_tokens":target_tokens[:, :-1],
                                "dec_mask":target_mask[:, :-1],
                                "enc_output":enc_output,
                                "enc_mask":input_mask}

    dec_result, dec_state = self.decoder(decoder_input, state=dec_state)


    y = target_tokens[:,1:]
    y_pred = dec_result["logits"]


    average_loss = self.loss(y, y_pred)
    self.tot_acc.update_state(y, y_pred, target_mask[:,1:])


    self.top_k_acc_loop(y, y_pred)


    acc_top1 = self.seq_acc_1.result()
    acc_top2 = self.seq_acc_2.result()
    acc_top3 = self.seq_acc_3.result()
    acc_top4 = self.seq_acc_4.result()
    acc_top5 = self.seq_acc_5.result()
    totA = self.tot_acc.result()

    return {'batch_loss': average_loss,
            "acc_1":acc_top1,
            "acc_2":acc_top2,
            "acc_3":acc_top3,
            "acc_4":acc_top4,
            "acc_5":acc_top5,
            "totA":totA}



  def evaluate_dataset(self, dataset):
    """
    Loops over dataset batches calculating and printing metrics
    for each batch. Final results are the batch averages.

    @return Dictionary with Batch average of metrics.
    """

    total_loss = 0
    acc_top1 = 0
    acc_top2 = 0
    acc_top3 = 0
    acc_top4 = 0
    acc_top5 = 0
    totA = 0

    for batch, (input_text, target_text) in enumerate(dataset.take(-1)):


      (input_tokens, input_mask,
      target_tokens, target_mask) = self._preprocess(input_text, target_text)



      max_target_length = tf.shape(target_tokens)[1]


      enc_output, enc_state = self.encoder(input_tokens)

      dec_state = enc_state


      

      pred_tokens = target_tokens[:, 0:1]
      
      for t in range(max_target_length-1):


        decoder_input = {"input_tokens":pred_tokens,
                              "dec_mask":target_mask[:, t:t+1],
                              "enc_output":enc_output,
                              "enc_mask":input_mask}


        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)

        if t == 0:
          logits = dec_result["logits"]
        else:
          logits = tf.concat((logits, dec_result["logits"]), 1)

        pred_tokens = tf.argmax(dec_result["logits"], -1)
      

      y = target_tokens[:,1:]
      y_pred = logits

      loss = self.loss(y, y_pred).numpy()

      total_loss += loss

      self.tot_acc.update_state(y, y_pred, target_mask[:,1:])


      self.top_k_acc_loop(y, y_pred)

      print((f"Batch: {batch} - batch_loss: {loss:.3f} -"),
            (f"acc_1: {self.seq_acc_1.result().numpy():.3f} -"),
            (f"acc_2: {self.seq_acc_2.result().numpy():.3f} -"),
            (f"acc_3: {self.seq_acc_3.result().numpy():.3f} -"),
            (f"acc_4: {self.seq_acc_4.result().numpy():.3f} -"),
            (f"acc_5: {self.seq_acc_5.result().numpy():.3f} -"),
            (f"TotA: {self.tot_acc.result().numpy():.3f}"))
      
      acc_top1 += self.seq_acc_1.result().numpy()
      acc_top2 += self.seq_acc_2.result().numpy()
      acc_top3 += self.seq_acc_3.result().numpy()
      acc_top4 += self.seq_acc_4.result().numpy()
      acc_top5 += self.seq_acc_5.result().numpy()
      totA += self.tot_acc.result().numpy()

      self.seq_acc_1.reset_state()
      self.seq_acc_2.reset_state()
      self.seq_acc_3.reset_state()
      self.seq_acc_4.reset_state()
      self.seq_acc_5.reset_state()
      self.tot_acc.reset_state()


    return {"batch_loss":total_loss/(batch+1),
            "acc_1":acc_top1/(batch+1),
            "acc_2":acc_top2/(batch+1),
            "acc_3":acc_top3/(batch+1),
            "acc_4":acc_top4/(batch+1),
            "acc_5":acc_top5/(batch+1),
            "totA":totA/(batch+1)}


