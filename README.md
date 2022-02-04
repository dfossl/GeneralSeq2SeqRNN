# GeneralSeq2SeqRNN

This repo has one python module GeneralSeq2SeqFactory.py. The main function of 
this module is to impliment a generic Seq2Seq model constructor for exploring
different number of layers and units, bidirection, attention, and cell types 
(RNN, GRU, LSTM).

It was constructed for a project exploring RNN arechetecutres in solving reaction
prediction. Although other functions are available the intended use case is:
```python
from GeneralSeq2SeqFactory import Seq2SeqModelConstructor

units = 512
n_layers = 2
isBidirectional = True
useAttention = True
layer_type = "lstm

model = Seq2SeqModelConstructor(
    units = units,
    input_rxn_processor=rxn_processor,
    output_rxn_processor=product_processor, 
    output_vocab_size=product_processor.vocabulary_size(),
    n_layers=n_layers,
    layer_type=layer_type,
    isBidirectional=isBidirectional,
    useAttention=useAttention)


model.compile(
    optimizer=tf.keras.optimizers.Adam(clipnorm=5),
    loss=MaskedLoss()
)

_ = model.fit(trainDataset, epochs=1)

_ = model.evaluate(testDataset)

h = model.evaluate_dataset(testDataset)
print(h)

```

**input_rxn_processor** and **output_rxn_processor** are expected to be TensorFlow TextVectorization objects. 
But any function that can take a list of input text tokens and output list of integer tokens will
function.

I had the idea to make this it's own repo because playing around with different models quickly may be
useful to some. There is customization missing. Like picking dropout, or using an embedding layer instead
of one hot encoding (one hot is favorable for reaction prediction). So I may make some more edits in the future
including preprocessing helper functions. Also should make it command line runnable with a yaml format for input.
On the todo list!

I have a [Google Colab in this repo](https://github.com/dfossl/IWSS_ReactionPrediction_CoLab) that walks through this code in detail, so that is my pseudo-documentation until I properly comment the rest of these.
