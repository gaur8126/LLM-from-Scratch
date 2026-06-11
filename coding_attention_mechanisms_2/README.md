## Implementing Attention Mechanism


At this point, we know how to prepare the input text for traning LLMs by splitting text into individual word and subword tokens, tokens to token IDs and token IDs to Embeddings + Positional Embeddings. 

Now it's time to look into integral part of the LLM architecture itself, attention mechanism. 

We will implement four different variants of attention mechanisms. 

First of all we need to know what was the probplem in Pre-LLM architectures that do not include attention mechanism. 

Suppose we want to develop a language translation model that translates text from one language into another. But we can't simply translate a text word by word due to the grammatical structure in the source and target language. 

To address this problem it is common to use a deep neural network with two submodules, an encoder and decoder. 

The job of the encoder is to first read in and process the
entire text, and the decoder then produces the translated text.

Before the advent of transformers, recurrent neural networks (RNNs) were the most popular encoder–decoder architecture for language translation. An RNN is a type of neural network where outputs from previous steps are fed as inputs to the current step, making them well-suited for sequential data like text. 

In an encoder–decoder RNN, the input text is fed into the encoder, which processes it sequentially. The encoder updates its hidden state (the internal values at the hidden layers) at each step, trying to capture the entire meaning of the input sentence in the final hidden state. 

The decoder then takes this final hidden state to start generating the translated sentence, one word at a time.

It also updates its hidden state at each step, which is supposed to carry the context necessary for the next-word prediction.

The key idea here is that the encoder part processes the entire input text into a hidden state (memory cell).  The decoder then takes in this hidden state to produce the output. 

The big limitation of encoder–decoder RNNs is that the RNN can’t directly access earlier hidden states from the encoder during the decoding phase. 

Consequently, it relies solely on the current hidden state, which encapsulates all relevant information. This can lead to a loss of contextt, especially in complex sentences where dependen
cies might span long distances.

Although RNNs work fine for translating short sentences, they don’t work well for longer texts as they don’t have direct access to previous words in the input. One major shortcoming in this approach is that the RNN must remember the entire encoded input in a single hidden state before passing it to the decoder. 

Researchers developed the Bahdanau attention mechanism for RNNs in
2014,  which modifies the encoder–decoder RNN such that the decoder can
selectively access different parts of the input sequence at each decoding step. 

Using an attention mechanism, the text-generating decoder part of the network can access all input tokens selectively. This means that some input tokens are more important than others for generating a given output token. (The importance is determined by the attention weights).

Interestingly, only three years later, researchers found that RNN architectures are not required for building deep neural networks for natural language processing and proposed the original transformer architecture including a self-attention mechanism inspired by the Bahdanau attention mechanism.

Self-attention is a mechanism that allows each position in the input sequence to consider the relevency of , or "attend to", all other positions in the same sequence when computing the representation of a sequence. 

Self-attention is a mechanism in transformers used to compute 
more efficient input representations by allowing each position in a sequence to interact with and weigh the importance of all other positions within the same sequence.

In self-attention, the “self” refers to the mechanism’s ability to compute attention weights by relating different positions within a single input sequence.
It assesses and learns the relationships and dependencies between various parts of the input itself,such as words in a sentence or pixels in an image.

The goal of self-attention is to compute a context vector for each input 
element that combines information from all other input elements.

Context vectors play a crucial role in self-attention. Their purpose is to create enriched representations of each element in an input sequence (like a sentence) by incorporating information from all other elements in the sequence.

which need to understand the relationship and relevance of words in a sentence to each other. Later, we will add trainable weights(Q, K, V) that help an LLM learn to construct these context vectors. 

