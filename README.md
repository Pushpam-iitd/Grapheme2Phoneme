# Grapheme2Phoneme
The sequence-to-sequence model for grapheme to phoneme conversion for better pronounciation of Indian names.

# Graphemes
A grapheme is a letter or a number of letters that represent a sound (phoneme) in a word. Another way to explain it is to say that a grapheme is a letter or letters that spell a sound in a word.
example of a 3 letter grapheme: n-igh-t. The sound /ie/ is represented by the letters ‘i g h’.

# Phoneme
A phoneme is a single "unit" of sound that has meaning in any language.
eg. phonemes of word "computer" is "kəmˈpjuːtə"

# embeddings
the embeddings used is analogous to the word2vec, we have used what can be called a letter2vec embedding. 
Why? Because words are simply discrete states, and we are simply looking for the transitional probabilities between those states: the likelihood that they will co-occur. So letter2vec, or sentence2vec etc. are all possible. e.g there is a high probability that "-g" occurs after "i" to complete the suffix "-ing" as in words playing, eating etc. 

#
