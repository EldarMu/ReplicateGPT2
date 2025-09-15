# ReplicateGPT2
Following along https://www.youtube.com/watch?v=l8pRSuU81PU after reading a whole lot of papers.


# GPT2AttentionScratchpaper
Working on experiments around interblock attention communication vector (dubbed scratchpaper)

Core idea: The residual stream is both the product on the assembly line and the only mechanism
for blocks to communicate intent with one another, allow any organization. I would like to see
what happens if I give them dedicated "notes" they can pass along. 

implementation: a kind of LSTM whose output gets added to Q*KT before softmax, but will explore various options.

Ideal outcome is lower perplexity at fewer steps, not even hoped for outcome is novel unexpected behavior,
real outcome will be hours debugging and that's ok.

I am exploring different ways that one could try to connect attention heads in different blocks
(not head communication in one blog)