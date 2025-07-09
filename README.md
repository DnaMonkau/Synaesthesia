# A Computational Neuroscientific Analysis of Synaesthesia's Effect on Working Memory
### Authors: D.N.A. (Denise) Monkau, Dr. O. (Olympia) Colizoli \& Dr. S. (Serge) Thill

Graphemes and colours are processed in distinct visual pathways. For most individuals, these two stimuli are separately integrated; however, in grapheme-colour synaesthesia, multimodal perception of the two arises. These synaesthetes receive grapheme stimuli yet perceive colours due to cross-talk between the modalities. We computationally studied how the mechanisms of cross-talk and the rise of multimodal perception affect working memory. We develop an emergent synaesthetic model for the emergence of grapheme-induced colours. These grapheme-colour pairs are then applied in a delayed matching-to-sample task through a neuronal-astrocyte working memory model. Results demonstrated grapheme-specific colour induction as seen in synaesthesia. Furthermore, these grapheme-colour pairs improved working memory retrieval over achromatic grapheme images. These findings provide a case for the emergence of grapheme-colour synaesthesia through cross-talk and improved working memory due to colour information.


## Emergent model:
To run the emergent model use ./ emergent.sh, this returns grapheme-colour combinations as images and the distributions in a for the network and random choice in a plot. This model was inspired by by Shriki et al.

## Working Memory model
The working memory model needs to be ran in matlab with the image source in the same folder as the Emergent model output.
Use Main.m to run the working memory model component. This model was an extension of Gordleeva et al.'s 2D model (to 3D).

Gordleeva, S. Y., Tsybina, Y. A., Krivonosov, M. I., Ivanchenko, M. V., Zaikin, A. A., Kazantsev,
V. B., & Gorban, A. N. (2021). Modeling working memory in a spiking neuron network
accompanied by astrocytes. Frontiers in Cellular Neuroscience, 15 , 631485.
Shriki, O., Sadeh, Y., & Ward, J. (2016). The emergence of synaesthesia in a neuronal network
model via changes in perceptual sensitivity and plasticity. PLoS computational biology, 12 (7),
e1004959
