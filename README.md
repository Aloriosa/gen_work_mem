# Generative working memory in Transformer decoder
This repo contains the implementation of the method for augmenting the Transformer with working memory in decoder.
The method was firstly presented [here](https://doi.org/10.1007/978-3-030-91581-0_34).

# Running the training and evaluation
See the `run.py` script for initial training of the model on TED dataset followed by the fine-tuning on Open Subtitles dataset.
```bash
python run.py
```

## Citation
```
@article{SAGIROVA202216,
title = {Complexity of symbolic representation in working memory of Transformer correlates with the complexity of a task},
journal = {Cognitive Systems Research},
volume = {75},
pages = {16-24},
year = {2022},
issn = {1389-0417},
doi = {https://doi.org/10.1016/j.cogsys.2022.05.002},
url = {https://www.sciencedirect.com/science/article/pii/S1389041722000274},
author = {Alsu Sagirova and Mikhail Burtsev},
keywords = {Neuro-symbolic representation, Transformer, Working memory, Machine translation},
abstract = {Even though Transformers are extensively used for Natural Language Processing tasks, especially for machine translation, they lack an explicit memory to store key concepts of processed texts. This paper explores the properties of the content of symbolic working memory added to the Transformer model decoder. Such working memory enhances the quality of model predictions in machine translation task and works as a neural-symbolic representation of information that is important for the model to make correct translations. The study of memory content revealed that translated text keywords are stored in the working memory, pointing to the relevance of memory content to the processed text. Also, the diversity of tokens and parts of speech stored in memory correlates with the complexity of the corpora for machine translation task.}
}
```
