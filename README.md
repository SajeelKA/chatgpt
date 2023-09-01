# Transformer development and experimentation

Common uses( https://arxiv.org/pdf/2306.07303.pdf ):

LANGUAGE TRANSLATION:
"Transformer2017"
pre-training: None
training: WMT 2014 EnglishGerman,WMT 2014 English-French

TEXT CLASSIFICATION & SEGMENTATION, QUESTION ANSWERING, TEXT SUMMARIZATION:

"T5  (Raffel et al., 2020)"
Pretraining: C4 (Colossal Clean Crawled Corpus)
Training: GLUE and SuperGLUE benchmarks, CNN/Daily Mail abstractive summarization, SQuAD question answering, WMT English to German, French,and Romanian translation

TEXT GENERATION:
"BART (Lewis et al., 2020)"
Pre-training: Corrupting documents, 1M steps on a combination of books and Wikipedia data,news, stories, and web text (Training)
training: SQuAD, MNLI, ELI5,XSum, ConvAI2, CNN/DM,CNN/DailyMail, WMT16 Romanian-English, augmented with back-translation data from Sennrich et al. (2016).

NATURAL LANGUAGE REASONING:
RoBERTa (Clark et al., 2020)
Pretraining: RACE 
Training: RuleTaker

AUTOMATED SYMBOLIC REASONING:
SATFormer(Shi et al., 2022)
Pretraining: No
Training: Synthetic



