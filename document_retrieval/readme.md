Coarse document retrieval consists of wikipedia retrieval and TF-IDF retrieval.

## Wikipedia Retrieval ##
For wikipedia retrieval, firstly download wikipedia database from [baidu disk](https://pan.baidu.com/s/1Q9uNTF8YaK6rrD1tZwVbuQ?pwd=7rr4). 

Then download constituency parser from [baidu disk](https://pan.baidu.com/s/1NqAbaGnW41aDZz9W18ad4Q?pwd=z7vg). In our experiments, we utilize the 2018 version. You can also choose the newest 2020 version from [Allennlp](https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz).

Then set appropriate file path parameters, and run

```
python wikipedia_retrieval.py
```

## TF-IDF Retrieval ##
For TF-IDF retrieval, download TF-IDF file from [baidu disk](https://pan.baidu.com/s/1SZ5KkpYMjQCZcOu7p9BGJA?pwd=xmd3).

Then set appropriate file path parameters, and run

```
python tf_idf_retrieval.py
```

## Merge ##
For each sample, merge wikipedia retrieval and tf-idf retrieval results as coarse document retrieval results.
