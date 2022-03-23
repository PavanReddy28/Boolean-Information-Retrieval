# IR-Assignment

- (Notebook)[Final_Notebook.ipynb] - Final Notebook with all the code. RUN THE QUERIES IN THE NOTEBOOK.
- Documentation.py - Python File version of the Notebook.

## Stopword Removal

```
$ python preprocess.py <word1> <word2> <word3>
```

## Normalization - Stemming or Lemmatization
Stemming
```
$ python preprocess.py -stem <word1> <word2> <word3>
```
Lemmatization
```
$ python preprocess.py -lemmatize <word1> <word2> <word3>
```

## Building Index 

```
$ python genIndex.py
```

## Querying
```
$ python main.py
```

## Documentation Generation
HTML Generated can be found at [here](API_Documentation.html).
```
$ pip install pdoc
$ pdoc .\Documentation.py
```

## Team

- [Pavan Kumar Reddy Yannam](https://github.com/PavanReddy28/)
- [BVS Ruthvik]()
- [PV Sri Harsha]()

```Python
/**
 * Pavan Kumar Reddy Yannam 2019A7PS0038H
 * BVS Ruthvik 2019A7PS0017H
 * PV Sri Harsha 2019A2PS1521H
 */
```
