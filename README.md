# Boolean-Information-Retrieval
Partial Fullfillment of CS F469 Information Retrieval

- [Notebook](Final_Notebook.ipynb) - Final Notebook with all the code. RUN THE QUERIES IN THE NOTEBOOK.
- Documentation.py - Python File version of the Notebook.\
- Run the below function in Notebook after running all cells.

## Stopword Removal

```
inputList = ['automation','stemmer','a','brutus','thou','cleopatra','intelligent']
filterTokens(inputList)
```

## Normalization - Stemming or Lemmatization
Stemming
```
inputList = ['automation','stemmer','a','brutus','thou','cleopatra','intelligent']
stemming(inputList)
```
Lemmatization
```
inputList = ['automation','stemmer','a','brutus','thou','cleopatra','intelligent']
lemmatize(inputList)
```

## Building Index 

```
file_index = loadFileInfo(<parent Directory with dataset>)
genWordDict(file_index)
```

## Querying
```
search_and_retrieve(query)
```

## Documentation Generation
HTML Generated can be found at [here](API Documentation.html).
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
