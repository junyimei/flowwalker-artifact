## Dataset

### Download
The raw datasets used in experiments can be downloaded through the links in `get_dataset.sh`. 

If the dataset is downloaded from https://law.di.unimi.it/datasets.php, then you need to install [WebGraph](https://webgraph.di.unimi.it/) first, and run the following command to convert origin graph to ASCII edgelist.
```
java -cp "*" it.unimi.dsi.webgraph.ArcListASCIIGraph  $INPUT $OUTPUT
```

### Converting to CSR
In this paper, we convert all dataset into undirected graphs can delete the isoalted vertices. You can run the following command 
```
g++ -o EdgeListToCSR EdgeListToCSR.cpp
```
to compile. And the following command to convert graphs:
```
EdgeListToCSR $INPUT $OUTPUT #NUMBER_OF_LABELS
```
The input file should be ASCII edgelist. The output files include five files:
   - `$OUTPUT_xadj.bin`: the CSR vertex array
   - `$OUTPUT_edge.bin`: the corresponding edge list
   - `$OUTPUT_weight.bin`: the edge weight
   - `$OUTPUT_label.bin`: the edge label
   - `$OUTPUT.edgelist`: the processed ASCII edgelist

