# Img2SceneGraph



## Contents
1. [Overview](#Overview)
2. [Install & Usage](#Install-Usage)
3. [Demo](#Demo) 
4. [Citiation](#Citations)



  

  
## Overview
Img2SceneGraph provides a pipeline that transfers images to scene graphs with node attributes. It can generate labeled graph datasets using on various downstream tasks. 
Here is a typical work-flow:

### Step 1: From labeled images to nodes and edges
For each images, we use the pre-trained model from [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) to synthesis the following outputs: 79 bounding boxes (b-boxes) labeled by a single word and over 6,000 relationship pairs (rel-pairs) between b-boxes, both of them are sorted by their corresponding confidence scores.

### Step 2: Select nodes and edges to form Scene graphs
To form a scene graph, we provide multiple methods to select edges and nodes.   
***Select edges first***  
    (a) Select the top *n%* rel-pairs as edges and corresponding b-boxes as nodes.   
    (b) Select rel-pairs with confidence score higher than *k* as edges and corresponding b-boxes as nodes.  
    (c) Select top *m* rel-pairs as the edges and corresponding b-boxes as nodes.  
***Select nodes first***  
    (d) Select the top *n%* b-boxes as nodes and corresponding rel-pairs as edges.   
    (e) Select the b-boxes with confidence score higher than *k* as nodes and corresponding rel-pairs as edges.  
    (f) Select the top *m* b-boxes as nodes and corresponding rel-pairs as edges.  

### Step 3: Load into Pytorch geometric (optional) 
In this part, we use a trick to "disguise" our custom dataset as their official benchmark datasets. You can name your custom dataset as anything you want. For more information and usage please refers to [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#common-benchmark-datasets).




  
  
## Install &Usage  
Tested on:  
    *Ubuntu 18.04*    
    *Cuda 10.1 & 11.0*  
    *Python 3.8*  



### ***Part 1: For Scene-Graph-Benchmark***
**1. Path you need to know:**    
    *$Repo* : path to main repo.  
    *$Check* : path to checkpoints.   
    *$Custom* : path to your images. Only .jpg files are allowed  
    *$Output* : path to save your result.   
    *$Glove* : path to save word vectors.    
    *$Model* : path to your pre-trained model. $Check/causal-motifs-sgdet.  

**2. Install Scene-Graph-Benchmark**  
Follow the instruction [here](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/INSTALL.md) to meet the requirements of Scene-Graph-Benchmark. Note you should clone Scene-Graph-Benchmark into *$Repo*.

**3. Download the dataset**   
Please following [this](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md). We will not really train anything on this dataset. But it's necessary for the procedure.

**4. Download the pretrained model**   
You can find it  [here](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21781947&authkey=AF_EM-rkbMyT3gs)

**5. Before Inference**  
* Extract pre-trained model into _$Model_
* Modify _$Repo/configs/e2e_relation_X_101_32_8_FPN_1x.yaml_ as follow:
> CUSTUM_EVAL: True                  
> CUSTUM_PATH: $Custom

**6. Inference**  

```
cd $Repo
    
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR $Glove MODEL.PRETRAINED_DETECTOR_CKPT $Model OUTPUT_DIR $Model TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH $Custom DETECTED_SGG_DIR $Output
```
* Outputs files:  
    *custom_data_info.json*   
    *custom_prediction.json*
    
    
### ***Part 2: For others***

**1. Clone our repo**  
Clone it to desired location [$Pipeline]
```
cd $pipeline

git clone https://github.com/gyhandy/Img2SceneGraph
```

**2. Form Scene Graphs :** *Data_P.py*  

```
pyhton Data_P.py --out_dir $Out_dir --root $Root --name Name --method Method --para Para -- dim Dim
```
*$Out_dir*: Location to store output files. **You should create this folder in advance.**  
*$Root* : Loacation to store the results from part 1. Aka the files in $Output.  
*Name* : Name of your custom dataset.  
*Method* : Method for selecting nodes and edges. From _a_ to _f_. Details refers to [Overview Step 2](#step-2-select-nodes-and-edges-to-form-scene-graphs).  
*Para* : Parameters for different methods. Note for *n%*, please input *n/100*. For example, 0.1 for 10%.  
*Dim* : Dimension of word_vector.

* Output files:  
    *Name_A.txt* :  Edges forms by node index.    
    *Name_graph_indicator.txt* : Node belongs to which graph  
    *Name_graph_labels.txt* : graph labels.  
    *Name_node_attributes.txt* : node attributes (get from word vector).  
    *node_labels.npy* : node label(word index).  
    *label_dict.npy* : Dictionary for word index and the real English word.  
    
* **Note**  
  
    In Line 54, there's a file named Ann. It's a json file that records the graph(image) label. Since it's hard to predict how different image datasets store their labels, we didn't make it applicable to every dataset.
    
    But the code logic is simple. You just need to modify the file name here and then change the code from Line 94-101 in order to store the graph(image) label into list Graph_Labels. There are a lot of comments so it should be no problem.
    
    

**3. Load into PyTorch Geometric(optional)**  

  Before you call this function in your code,
```
data_set = TUDataset(datapath, name=Name, use_node_attr=True)
```
please make sure that in the *datapath*, you have following folders: (Note the Name is your dataset name)
> /datapath/Name/Name/processed  
> /datapath/Name/Name/raw

Also, you should copy the all the .txt files from Part2.2 into the second folders. Then it should works fine.
To create the folders, you could just run the command once. It will report error but will also create these folders. Or, you could run the following command:

```
mkdir -p /datapath/Name/Name/processed
mkdir -p /datapath/Name/Name/raw
```
Then, move all the .txt files into folder raw by using this command:
```
cd $Out_dir
cp *.txt /datapath/Name/Name/raw
```
We also provides a script *PyGeo.sh* performs the commands above. Note you should modify the path inside it.
```
cd $Pipeline
sh ./PyGeo.sh
```

**4. Generate global dictionary(optional):** *Global_dict.py*    
   If you inference multiple times in part 1, the word dictionary may overlapped. This script could merge multiple sets of *node_labels.npy* and *label_dict.npy*  so that we can train the word vector properly. Details please refers to the comment in this script. You could comment Line 217 in *Data_P.py* before calling *Global_dict.py*.  





## Demo

In this part, we will show the complete work-flow that how we generate the IMG2SCENEGRAPH-ACSG dataset that used in our paper [Graph Autoencoder for graph compression and representation learning](https://openreview.net/forum?id=Bo2LZfaVHNi). Hope this will gives your a clear direction on how to use our pipeline. 



### Step 1: From labeled images to nodes and edges
**1. Path we used**  
    *$Repo* : /home/shana/Scene-Graph-Benchmark.pytorch  
    *$Check*: /home/shana/checkpoints  
    *$Custom*: /home/shana/checkpoints/custom_images  
    *$Output*: /home/shana/checkpoints/results   
    *$Glove*: /home/shana/glove  
    *$Model*: /home/shana/checkpoints/causal-motifs-sgdet  

**2. Command we called**
```
cd /home/shana/Scene-Graph-Benchmark.pytorch
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/shana/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/shana/checkpoints/causal-motifs-sgdet OUTPUT_DIR /home/shana/checkpoints/causal-motifs-sgdet TEST.CUSTUM_EVAL True TEST.CUSTUM_PATH /home/shana/checkpoints/custom_images DETECTED_SGG_DIR /home/shana/checkpoints/results
```

**3. Problem**  
This procedure will generates huge files when you have many images to process and will easily caused OOM error when saving the result. A kind suggestion is that you could drop some information that you don't need by commenting some lines around Line 177 to Line 194 in _$Repo/maskrcnn_benchmark/engine/inference.py_ according to your needs.



### Step 2: Form scene graphs
**1. Path we used**    
    *$Out_dir*: /home/shana/results  
    *$Root* : /home/shana/checkpoints/results  
    *$Pipeline* : /home/shana/Img2SceneGraph  

**2. Command we called**

```
cd /home/shana/Img2SceneGraph
python Data_P.py --out_dir /home/shana/results --root /home/shana/checkpoints/results --name mikasa --method a --para 0.1 --dim 400
```
Here we use method a with n% = 10%, dimension of word_vector = 400.  

### Step 3: Load into Pytorch geometric
**1. Script we used**
```
mkdir -p /home/shana/mikasa/mikasa/processed
mkdir -p /home/shana/mikasa/mikasa/raw
cd /home/shana/results
cp *.txt /home/shana/mikasa/mikasa/raw
```
**2. Command we called**
```
cd /home/shana/Img2SceneGraph
sh ./Pygeo.sh
```
  
  
## Citation
