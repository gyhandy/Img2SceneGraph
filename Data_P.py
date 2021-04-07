"""
@Author: Shana
@File: Data_P.py
@Time: 2/18/21 12:15 PM
"""

import json
import os
import argparse
import shutil
import getpass
import json
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def g_word_vec(root, out, Name, dim):
    """
    Function use to replace word index into real feature: the word vector
    :param root: path to nodel_labels.npy & label_dict.npy
    :param out: path to save the result
    :param Name: dataset name
    :param dim: dimension of word vector
    :return:
    """
    O_Natrr = Name + '_node_attributes.txt'
    label_dict = np.load(os.path.join(root, 'label_dict.npy'))
    node_labels = np.load(os.path.join(root, 'node_labels.npy'))

    ##  get word vector for each word in the label_dict
    result = dict()
    unique_idx = np.unique(node_labels)
    max_v = np.max(unique_idx) + 1
    embeds = nn.Embedding(max_v, dim)
    np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
    for idx in range(max_v):
        embed_idx = Variable(torch.LongTensor([idx]))
        result[idx] = embeds(embed_idx)[0].detach().cpu().numpy()

    ## get the corresponding node feature
    with open(os.path.join(out, O_Natrr), 'w+') as f:
        for i in node_labels:
            s = str(list(result[int(i)]))
            s = " " + s[1: -1]
            f.write(s + "\n")
    f.close()


def main(args):

    Info = "custom_data_info.json"
    Pred = 'custom_prediction.json'
    Ann = 'scene_validation_annotations_20170908.json'  # Store file names and it's label id
    Root = args.root
    Out = args.out_dir
    Name = args.name
    method = args.method
    dim = args.dim

    print(Root)
    print(Out)
    print(Name)
    O_Edge = Name + '_A.txt'
    O_Gid = Name + '_graph_indicator.txt'
    O_Gla = Name + '_graph_labels.txt'

    # Num_Nodes = 79

    Num_Edges = 5
    Edges = []
    Node_Labels = []
    Node_Index = []
    Graph_Labels = []

    a = open(os.path.join(Root, Info), 'r')
    info = json.load(a)
    a = open(os.path.join(Root, Pred), 'r')
    pred = json.load(a)
    a = open(os.path.join(Root, Ann), 'r')
    ann = json.load(a)

    Num = len(info['idx_to_files'])
    Node_L = info['ind_to_classes']
    if method == 'a' or method == 'b' or method == 'c':
        if method == 'a':
            percent = float(args.para)
            Num_Edges = int(79 * 79 * percent)
        if method == 'c':
            Num_Edges = int(args.para)

        global_node_index = 1
        for i in range(Num):
            print("======================")
            ## Step 1: Get Graph Label
            filename = info['idx_to_files'][i].split('/')
            filename = filename[len(filename) - 1]
            # now filename is the current image's file name
            # Then from ann, find this image using file name, and get it's label (ann[j]['label_id'])
            for j in range(len(ann)):
                if ann[j]['image_id'] == filename:
                    Graph_Labels.append(ann[j]['label_id'])

            pr_now = pred[str(i)]
            node_temp = []
            Edges_temp = []
            ## Step 2: Get_temp_Edges
            pairs = pr_now['rel_pairs']
            temp = 0

            if method == 'b':
                Num_Edges = 0
                scores = float(args.para)
                ss = pr_now['rel_scores']
                for score in ss:
                    if float(score) >= scores:
                        Num_Edges = Num_Edges + 1
            for j in range(Num_Edges):
                a = pairs[j][0]
                b = pairs[j][1]
                Edges_temp.append([a, b])
                # print(str(a)+" "+str(b))
                if not a in node_temp:
                    node_temp.append(a)
                if not b in node_temp:
                    node_temp.append(b)

            Num_Nodes = len(node_temp)

            ## Step 3: Get Node and real edges
            for j in range(Num_Nodes):
                # node_index.append(j)
                Node_Labels.append(pr_now['bbox_labels'][node_temp[j]])
            for j in range(len(Edges_temp)):
                tp1 = Edges_temp[j][0]
                tp2 = Edges_temp[j][1]
                # print("shh"+str(tp1) + " " + str(tp2))
                for k in range(Num_Nodes):
                    if tp1 == node_temp[k]:
                        a = k
                    if tp2 == node_temp[k]:
                        b = k
                Edges.append([a + global_node_index, b + global_node_index])

            ## Step 4: Indicate Node
            Num_Nodes = len(node_temp)
            for j in range(Num_Nodes):
                Node_Index.append(i + 1)
            global_node_index += Num_Nodes
        # print(global_node_index)


    else:
        global_node_index = 1
        for i in range(Num):
            ## Step 1: Get Graph Label
            filename = info['idx_to_files'][i].split('/')
            filename = filename[len(filename) - 1]
            for j in range(len(ann)):
                if ann[j]['image_id'] == filename:
                    Graph_Labels.append(ann[j]['label_id'])

            pr_now = pred[str(i)]

            ## Step 2: Get Node Index and Node Labels
            Num_Nodes = 0
            if method == 'd':
                Num_Nodes = int(79 * float(args.para))
            if method == 'f':
                Num_Nodes = int(args.para)
            if method == 'e':
                scores = float(args.para)
                ss = pr_now["bbox_scores"]
                for score in ss:
                    if float(score) >= scores:
                        Num_Nodes = Num_Nodes + 1
            for j in range(Num_Nodes):
                Node_Index.append(i + 1)
                Node_Labels.append(pr_now['bbox_labels'][j])

            ## Step 3: Get Edges
            pairs = pr_now['rel_pairs']
            for j in range(len(pairs)):
                a = pairs[j][0]
                b = pairs[j][1]
                if a <= Num_Nodes and b <= Num_Nodes:
                    Edges.append([a + global_node_index, b + global_node_index])



    # Save
    # Edges
    with open(os.path.join(Out, O_Edge), 'w+') as f:
        for i in Edges:
            f.write(str(i[0]) + ", " + str(i[1]) + "\n")
    f.close()
    # Node from which graph
    with open(os.path.join(Out, O_Gid), 'w+') as f:
        for i in Node_Index:
            f.write(str(i) + "\n")
    f.close()
    # Graph labels
    with open(os.path.join(Out, O_Gla), 'w+') as f:
        for i in Graph_Labels:
            f.write(str(i) + "\n")
    f.close()

    a = np.array(Node_Labels)
    b = np.array(Node_L)
    # Node labels
    np.save(os.path.join(Out, 'node_labels.npy'), a)
    # Node dict
    np.save(os.path.join(Out, 'label_dict.npy'), b)

    g_word_vec(Out, Out, Name, dim)  # Please comment this line if you need a global dictionary to generate word vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing for Scene Graph')
    parser.add_argument('--out_dir', type=str, default='',
                        help="Output dir for result")
    parser.add_argument('--root', type=str, default='', help="Scene Result files location")
    parser.add_argument('--name', type=str, default='shana', help="Name of our dataset")
    parser.add_argument('--method', type=str, default='a', help="Methods to select edges and nodes")
    parser.add_argument('--para', type=str, default='0', help="parameters that using in different methods")
    parser.add_argument('--dim', type=int, default=500, help="word_vector dimension")
    args = parser.parse_args()
    main(args)
