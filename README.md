## SimpSim 参数设置

#### 结构相似度计算参数

计算结点到第n层的度序列，参数为n，当前设置为**2**

```python
# main.py

generate_degreeSeq_with_bfs(G, n)	# n为设置的层数，当前为2
```



衰减因子α，结构距离计算时用到，当前设置为**0.5**

```python
# algorithm_dis.py

alpha = 0.5		# line 144
```



#### 多层网络随机游走参数

网络层数为2时，需设置游走到每层结点的概率值，当前设为**0.3**

```python
# algorithm_walk.py

prob_move = 0.3		# line 364
```

运行多层网络随机游走代码，需将单层网络随机游走代码注释掉

```python
# algorithm_walk.py

# line 372 —— 391
for l in range(num_graphs):
    if r < (l + 1) * prob_move:
        # 跳转到l层
        layer = l
        # 如果仍在当前层, 则添加新结点
        if current_layer == layer:
            v = chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer)
            path.append(v)
            break

if r < prob_move:
	layer = 0
	if current_layer == layer:
		v = chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer)
		path.append(v)
        
else:
	layer = 1
    if current_layer == layer:
    	v = chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer)
		path.append(v)

# 注释掉line 394 - 395
```



#### 单层网络随机游走参数

网络只有一层，游走只在这一层网络中游走

```python
# algorithm_walk.py

# line 394 —— 395
v = chooseNeighbor(v, graphs, alias_method_j, alias_method_q, layer)
path.append(v)

# 注释掉line 372 —— 391
```



> 注：在main.py中注释相应的模块，可根据需求仅运行某个模块的代码，例如可将计算好的相似度文件直接放入 `save` 文件夹中，并在main.py输入相应的文件名列表，即可运行。
>
> 在构建的多层网络中，chemical对应 layer - 0，genes对应 layer - 1