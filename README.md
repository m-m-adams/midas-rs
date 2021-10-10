# count-min-sketch
A simple CMS datstructure in rust. CMS can be created with explicit parameters, with tolerance and probability, or by copying an existing CMS to enable combinations.

Each CMS provides methods to insert and retrieve counts, combine with another CMS, and to multiply all entries by a constant.

The python interface provides access to the new_with_probs constructor which builds a CMS with the methods insert, retrieve, clear, and scale. A python CMS can store any object which implements __hash__()
```
>>> import count_min_sketch
>>> cms1 = count_min_sketch.CMS(0.1,0.001,10000)
>>> cms1.insert("test")
1
>>> cms1.insert(54)
1
>>> cms1.insert(54)
2
>>> cms1.retrieve(54)
2
```

A CMS basis hash set can be cloned allowing it to be combined with an existing CMS. This is useful to maintain multiple sets of counts and periodically combine them. This method will throw an error if they did not originate as clones.

```
>>> cms2 = count_min_sketch.clone_cms(cms1)
>>> cms2.combine(cms1)
>>> cms2.retrieve("test")
1
>>> cms2.retrieve(54)
2
>>> 
```

