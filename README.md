# Midas-RS
This is a rust backed python program implementing the relational, filtering and normal core from MIDAS

usage: midas.py [-h] [-o OUTPUT] [-t {R,r,N,f,n,F}] [-s SCALE] FILE FILE

positional arguments:
  FILE                  file to read input from, source,dest,time on newlines
  FILE                  file to read labels from, each on a new line

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        file to write output to
  -t {R,r,N,f,n,F}, --type {R,r,N,f,n,F}
                        choice of core type. R for relational, F for filtering, N for normal
  -s SCALE, --scale SCALE
                        Factor to decay current time counter by in filtering and relational core

Runtime is approximately 6-20 seconds on the DARPA dataset depending on core option, a 4-5x speedup over pure python.

# count-min-sketch
This library also provides a simple CMS datstructure accessible in rust and python. CMS can be created with explicit parameters, with tolerance and probability, or by copying an existing CMS to enable combinations.

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

