# count-min-sketch
A simple CMS datstructure in rust. CMS can be created with explicit parameters, with tolerance and probability, or by copying an existing CMS to enable combinations.

Each CMS provides methods to insert and retrieve counts, combine with another CMS, and to multiply all entries by a constant.

a python package can be built with ```maturin develop``` which provides a simpler interface to the new_with_probs (via count_min_sketch.PyCMS) constructor which builds a CMS with the methods insert, retrieve, clear, and scale. 

