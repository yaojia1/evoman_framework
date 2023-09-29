#!/bin/bash

for i in {1..8}
do 
	python optimization_dummy.py -m data -n $i
done

exit 0
	
