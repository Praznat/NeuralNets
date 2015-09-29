#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/mhollen/sift/SDL/lib:/u/mhollen/sift/HyperNEAT/SDL2_image-2.0.0
export LIBRARY_PATH=$LIBRARY_PATH:/u/mhollen/sift/SDL/lib:/u/mhollen/sift/HyperNEAT/SDL2_image-2.0.0
ALE=/u/mhollen/sift/ale/ale

$ALE $@
