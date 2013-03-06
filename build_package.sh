#!/bin/sh
rm -f src/cscope.files
rm -f src/cscope.out
rm -f src/tags
rm -f src/cuv/tensor_ops/instantiations/inst*.cu
git-buildpackage --git-upstream-branch=master --git-debian-branch=debian --git-upstream-tree=branch --git-ignore-new
