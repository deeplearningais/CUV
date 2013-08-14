#!/bin/sh
rm -f src/cscope.files
rm -f src/cscope.out
rm -f src/tags
rm -f src/cuv/tensor_ops/instantiations/inst*.cu
git clean -f
git checkout CMakeLists.txt
DEB_BUILD_OPTIONS=parallel=12 git-buildpackage --git-debian-branch=debian --git-upstream-branch=master --git-dist=quantal
