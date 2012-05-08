Guide to functions:
3 Types:
  - filterActs --> f = convolve (input, w) 
  - imgActs    --> df/dinput
  - weightActs --> df/dw

if shape = (a,b,c) then matrix \in R^{a*b,c}

a "module" is every "how many steps" a filter should be applied.
e.g. when numModulesX==imgSize, then nModules==imgSize*imgSize


================================================================
================================================================
convFilterActs, localFilterActs
- img:    (nImgChan, nImgPix, nImg)
- filt:   (nFiltChan, nFiltPix, nFilt)            -- conv
-         (nModules, nFiltChan, nFiltPix, nFilt)  -- local
- target: (nFilt, nModules, nImg)

================================================================
convImgActs, localImgActs
- hidActs: (nFilt, nModules, nImg)
- filters: (nFilterColors, filterPixels, nFilters)               if conv
-          (nModules, nFilterColors, filterPixels, nFilters)   otherwise
- targets: (nImageColors, imgPixels, nImages)

================================================================
convWeightActs, localWeightActs
-  images:   (nImgColors, imgPixels, nImages), with stride given
-  hidActs:  (nFilters, numModules, nImages)
-  targets:  (nModules/partialSum, nFilterColors, filterPixels, nFilters)

================================================================
================================================================

convFilterActsSparse, localFilterActsSparse
- images:          (numImgColors, imgPixels, numImages) with stride given
- filters:         (numFilterColors, filterPixels, numFilters)             if conv
-                  (numModules, numFilterColors, filterPixels, numFilters) otherwise
- targets:         (numFilters, numModules, numImages)
- colorIndices:    (numGroups, numFilterColors)

================================================================
convWeightActsSparse, localWeightActsSparse
- images:      (numImgColors, imgPixels, numImages), with stride given
- hidActs:     (numFilters, numModules, numImages)
- targets:     (numModules/partialSum, numFilterColors, filterPixels, numFilters)
 
================================================================
convImgActsSparse, localImgActsSparse
- hidActs:         (numFilters, numModules, numImages)
- filters:         (numFilterColors, filterPixels, numFilters)               if conv
                   (numModules, numFilterColors, filterPixels, numFilters)   otherwise
- targets:         (overSample, numImgColors, imgPixels, numImages)
- colorIndices:    (numGroups, numFilterColors)
  where overSample := (numFilterColors * numGroups) / numImgColors
