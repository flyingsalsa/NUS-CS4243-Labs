# CS4243 Lab 1 Report A0273044L

## Gabor Implementation

The Gabor branch is built in two stages: a bank constructor and a pooled-energy extractor. In `make_gabor_bank`, the function first validates the kernel configuration by requiring `kernel_size` to be an odd integer no smaller than 3; this keeps the filter centred and ensures a symmetric sampling grid with a valid local neighbourhood. The filter grid is created with evenly spaced coordinate values on both sides of zero, and each kernel is generated as a Gaussian envelope multiplied by a sine carrier. The orientation is handled by rotating the horizontal coordinate into the desired angle, while the phase offset shifts the carrier so the same texture can be probed at multiple sine phases. The resulting kernel is then demeaned to suppress constant-image responses and normalised to unit Euclidean norm; if the norm is numerically degenerate or non-finite, the function raises an error rather than returning a broken filter. Each tuple in the returned bank contains the numeric kernel and a metadata dictionary with the frequency, orientation, phase, and full config snapshot, preserving the required ordering: frequency outer loop, orientation middle loop, phase inner loop.

The second function, `gabor_energy_maps`, computes one response map per filter by correlating the grayscale image with each kernel. This matches the notebook’s “correlate, do not convolve” requirement and preserves the original image size while measuring how strongly the image matches each oriented Gabor response. The signed responses are converted into a non-negative energy value using either squared magnitude or absolute value, depending on the `energy` argument, and then each energy map is locally averaged with `box_mean` using the configured pooling window. The final output is a stack of H x W maps along a new channel axis, making the bank behave like a multi-channel texture descriptor. A final finite-value check prevents NaN or infinite values from silently entering downstream feature extraction.

### Question 1

Higher frequency responds to fine details, closely spaced stripes, and small textures. Lower frequency responds to broader, more widely spaced patterns.\
Orientation	Changes the direction of patterns the filter detects. Rotating the filter makes different edges or stripes stand out in the map.\
Phase shifts the filter’s light and dark bands within its window. This changes how they line up with the image, so responses may strengthen, weaken, or change sign.\
Pooling size sets the area over which responses are combined. Larger pooling produces a coarser map with less precise locations but greater tolerance to small shifts. Smaller pooling preserves more spatial detail.


![Improves stability](student_files/Report_images/image.png)\
In this image pooling increases stability due to the well defined features\
\
![Erases small Features](student_files/Report_images/image-1.png)\
In this image pooling loses the small details in the stripes

## Edge Implementation

The edge branch contains four reusable operations and one end-to-end pipeline. It first computes gradient information using the supplied Gaussian smoothing and Sobel functions, then applies exact-direction non-maximum suppression, chooses data-dependent thresholds, and links the surviving responses with hysteresis.

`bilinear_sample` samples a two-dimensional image at floating-point coordinates. The input coordinates are broadcast so that one call can process an entire image grid. Coordinates are clamped to the valid image range, and the surrounding integer row and column indices are computed with `floor`, with the next indices clamped at the last row or column. The four surrounding pixel values are combined using weights based on the fractional row and column offsets. This gives continuous interpolation between pixels while ensuring that integer coordinates reproduce the original pixel exactly. Clamping is important for NMS because comparison points near the image boundary can lie outside the image.

`nms_interpolated` performs non-maximum suppression along the true gradient direction rather than assigning each direction to one of four coarse bins. It first validates that magnitude and direction are matching two-dimensional arrays. For every pixel, the direction angle $\theta$ defines the student_files/Report_images/image-coordinate step $dy = \sin(\theta)$ and $dx = \cos(\theta)$. The function uses `bilinear_sample` to obtain the magnitude one step forward and one step backward along this direction. A pixel is retained when its magnitude is greater than or equal to both interpolated neighbours; all other pixels are set to zero. The result is converted to `float32`, and every border pixel is explicitly cleared because border comparisons do not have a complete neighbourhood.

`adaptive_thresholds` estimates low and high thresholds from the strictly positive NMS responses, so zero-valued background pixels do not affect the statistics. The `percentile` method sets the high threshold to the configured percentile. The `robust` method uses the median plus $2.5 \times 1.4826 \times MAD$, where MAD is the median absolute deviation; this is less sensitive to unusually large responses than a mean and standard deviation. The low threshold is the configured ratio multiplied by the high threshold. Unknown threshold methods raise `ValueError`, and an image with no positive responses returns zero thresholds; the later positive-response check still prevents such an image from producing edges.

`hysteresis` converts the strong and weak inputs to independent Boolean masks and validates both their shapes and the requested connectivity. Every strong pixel is used as an initial seed in a queue. With 4-connectivity, the search visits only vertical and horizontal neighbours; with 8-connectivity, diagonal neighbours are included as well. When a weak pixel is adjacent to an accepted pixel, it is marked as an edge and queued for further traversal. Consequently, weak responses are retained only when they form a connected chain to a strong response, while isolated weak responses are discarded.

`detect_edges` combines the complete process. It smooths the grayscale image, obtains Sobel gradients, magnitudes, and directions, then calls `nms_interpolated` and `adaptive_thresholds`. Positive NMS pixels at or above the high threshold become strong seeds, while positive pixels at or above the low threshold but below the high threshold become weak candidates. `hysteresis` links these candidates using the configured connectivity. The function returns the original gradient maps, magnitude and direction, thinned NMS map, both thresholds, and the final Boolean edge mask under stable dictionary keys for downstream feature extraction.

### Question 2

![alt text](student_files/Report_images/image-4.png)    
![alt text](student_files/Report_images/image-2.png)\
The first image is the nearest-8 NMS while the second is the interpolated NMS. It is clear that interpolate NMS detects a diagonal that the nearest NMS rejects (this comes from the top right hand corner of the wood sample image) The approximations change which pixels survive. For example, if the gradient points at 30°, nearest-8 NMS checks along 45°. Those neighbours may have different magnitudes from the values along 30°, so it can wrongly keep a non-maximum or remove a real maximum. Consequently, nearest-direction NMS can produce jagged edges, uneven thickness, or gaps, especially on slanted and curved boundaries. Small angle changes near a rounding boundary can also abruptly change which neighbours are checked.
Interpolated NMS makes those comparisons vary more smoothly with angle, generally giving more consistent thinning and better edge localisation by using the actual gradient angle and estimating the magnitudes on either side by interpolating between nearby pixels. It still approximates the magnitudes between pixels, so it is not exact.

![alt text](student_files/Report_images/image-8.png)
The two edge maps use the same interpolated NMS and thresholds, but different hysteresis connectivity. With 8-connectivity, weak pixels can remain connected to a strong seed through diagonal neighbours; 4-connectivity permits only horizontal and vertical connections. Consequently, some curved boundaries remain more complete under 8-connectivity. (taken from student_files/NMS_compare.py, image from https://forum.image.sc/t/weak-boundaries-extraction/34048) `[AI-CODE]` `[HUMAN-CHECK]`

## Features and Representations

The feature branch builds one aligned local descriptor map with shape $H \times W \times D$. Every feature family preserves the original spatial positions, so channels can be compared pixel by pixel and passed directly to the normality model.

`extract_local_features` first converts the input to `float32`. A grayscale image is expanded into three identical channels so that the colour branch has a consistent shape, while images using 0--255 values are scaled to the 0--1 range. Grayscale is computed once with the supplied `rgb2gray` function and reused by the Gabor and edge branches. The function then appends enabled feature chunks in deterministic order: colour channels, Gabor energy channels, gradient energy, and edge-density channels. Colour channels are named `colour_r`, `colour_g`, and `colour_b`. Each Gabor filter is generated from the configured bank, pooled with the configured window, and given a name containing its frequency, orientation, and phase. Gradient energy is the locally pooled squared Sobel magnitude. The edge branch contributes total edge density and one pooled density map for each orientation bin after directions are mapped into the unsigned range $[0, \pi)$. Edge detection is called at most once even when both gradient and edge features are requested. The function rejects configurations with no enabled families, concatenates all chunks along the channel axis, and can optionally standardise each channel over the spatial dimensions using an epsilon floor on the standard deviation.

The filters and pooling operations preserve height and width, so `features[y, x, i]` and `features[y, x, j]` describe the same image location. A feature channel is therefore an entire response map, rather than a single student_files/Report_images/image-level number.

### Exact channel numbers in Notebook 1

The channel-scale plot uses **zero-based indices**. Its order comes from the sequence of `chunks.append(...)` and `names.extend(...)` operations in `extract_local_features`, followed by `np.concatenate(chunks, axis=-1)`. Notebook 1 enables every family, uses Gabor frequencies `(0.08, 0.16, 0.28)`, four orientations approximately equal to $0^\circ$, $45^\circ$, $90^\circ$, and $135^\circ$, one phase of 0 radians, and eight edge-orientation bins. This produces

$$
D = 3 + (3 \times 4 \times 1) + 1 + 1 + 8 = 25.
$$

| Channel indices | Feature group | Meaning |
|---|---|---|
| 0--2 | RGB colour | Red, green, and blue intensities, in that order, normalised to 0--1. |
| 3--14 | Gabor energy | Twelve locally pooled response-energy maps: three frequencies (0.08, 0.16, and 0.28 cycles/pixel), each with four orientations (approximately 0, 45, 90, and 135 degrees) and phase 0. Channels 3--6, 7--10, and 11--14 correspond to the three frequencies, respectively. |
| 15 | Gradient energy | Local average of squared Sobel magnitude, $\operatorname{box\_mean}(g_x^2 + g_y^2)$, measuring local contrast strength. |
| 16 | Connected-edge density | Local fraction of pixels accepted by connected-edge hysteresis. |
| 17--24 | Orientation-specific connected-edge density | Eight locally pooled edge-density maps for consecutive 22.5-degree gradient-direction bins spanning $[0^\circ, 180^\circ)$. Channel 17 is bin 0 and channel 24 is bin 7. |

The Gabor bank iterates over frequency first, orientation second, and phase last. Consequently, channels 3--6 share the lowest frequency, 7--10 share the middle frequency, and 11--14 share the highest frequency. Their approximate wavelengths are 12.5, 6.25, and 3.57 pixels, respectively. Lower frequencies probe broader patterns, while higher frequencies probe finer details. Each channel correlates the luminance image with its filter, squares the response for the configured `squared` energy mode, and averages it over the configured 9 x 9 pooling window. The orientation specifies the direction in which the sinusoidal carrier varies; its visible bands run perpendicular to that direction.

Gradient energy measures local contrast strength without requiring an edge to survive thresholding or connectivity checks. Edge density measures the local proportion of accepted edge pixels, so it describes edge occupancy rather than gradient amplitude. Notebook 1 uses 4-connectivity for hysteresis and a 9 x 9 window for gradient and edge pooling. The edge-orientation bins use the **gradient direction**, which is perpendicular to the edge tangent. For example, a gradient direction near 0 degrees corresponds to a roughly vertical boundary. Mapping directions modulo $\pi$ groups opposite contrast polarities together. Each accepted edge belongs to one bin, so the eight orientation-density maps sum to the total edge-density map, up to floating-point rounding. Each bin is divided by the full pooling-window area, rather than by the number of edges in that window.

These numerical positions depend on the configuration. Disabling a family or changing the Gabor bank or edge-bin count changes subsequent indices. Notebook 2 currently uses a different Gabor bank, so the 25-channel table above applies to Notebook 1's plot. The returned `names` list is the reliable mapping for any configuration:

```python
features, names = extract_local_features(image, config)
for index, name in enumerate(names):
    print(index, name)
```

### Global image descriptors

`global_pool` converts an aligned $H \times W \times D$ feature map into a one-dimensional image descriptor. It reshapes the spatial positions into rows while preserving channel order, then computes each requested statistic independently for every channel. Supported statistics are mean, standard deviation, and the 10th, 50th, and 90th percentiles. Statistics are evaluated in the exact order supplied by the caller, concatenated into one vector, and returned as `float32`. Unsupported statistic names raise `ValueError`, preventing an accidental change in descriptor layout.

For the 25-channel Notebook 1 configuration, the default `("mean", "std", "p90")` pooling produces 75 values: positions 0--24 are channel means, 25--49 are channel standard deviations, and 50--74 are channel 90th percentiles. Thus local channel 15 (`gradient_energy`) contributes to descriptor positions 15, 40, and 65. These pooled descriptor positions should not be confused with the 25 local channel indices in the scale plot.

`feature_family_indices` creates index lists for the colour, Gabor, gradient, and edge families. It takes the text before the first underscore in each channel name and appends the original channel position to the matching family. All four keys are returned even when a family is absent, which gives downstream code a stable interface for selecting descriptor subsets.


## Normality Model

The normality branch models each material using feature statistics collected from
normal training images only. `fit_normal_model` accepts either one feature map or
a non-empty collection of $H \times W \times D$ maps. It validates that every map
has spatial dimensions, a common feature dimension, and only finite values. The
maps are flattened into patch rows and concatenated, so every spatial location
contributes one $D$-dimensional training sample. When the number of rows exceeds
`max_samples`, exactly that many rows are selected without replacement using the
configured random seed, making the fitting cost bounded and repeatable.

The fitted model is diagonal: for each feature channel it stores the sample mean
and population standard deviation. A provisional mask threshold is estimated
without using defective images. Each normal map is scored using the same
standardised RMS distance used during prediction,
$$
s(x)=\sqrt{\frac{1}{D}\sum_{d=1}^{D}
\left(\frac{x_d-\mu_d}{\sigma_d+\epsilon}\right)^2}.
$$
The score map is then smoothed with the configured odd-sized box-pooling window.
The unreliable outer border is removed before all normal scores are concatenated.
If `mask_threshold` is supplied it is used directly; otherwise the threshold is
the configured percentile of these pooled, interior normal scores. The returned
means and standard deviations are `float32`, while the threshold is stored as a
Python float together with the optional feature names.

`predict_anomaly` first checks that the input is a finite three-dimensional feature
map whose final dimension matches the fitted mean vector. It computes the dense
RMS standardised distance with the model statistics and `epsilon`, then applies
the same score pooling used during fitting. The student_files/Report_images/image-level score is the selected
percentile of only the interior score pixels, so unreliable convolution/pooling
boundaries cannot dominate the image decision. For localization, the full pooled
score map is compared with the model threshold. The configured outer border is
then forced to `False` in the Boolean mask, without changing the score map. The
function returns the `float32` score map, a Python-float image score, and a
Boolean localization mask.

`select_mask_threshold` performs threshold calibration on public validation data.
It requires non-empty score and mask collections of equal length, at least two
candidate thresholds, matching shapes for every pair, and finite score values.
Scores and masks are converted to floating-point and Boolean arrays, flattened,
and concatenated across validation images. Candidate thresholds are sampled at
evenly spaced quantiles from 0.5 to 0.999. For each candidate, pixels with
`score >= threshold` are predicted as anomalous, and pixel F1 is computed as
$$
F_1=\frac{2TP}{\max(1,\,2TP+FP+FN)}.
$$
The threshold with the greatest F1 is selected, with strict improvement ensuring
that ties retain the first candidate and therefore remain deterministic. Border
cropping is intentionally not repeated here because callers are responsible for
providing the validation region to evaluate.


## Results and Analysis Task A-C
## Task A
### Question 3
![alt text](student_files/Report_images/image-9.png) ![alt text](student_files/Report_images/image-10.png)

In the Notebook 1 plot of the selected wood image, the RGB channels (0--2) have means of approximately 0.62, 0.46, and 0.34, considerably larger than every other channel. The largest individual raw scale mean difference is between the red channel and any of the zero-valued edge-orientation channels, although those zeros reflect the absence of detected edges in those directions in this image. The standard-deviation plot also shows substantial differences in variation between channels.

These scale differences arise because the families measure different quantities: colour intensities, filter-response energies, and local edge fractions. Before fitting the regularised linear classifier, StandardScaler centres each globally pooled descriptor component and scales nonconstant components to unit variance using statistics from the normal training images. This improves numerical conditioning and makes regularisation comparable across features. Without scaling, a feature with larger numerical values can influence the classifier’s score using a smaller coefficient and therefore incur a smaller regularisation penalty, even if it is no more informative than a smaller-scale feature.

![alt text](student_files/Report_images/image-27.png)

![alt text](student_files/Report_images/image-11.png)

'count': 248, 'accuracy': 1.0, 'macro_f1': 1.0

 ![alt text](student_files/Report_images/image-12.png)![alt text](student_files/Report_images/image-13.png)

As can be seen from the confusion matrix, no classification errors were detected from the test image set, there were however 5 errors if using the same model on the validation set ![alt text](student_files/Report_images/image-15.png)
These are the feature maps of the 2 with the highest confidence:
![alt text](student_files/Report_images/image-14.png)
![alt text](student_files/Report_images/image-16.png)
[code to retrieve these has been added to notebook 2 from cells 10-12] `[AI-CODE]` `[HUMAN-CHECK]`

### Question 4
The classifier did not become less accurate on defective images in the test split. It achieved 100% accuracy on both the 65 good images and the 183 defective images. On the validation split, accuracy was slightly lower for defective images: ~97.99% (195 / 199), compared with ~98.53% (67 / 68) for good images. This difference of approximately 0.54 percentage points suggests a small decrease in accuracy in this split, but does not establish that defects generally reduce classification accuracy. However Given that only 1 good image was classified wrongly with 68.30% confidenece compared to 2/4 of the defective ones scoring over 90% confidenece incorrectly, I am still inclined to believe classifier would be less accurate on defective images if we had a larger sample size. 

![alt text](student_files/Report_images/image-17.png)
A confident validation error was the image carpet\test\color\008.png, which was incorrectly classified as tile with 93.94% confidence. The source image and response maps show a dark central stain interrupting the regular woven texture. After resizing to 64 × 64, the fine weave becomes less distinct. The low-frequency Gabor maps respond strongly around the stain boundary, while the higher-frequency responses are much weaker on the shared display scale. The gradient-energy map also highlights the stain boundary.

These observations suggest that the stain introduces prominent responses while resizing weakens the representation of the characteristic carpet weave. Global pooling summarises response strengths without preserving their precise spatial arrangement, which may shift the descriptor towards features the classifier associates with tile. This provides a plausible explanation for the confident mistake, although the response maps alone do not prove its cause.

## Task B
### From Notebook 2:
Numbers and figures are taken after setting `MAX_PER_GROUP = None`

| Features | Validation-selected threshold | AUROC | Test pixel IoU | Test pixel F1 |
|---|---|---:|---:|---:|
| Gabor only | 4.8237 | **0.6945** | 0.0909 | 0.1667 |
| Edge only | 2.9100 | 0.6145 | 0.0922 | 0.1688 |
| Combined | 3.1952 | 0.6516 | **0.0942** | **0.1722** |

![alt text](student_files/Report_images/image-26.png)
These resulst line up with the expected behaviour from the features.

- Gabor-only: produces the strongest responses at all three ground-truth defect locations, particularly the middle-left defect. The hotspots spread beyond the labelled regions, showing that it identifies the defects but localises their boundaries imprecisely.
- Edge-only: produces much weaker responses, especially at the two interior defects. For this example, edge and gradient features provide less pronounced separation between defects and background. However, darker colours do not necessarily mean missed detections: its threshold is also relatively low.
- Combined: highlights the same three locations as Gabor-only, but with reduced intensity. This is consistent with averaging strong Gabor deviations with weaker responses from other channels. The figure does not demonstrate a clear localisation improvement over Gabor-only for this image.
All three maps leave most normal wood texture relatively dark. The top defect touches the image border, so part of its response will be excluded from the final predicted masks.

The overall results still stand: combined features achieve the highest aggregate pixel F1, while Gabor-only achieves the highest image AUROC. This particular image illustrates stronger Gabor responses; judging which branch has the best mask overlap requires the thresholded, border-excluded masks, rather than heatmap brightness alone.

| Defect | IoU | F1 |
|---|---:|---:|
| Color | 0.1882 | 0.3168 |
| Cut | 0.0334 | 0.0647 |
| Hole | 0.0672 | 0.1260 |
| Metal contamination | 0.0000 | 0.0000 |
| Thread | 0.0425 | 0.0816 |
| Bent | 0.0000 | 0.0000 |
| Broken | 0.0000 | 0.0000 |
| Glue | 0.1121 | 0.2016 |
| Fold | 0.1195 | 0.2135 |
| Poke | 0.0145 | 0.0286 |
| Crack | 0.0000 | 0.0000 |
| Glue strip | 0.0002 | 0.0005 |
| Gray stroke | 0.0000 | 0.0000 |
| Oil | 0.0000 | 0.0000 |
| Rough | 0.0057 | 0.0113 |
| Combined | 0.1848 | 0.3120 |
| Liquid | 0.3047 | 0.4671 |
| Scratch | 0.4262 | 0.5977 |
| **Overall, including normal images** | **0.0942** | **0.1722** |



![alt text](student_files/Report_images/image-18.png)
Connected edges are most evident in structural defects. Cuts, cracks, threads, scratches, and folds can create elongated boundaries. Holes, punctures, and metal contamination produce more localized contours. Bent and broken grid defects distort or interrupt the material’s existing connected edges. However, visible edges do not guarantee successful localization: resizing to $64\times64$, feature pooling, and score smoothing can weaken thin defects or spread their responses beyond the ground-truth region.

Colour and diffuse texture changes dominate stains and surface changes. Colour defects, oil, liquid, and gray strokes primarily alter intensity or colour over a region. Rough defects change local texture statistics rather than forming one clear contour. Glue and glue strips can combine texture or brightness changes with distinct boundaries, while combined defects contain several cues. Liquid and colour achieve the strongest combined-feature results here. The edge branch can also detect colour defects because it includes gradient energy: a stain can change local contrast even without changing the material’s structure.

### Question 5
![alt text](student_files/Report_images/image-19.png) 
[image generated in notebook 2 cell 21] `[AI-DESIGN]` \
A concrete border failure occurs in wood/test/combined/001.png, shown in the above image. At $64\times64$, 51 of its 94 defect pixels lie inside the excluded five-pixel border as specified by the config. All 51 exceed the mask threshold, but border exclusion forces them to False. For this individual image, F1 falls from 0.3008 before border exclusion to 0.1617 afterward, and IoU falls from 0.1770 to 0.0879. The border safeguard therefore removes genuine detections.

![alt text](student_files/Report_images/image-20.png)

| Change | Image score | Predicted mask |
|---|---|---|
| Increase aggregation percentile | Increases or stays unchanged, emphasizing the highest-scoring pixels | Unchanged |
| Increase mask threshold | Unchanged | Shrinks or stays unchanged; fewer pixels qualify |

Increasing the aggregation percentile makes the image score emphasize the strongest anomaly responses. For this wood example, increasing it from 99 to 99.9 raises the image score from 17.5514 to 22.4837. The mask remains unchanged because its pixels are selected using a separate threshold.

Increasing the mask threshold by 20%, from 3.1952 to 3.8342, shrinks the predicted regions, especially the weaker lower-right region. The predicted pixel count falls from 438 to 388. For this image, F1 improves from 0.1617 to 0.1784, and IoU improves from 0.0879 to 0.0979. This shows that reducing the overextended predictions improves their overlap with the ground truth. The image score remains 17.5514, since the aggregation percentile is unchanged and score map is unrelated to threshold mask. However outside this specific instance, IoU and F1 may improve if mostly false positives are removed but can also remove weaker true detections and fragment connected regions, reducing recall. F1 and IoU may improve or worsen depending on which pixels disappear.

However the top defect remains clipped at the cyan boundary in both masks. Neither change can recover pixels that border exclusion forces to False.

## Task C
Images were resized to 64 × 64 and represented using globally pooled colour, Gabor, gradient-energy, and edge-density features. The selected Gabor frequencies were 0.10 and 0.22, with orientations of 0 and π/2. [As taken from provided FeatureConfig and GaborConfig] A one-vs-rest logistic-regression classifier was fitted on 520 training images, with feature standardisation fitted only on training data. Performance was evaluated on 520 validation images using multi-label targets and an F1 probability threshold of 0.5. The fitted classifier and feature configuration were saved for reuse.

![alt text](student_files/Report_images/image-21.png)
The PCA plot shows substantial overlap between primary attribute classes, with many images concentrated in the same region. Some banded and striped examples extend away from this concentration, although they do not form completely separate clusters. This suggests that the descriptor captures distinctive variation for some examples while representing many different texture words similarly. However, overlap in the first two principal components does not prove that the classes are inseparable in the full feature space.
![alt text](student_files/Report_images/image-22.png)

| Attribute | AP | F1 |
|---|---:|---:|
| banded | 0.470663 | 0.507937 |
| blotchy | 0.175333 | 0.000000 |
| braided | 0.111015 | 0.000000 |
| bumpy | 0.193062 | 0.103448 |
| cracked | 0.133680 | 0.000000 |
| fibrous | 0.137457 | 0.000000 |
| grid | 0.224054 | 0.161290 |
| marbled | 0.236510 | 0.083333 |
| pitted | 0.076864 | 0.000000 |
| porous | 0.106069 | 0.000000 |
| stained | 0.179838 | 0.131148 |
| striped | 0.539917 | 0.571429 |
| woven | 0.281987 | 0.178571 |

Striped and banded achieved the strongest performance, with AP values of 0.5399 and 0.4707 and F1 scores of 0.5714 and 0.5079, respectively. This is consistent with directional repetition being measurable through Gabor responses and edge-orientation statistics. Pitted and porous achieved much lower AP, suggesting that these pooled features capture their appearance poorly. Several attributes have nonzero AP but zero F1: their probabilities can still provide ranking information, but at the 0.5 cutoff the classifier produces no true positives. These results describe the combined descriptor and do not establish which feature family caused each prediction.

![alt text](student_files/Report_images/image-24.png)
![alt text](student_files/Report_images/image-23.png)
The banded, grid, and woven examples receive probabilities of 0.92, 0.79, and 0.52 for their primary attributes, respectively, exceeding the 0.5 prediction threshold. Their broad bands, intersecting lines, and repeated weave remain visually apparent after resizing to 64 × 64. In particular, the grid retains its main structure, suggesting that downscaling preserves useful evidence for this example.
In contrast, the bumpy example receives only 0.05 for bumpy but 0.58 for banded. Its only annotated label is bumpy, so banded is a false positive at the chosen threshold. The raised blocks create repeated, roughly parallel boundaries that may generate directional Gabor and edge responses resembling banding. Global pooling summarises these responses without preserving the arrangement of the blocks or explicitly representing their three-dimensional relief. This provides a plausible explanation for the confusion, although the response features would need further inspection to confirm it.
This is compounded by other images inspected in the bumpy folder showing more rounded protrusions, as illustrated below. This demonstrates variation within the attribute: angular blocks and rounded bumps can share the same human label while producing different colour, frequency, and edge statistics resulting in incorrect predictions.
![alt text](student_files/Report_images/image-25.png)
### Question 6
The handcrafted global descriptor retains summaries of colour, selected spatial frequencies and orientations, local contrast strength, and edge density. Means, standard deviations, and 90th percentiles describe the overall strength and variation of each feature channel. These measurements can capture directional repetition in banded and striped textures, multiple edge directions in grid and woven textures, and colour or contrast variation in blotchy, stained, and marbled textures.
Global pooling discards the locations and spatial relationships of responses. Consequently, it does not explicitly describe how lines intersect in a grid, how fibres interlace in braided or woven textures, or how edges connect into crack networks. Fibrous textures may generate directional responses, but direction alone does not identify fibres. Bumpy, pitted, and porous describe surface relief or holes, whereas this descriptor measures only their indirect appearance through shading, contrast, and edges. Different structures can therefore produce similar pooled statistics.
Resizing to 64 × 64 can also weaken fine texture details, while the limited Gabor frequencies and orientations may miss patterns at other scales or directions. The descriptor therefore represents useful appearance evidence but does not fully represent the structure or meaning of every attribute.
## Results and Analysis Task D

### Experimental setup

The five queries in `task_d_queries.csv` are Q1 (carpet), Q2 (grid), Q3 (leather), Q4 (tile), and Q5 (wood). Checking their paths against `mvtec_textures.csv` confirms that all five belong to the lab's **test** split. The fixed D3 support manifest contains five labelled examples per material, one for each allowed defect, giving 25 support images overall. All 25 belong to the lab's **validation** split, and none is a query image. Although the support paths contain the original MVTec `test` directory, their split assignments in the lab manifest are validation.

All 15 responses were collected on **6 October 2026**, with the model recorded as **Gemini Flash 3.6** (user-reported). The log preserves each prompt, response, attachment path, and source reference. Decoding settings are recorded as unavailable, and the exact provider model snapshot is unverified. The protocol calls for fixed settings, Q1--Q5 order, and a fresh conversation per query and condition; these execution details cannot all be confirmed from the response records.

![Query Images](student_files/Report_images/task_d_queries.png)


### Prompt templates and response records `[AI-DESIGN]`
**D1**

```text
Inspect this surface image. Is anything visibly abnormal? Return JSON with
is_defective, defect_name, confidence, and evidence. Use a Boolean for
is_defective, a short free-form defect name if abnormal or "good" otherwise,
confidence from 0 to 100 in your selected decision and name, and one sentence
of visible evidence.
```

**D2**

```text
Inspect the attached query image. The material is {material}.
Allowed defect labels and definitions:
{one label: definition per line, in manifest order}
Return JSON with is_defective, defect_name, confidence, and evidence.
Use a Boolean for is_defective. defect_name must be exactly one listed label
or "good" if no defect is visible. These two fields must agree. Give confidence
from 0 to 100 in your selected decision and name, and one sentence of visible
evidence.
```

**D3** uses the same D2 text and adds:

```text
The additional images are labelled validation examples in this order:
{support image number: defect label, in manifest order}
Compare the query with all five examples and classify only the query image.
Return the same JSON schema.
```

The completed [prediction log](student_files/task_d/task_d_predictions.csv) contains all 15 rows and preserves the original JSON responses alongside parsed fields. Notebook 4 checks completeness, field consistency, and vocabulary, then joins the frozen predictions to the answer key. It exports [scored predictions](student_files/task_d/analysis/scored_predictions.csv), a [condition summary](student_files/task_d/analysis/condition_summary.csv), [query comparisons](student_files/task_d/analysis/query_comparison.csv), and a [confidence summary](student_files/task_d/analysis/confidence_summary.csv).

### Results and interpretation `[AI-CODE]` `[HUMAN-CHECK]`

| Query | Material | True label | D1 label (confidence) | D2 label (confidence) | D3 label (confidence) |
|---|---|---|---|---|---|
| Q1 | Carpet | color | stain (98) | color (95) | color (95) |
| Q2 | Grid | broken | bent wire (98) | broken (98) | bent (98) |
| Q3 | Leather | cut | cut (98) | cut (95) | cut (95) |
| Q4 | Tile | crack | crack (100) | crack (100) | crack (98) |
| Q5 | Wood | scratch | crack (95) | scratch (95) | good (95) |

| Condition | Correct defect decisions | Exact-label matches | Vocabulary violations | Mean confidence |
|---|---:|---:|---:|---:|
| D1 | 5/5 (100%) | 2/5 (40%), diagnostic only | Not applicable | 97.8 |
| D2 | 5/5 (100%) | 5/5 (100%) | 0/5 | 96.6 |
| D3 | 4/5 (80%) | 3/5 (60%) | 0/5 | 96.2 |

![Task D defect detection and naming outcomes](student_files/task_d/analysis/task_d_outcomes.png)

D1 discovered an abnormality in every query. Its exact-string matches are only a naming diagnostic: it was allowed free-form descriptions and never received the target vocabulary. No synonym mapping is applied. Its three labels outside the material-specific list are therefore **not vocabulary violations** and not judged. D2 and D3 satisfy the supplied vocabulary in all cases, and all 15 binary decisions agree with whether their predicted label is `good`. Valid format and vocabulary do not guarantee a correct prediction.

The D1-to-D2 changes clarify how domain information affected these responses. Q1 changes from `stain` to `color`, while both evidence sentences describe the same dark patch; this is chiefly alignment with the supplied terminology. Q2 changes from `bent wire` to the correct `broken`. Its D1 evidence already mentions both broken and bent regions, while D2 focuses on severed segments and broken ends. Q5 changes from `crack` to `scratch`, with evidence shifting from a vertical crack to an abrasion across the wood grain. D2 adds the material name, allowed names, and definitions together, so their individual effects cannot be separated.

Across the five queries, adding labelled examples in D3 reduced exact-label accuracy from 5/5 in D2 to 3/5, with no improvements. Q1, Q3, and Q4 remained correct, while Q2 changed from broken to bent, with evidence emphasising warped strands. Q5 changed from scratch to good, interpreting the visible lines as natural wood fissures and explicitly denying scratches, making it the only missed defect. In this run, defect names and definitions alone were therefore more effective than adding examples. One possible explanation is the difficulty of comparing six images within a single query: the test image and five support examples. Although each image file was named, the model still had to distinguish the query from the references, match examples to labels, and identify the relevant visual differences in the defects from the provided images.

Differences in support appearance could influence these interpretations, but the responses alone do not establish that mechanism or show that examples generally reduce accuracy especially with such low sample size.

### Confidence and correctness

![Task D confidence by condition and prediction correctness](student_files/task_d/analysis/task_d_confidence.png)

The plot groups D1 by binary correctness and D2/D3 by exact-label correctness. The prompts request confidence in both decision and name, so this is a descriptive comparison, not a calibration test. D1 averages 97.8 with five correct anomaly decisions; D2's five correct labels average 96.6. In D3, correct labels average 96.0, whereas the two incorrect labels average 96.5. Q2 remains at 98 when its correct D2 label becomes incorrect in D3, and Q5 remains at 95 when its defect is missed. Confidence therefore did not flag the two D3 errors. Empty error groups have undefined means rather than zero.

### Limitations

With five queries and one image per material, one changed answer shifts accuracy by 20 percentage points. All five true labels are defects, so binary accuracy here equals defect recall; specificity and false-positive behaviour cannot be assessed. The support set also contains only defects. This is a controlled qualitative probe, with D2 performing best on these particular queries.

There is only one response per query and condition. Repeated trials would be needed to measure nondeterminism and distinguish a stable condition effect from run-to-run variation. Prompt wording, attachment order, resolution, and support appearance may affect predictions, but no separate sensitivity experiment was performed. Unrecorded decoding settings and an unpinned provider snapshot further limit reproducibility; future model updates could change the answers.

MVTec is public, so possible prior exposure during VLM training cannot be ruled out by this lab's split separation. Self-reported confidence is also not a calibrated probability: these five observations per condition are insufficient to establish calibration or general VLM superiority. 

## Results and Analysis Custom Photos 

### Collection and frozen inference protocol [HUMAN-DATA] 

The [personal-photo manifest](manifests/personal_annotations.csv) contains **15 photographs: five carpet, four tile, and six wood**, covering three MVTec material classes. Ten are recorded with anomalies and five as `good`. The recorded conditions include frontal and angled views, close-up and mid-distance framing, uneven illumination, brightness gradients, and specular reflections. The files contain no EXIF as required by the Lab instructions.

Both annotator columns contain three or four terms per image.  The [annotation audit](student_files/personal_photo_analysis/annotation_audit.csv) preserves counts, unsupported terms, source dimensions, and image hashes. (As I am doing this Lab individually the second set of annotations was taken from my friends, splitting it into sets of 4 per person so the criteria for judgement of these annotations may be inconsistent)

Notebook 4's setup cell loads the saved Task B and Task C models, and its sixth cell applies them without retraining or adaptation. Both pipelines resize inputs to **64 x 64** and use their saved feature configurations and fitted scaling. Each passport reports the predicted material and confidence, five ranked DTD attributes with scores, an anomaly score, heatmap, predicted mask, visible-evidence description, and uncertainty statement. The source photograph is shown at higher resolution than inference. Attribute scores come from independent one-vs-rest classifiers and need not sum to one, listing the top five does not mean all five exceed the classification threshold.

The passport uses the **predicted material to select the normality model**. Anomaly scores are the 99th percentile of the smoothed scores in the retained interior. Masks use the frozen Task B validation-selected pixel threshold, **3.1952**, and exclude a five-pixel border. Reported mask percentages therefore use the **54 x 54 interior** as their denominator. The CSV additionally records fractions of the full image. This pixel threshold is not a separately calibrated image-level defect threshold. A second pass through the normality model selected by the recorded material is exported only as a routing diagnostic; it does not replace passport predictions. `[AI-LANGUAGE]` `[AI-CODE]`

### Passport results and material generalisation

The [complete passport gallery](student_files/personal_photo_analysis/index.html), [prediction CSV](student_files/personal_photo_analysis/predictions.csv), and [passport JSON](student_files/personal_photo_analysis/passports.json) contain all outputs. [Analysis metadata](student_files/personal_photo_analysis/analysis_metadata.json) records model and manifest hashes, configurations, vocabulary, and limitations. Each image identifier below links to its full passport; confidence values are rounded and uncalibrated on these photographs.

| Passport | Recorded material / condition | Predicted material | Confidence | Anomaly score | Mask interior (%) |
|---|---|---|---:|---:|---:|
| [carpet1](student_files/personal_photo_analysis/passports/carpet1.png) | carpet / color | tile | 1.0000 | 2.263 | 0.0 |
| [carpet2](student_files/personal_photo_analysis/passports/carpet2.png) | carpet / combined | tile | 1.0000 | 5.542 | 7.8 |
| [carpet3](student_files/personal_photo_analysis/passports/carpet3.png) | carpet / color | wood | 0.9996 | 8.890 | 6.0 |
| [carpet4](student_files/personal_photo_analysis/passports/carpet4.png) | carpet / color | wood | 0.9712 | 11.403 | 35.3 |
| [carpet5](student_files/personal_photo_analysis/passports/carpet5.png) | carpet / good | grid | 0.9981 | 1.342 | 0.0 |
| [tile1](student_files/personal_photo_analysis/passports/tile1.png) | tile / good | grid | 0.5206 | 1.567 | 0.0 |
| [tile2](student_files/personal_photo_analysis/passports/tile2.png) | tile / color | wood | 0.9281 | 37.940 | 20.1 |
| [tile3](student_files/personal_photo_analysis/passports/tile3.png) | tile / liquid | wood | 1.0000 | 11.187 | 23.0 |
| [tile4](student_files/personal_photo_analysis/passports/tile4.png) | tile / scratch | wood | 0.9958 | 22.905 | 40.1 |
| [wood1](student_files/personal_photo_analysis/passports/wood1.png) | wood / good | leather | 0.6221 | 118.533 | 100.0 |
| [wood2](student_files/personal_photo_analysis/passports/wood2.png) | wood / good | wood | 0.9995 | 11.877 | 31.5 |
| [wood3](student_files/personal_photo_analysis/passports/wood3.png) | wood / liquid | tile | 1.0000 | 2.487 | 0.0 |
| [wood4](student_files/personal_photo_analysis/passports/wood4.png) | wood / color | wood | 0.9180 | 4.333 | 1.6 |
| [wood5](student_files/personal_photo_analysis/passports/wood5.png) | wood / color | wood | 0.9960 | 7.851 | 20.5 |
| [wood6](student_files/personal_photo_analysis/passports/wood6.png) | wood / good | wood | 0.9933 | 3.412 | 0.0 |

Material accuracy against the recorded labels is **4/15 (26.7%)**: 0/5 carpet, 0/4 tile, and 4/6 wood. This contrasts sharply with the earlier MVTec test result. Personal surfaces, framing, illumination, and scale differ from the training distribution; these results show poor transfer on this collection, without isolating which difference causes the failures. Several errors have confidence close to one, so confidence alone does not identify reliable predictions.

Routing can substantially alter the anomaly result. For recorded-good wood1, the predicted leather model gives **118.533** and flags the entire interior; the recorded-wood diagnostic gives **5.022** and 13.1%. For wood3, predicted tile gives **2.487** with an empty mask, while the recorded-wood diagnostic gives **58.877** and flags the entire interior. Tile2 shows the reverse direction: predicted wood gives **37.940** and 20.1%, whereas recorded tile gives **1.819** and an empty mask. Thus an extreme score can reflect disagreement with the selected material model, and a low score can hide visible abnormalities. Correcting the route alone also does not guarantee precise localisation.
`[AI-LANGUAGE]` `[AI-CODE]`
### Attribute agreement and visual evidence `[AI-CODE]` `[AI-DESIGN]` `[HUMAN-CHECK]`

![Material confidence versus anomaly score and human attribute agreement](student_files/personal_photo_analysis/confidence_and_attributes.png)

Set Jaccard overlap, intersection divided by union, is averaged equally across photographs. The two human sets agree exactly on **13/15** images, giving mean Jaccard **0.9667**. Human-model overlap compares the model's top-five set separately with each annotator, then averages the two comparisons. Its mean is **0.1327** using all supplied human terms and **0.1551** after restricting human terms to the model's 13-term vocabulary. Only **62/94 annotation-term occurrences (66.0%)** are supported by that vocabulary. Terms such as `lined`, `flecked`, `matted`, `interlaced`, `chequered`, and `smeared` are retained in the records, rather than silently replaced. The small improvement after vocabulary restriction indicates that unavailable terms explain only part of the disagreement. Top-five versus three-to-four-term sets also affects achievable overlap, so these values are descriptive agreement measures rather than calibrated classification accuracy.

The carpet5 material error still produces plausible attributes: **grid 0.265, fibrous 0.153, and woven 0.141** match its visible basket weave. The classifier calling it grid illustrates how shared texture structure can be meaningful at the attribute level while confusing the material label.

![Carpet2 passport: central dark glossy patch and displaced strongest anomaly response](student_files/personal_photo_analysis/passports/carpet2.png)

In carpet2, a dark glossy patch interrupts the central woven pattern. `Stained` (0.407), `striped` (0.400), and `fibrous` (0.196) are visually plausible. However, the strongest heatmap response is above the central patch, around the upper red patterned region, and the mask includes several disconnected areas. Its 7.8% coverage therefore does not establish successful localisation of the visible patch.

![Wood3 passport: visible droplets with an empty predicted mask](student_files/personal_photo_analysis/passports/wood3.png)

Wood3 contains clearly visible droplets across horizontal grain and a broad reflection. Its empty passport mask misses these visible structures after routing through tile. The recorded-material diagnostic flags 100% of the interior, which exposes sensitivity to routing but still cannot isolate the droplets.

![Wood2 passport: anomaly response aligned with a bright reflection](student_files/personal_photo_analysis/passports/wood2.png)

Wood2 is recorded as good and correctly classified as wood. Nevertheless, its strongest response aligns with the bright region left of centre, and 31.5% of the interior is flagged. This is evidence that appearance changes associated with illumination can trigger the normality model even when material classification succeeds.

### Highlights & Reflections relation to anomaly scores `[AI-CODE]` `[HUMAN-CHECK]`

![Anomaly scores grouped by recorded lighting under predicted and recorded material routing](student_files/personal_photo_analysis/lighting_and_routing.png)

The focused comparison groups illumination descriptions containing `highlight`, `specular`, or `reflective` into **eight photographs**, with seven in the remaining group. Median passport scores are **9.519 versus 3.412**. With recorded-material routing, the medians are **6.092 versus 3.412**. Both comparisons associate the highlight/reflection group with a higher median in this sample.

However, the primary means reverse this ordering: **13.015 versus 21.058**, driven by wood1's score of 118.533 in the other-lighting group. Recorded-material means are **15.913 versus 5.749**. These [lighting summaries](student_files/personal_photo_analysis/lighting_summary.csv) show why the conclusion depends on both routing and summary statistic. Anomaly distances are relative to each material's normality model and are not calibrated across classes, so even the recorded-material comparison mixes different reference distributions. The groups differ in material, defects, viewpoint, and scale; “other lighting” also includes strong brightness gradients. This is an observational association, not a controlled illumination effect or a general claim about lighting sensitivity.

There are no personal ground-truth masks, so pixel IoU, F1, or localisation accuracy cannot be established. Mask coverage describes model behaviour, not defect extent. The small collection, uncertain instance relationships, 64-pixel resizing, border exclusion, and unverified annotation provenance limit stronger claims. The passports make these failures inspectable while preserving the frozen model outputs.


## Report Traceability Table
The table cross-references the implementation explanations and analysis in this report to the supporting notebook outputs. Cell numbers are the one-based positions of all cells, including Markdown cells, rather than execution counts. [AI-LANGUAGE]

| Student file or task | Functions or work covered | Report discussion | Supporting notebook figures, tables, or numerical results |
|---|---|---|---|
| [gabor_branch.py](texturelab/gabor_branch.py) | `make_gabor_bank`, `gabor_energy_maps` | [Gabor Implementation](#gabor-implementation); [Question 1](#question-1) on frequency, orientation, phase, and pooling. | [Notebook 1](notebooks/01_components_and_visualisation.ipynb), cells 8, 10, and 13: kernel-bank visualisation and kernel-mean check, pooled energy maps, and mean response by frequency/orientation/phase. |
| [edge_branch.py](texturelab/edge_branch.py) | `bilinear_sample`, `nms_interpolated`, `adaptive_thresholds`, `hysteresis`, `detect_edges` | [Edge Implementation](#edge-implementation); [Question 2](#question-2) on nearest-direction versus interpolated NMS and hysteresis connectivity. | [Notebook 1](notebooks/01_components_and_visualisation.ipynb), cells 15 and 17: smoothing/gradient/NMS/final-edge panels, NMS difference map, and retained-response histogram. The additional [4/8-connectivity figure](student_files/Report_images/image-8.png) is discussed in Question 2. |
| [features.py](texturelab/features.py) | `extract_local_features`, `global_pool`, `feature_family_indices` | [Features and Representations](#features-and-representations), including [channel-family mapping](#exact-channel-numbers-in-notebook-1) and [global descriptors](#global-image-descriptors); [Questions 3](#question-3), [4](#question-4), and [6](#question-6). | [Notebook 1](notebooks/01_components_and_visualisation.ipynb), cells 19 and 21: feature-shape, channel-name and finite-value audit, feature maps, and channel-scale plots. [Notebook 2](notebooks/02_tasks_and_evaluation.ipynb), cells 9–12: material metrics, confusion/calibration/confidence plots, and validation-error response maps. [Notebook 3](notebooks/03_DTD_attribute_prediction.ipynb), cell 8: descriptor/target dimensions and training-label counts. |
| [normality.py](texturelab/normality.py) | `fit_normal_model`, `predict_anomaly`, `select_mask_threshold` | [Normality Model](#normality-model); [Task B](#task-b), including feature-family ablation and per-defect results; [Question 5](#question-5) on border exclusion, aggregation percentile, and mask threshold. | [Notebook 2](notebooks/02_tasks_and_evaluation.ipynb), cells 16 and 18: validation-selected threshold and score-distribution diagnostic; cells 20, 21, and 23: test metrics, border/threshold numerical comparison, and anomaly-map/mask examples; cell 26: separately calibrated Gabor-only, edge-only, and combined test results. |
| Notebook 2 — Feature-family ablation | Configure Gabor-only, edge-only, and combined features; fit each branch on normal training images; select its threshold on validation data; evaluate on test data; compare the same image's anomaly maps using shared colour limits. | [Task B](#task-b): ablation results table and interpretation of the three feature families. | [Notebook 2](notebooks/02_tasks_and_evaluation.ipynb), cells 25–26: `ablation_configs`, `ablation_reports`, and `ablation_example_maps`; validation-selected thresholds, image AUROC, pixel IoU/F1, and the [shared-scale anomaly-map comparison](student_files/Report_images/image-26.png) for `wood/test/combined/001.png`. |
| Notebook 2 — Border-exclusion box overlays and sensitivity analysis | Add cyan `Rectangle` overlays marking the retained interior; compare masks before and after border exclusion and after a 20% threshold increase; compute pixel metrics, mask sizes, and image scores at the 99th and 99.9th percentiles. | [Question 5](#question-5): loss of genuine detections at the excluded border and the separate effects of mask threshold and image-score percentile. | [Notebook 2](notebooks/02_tasks_and_evaluation.ipynb), cell 21: input, score, before/after masks, higher-threshold mask, and ground-truth panels with cyan boxes aligned to pixel boundaries. The [border comparison](student_files/Report_images/image-19.png) and [sensitivity results](student_files/Report_images/image-20.png) support the reported IoU/F1, predicted-pixel counts, and image-score comparisons. |
| Notebook 3 — Task C | Handcrafted descriptors, PCA, one-vs-rest fitting and model export, per-attribute evaluation, and qualitative inspection. | [Task C](#task-c); [Question 6](#question-6) on information retained and lost by the descriptor. | [Notebook 3](notebooks/03_DTD_attribute_prediction.ipynb), cells 8, 10, 12, and 15: descriptor audit, PCA plot, attribute evaluation/model-save output, and per-attribute AP/F1 table and chart; cells 17–18: qualitative examples and probability heatmap. |
| Notebook 4 — Task D | D1–D3 prompts and frozen VLM responses; defect decisions, label accuracy, vocabulary compliance, and confidence analysis. | [Task D](#results-and-analysis-task-d), including [prompt and response records](#prompt-templates-and-response-records), [results](#results-and-interpretation), [confidence](#confidence-and-correctness), and [limitations](#limitations). | [Notebook 4](notebooks/04_personal_photo_and_vlm_analysis.ipynb), cell 17: compiled 15-response table; cells 20, 22, and 24: response audit, condition/query result tables, outcome and confidence plots, and exported analysis files. |
| Notebook 4 — Personal photographs | Frozen-model Texture Passports, material-routing diagnostics, human attribute agreement, and lighting comparison. | [Collection and frozen inference protocol](#collection-and-frozen-inference-protocol); [passport results](#passport-results-and-material-generalisation); [attribute agreement](#attribute-agreement-and-visual-evidence); [lighting analysis](#do-highlights-and-reflections-accompany-higher-anomaly-scores). | [Notebook 4](notebooks/04_personal_photo_and_vlm_analysis.ipynb), cell 6: passport generation, prediction summaries, attribute-agreement and lighting plots, and exported numerical results. The [passport gallery](student_files/personal_photo_analysis/index.html) contains the individual image, heatmap, mask, scores, description, and uncertainty statement for each photograph. |

## AI-use disclosure table

Cell numbers below count all notebook cells, including Markdown cells, starting from 1. The entries consolidate the report and notebook markers and retain the ablation and DTD implementation assistance already declared in this report. Tool/model names and student verification descriptions are retained from the existing disclosure; the checks described are student-reported. [AI-LANGUAGE]

| Tool/model | Exact affected artefacts | Markers | Contribution | `[HUMAN-CHECK]` |
|---|---|---|---|---|
| ChatGPT-6.1 Sol | [Question 2](#question-2), the 4/8-connectivity comparison paragraph and [edge-map figure](student_files/Report_images/image-8.png). | `[AI-CODE]`  | Assisted with code comparing the final edge maps under 4- and 8-connected hysteresis. | Inspected the generated outputs; the report marks this comparison as human-checked. |
| ChatGPT-6.1 Sol | [Notebook 2](notebooks/02_tasks_and_evaluation.ipynb), cells 10–12; [Question 3](#question-3), the tagged note identifying these added cells and the associated confidence/error figures. Cell 12 also contains the local markers. | `[AI-CODE]`  | Assisted with confidence plotting, identifying and ranking validation errors, and displaying their source images and feature maps. | Checked numerical outputs against the manifest and inspected the generated results. |
| ChatGPT-6.1 Sol | [Notebook 2](notebooks/02_tasks_and_evaluation.ipynb) [Task B](#task-b), the ablation table and [shared-scale anomaly-map comparison](student_files/Report_images/image-26.png). | `[AI-CODE]` | Assisted with the Gabor-only, edge-only, and combined validation-to-test workflow, metrics, and comparison plots. | Checked split membership, calculations, and saved outputs|
| ChatGPT-6.1 Sol | [Notebook 2](notebooks/02_tasks_and_evaluation.ipynb), cell 21; [Question 5](#question-5), the tagged border-exclusion figure note and border/threshold comparison. | `[AI-DESIGN]` | Proposed using one image to visualise border exclusion and threshold effects, including cyan boxes marking the retained interior and before/after mask comparisons. | N/A |
| ChatGPT-6.1 Sol | [Notebook 3](notebooks/03_DTD_attribute_prediction.ipynb), cells 8, 10, 12, and 15 for the previously declared DTD implementation; cell 18 for the locally tagged probability heatmap; [Task C](#task-c), the corresponding metrics and [heatmap](student_files/Report_images/image-23.png). | `[AI-CODE]` | Assisted with descriptor extraction, PCA, classifier fitting/export, evaluation, and probability-heatmap plotting. The broader implementation assistance was declared in the existing table; only cell 18 currently carries local markers. | Checked row alignment, calculations, and outputs against the manifest; inspected the results.|
| ChatGPT-6.1 Sol | Report, Task D — “Prompt templates and response records”; exact D1–D3 prompts preserved in [D1 records](student_files/task_d/outputs/D1/JSONs.md), [D2 records](student_files/task_d/outputs/D2/JSONs.md), [D3 records](student_files/task_d/outputs/D3/JSONs.md), and the [prediction log](student_files/task_d/task_d_predictions.csv). | `[AI-DESIGN]` | Helped formulate the prompts and comparison used for the three Task D information conditions. | Checked suitability and carried out the proposed experiment. |
| ChatGPT-6.1 Sol | [Notebook 4](notebooks/04_personal_photo_and_vlm_analysis.ipynb), cells 20, 22, and 24; report, Task D — “Results and interpretation” and associated confidence results; exported files in `student_files/task_d/analysis/`. | `[AI-CODE]` | Assisted with loading and auditing the frozen responses, scoring decisions and labels, checking vocabulary, summarising confidence, and exporting tables and plots. | Checked split membership, row alignment, numerical calculations, and saved outputs against the manifest and response records. |
| ChatGPT-6.1 Sol | [Notebook 4](notebooks/04_personal_photo_and_vlm_analysis.ipynb), cell 6; report, Custom Photos — “Collection and frozen inference protocol”, “Passport results and material generalisation”, “Attribute agreement and visual evidence”, and “Highlights & Reflections relation to anomaly scores”; outputs in `student_files/personal_photo_analysis/`. | `[AI-CODE]` `[AI-DESIGN]`  | Assisted with frozen-model inference, material-routing diagnostics, passport generation, human/model attribute agreement, and the lighting comparison, including its analysis code and plots. | Checked outputs against the personal-photo manifest, inspected results, and checked calculations, saved files and code against provided files (pipelines.py visualize.py etc) |
| ChatGPT-6.1 Sol | Report, Custom Photos — the tagged routing/threshold paragraph in “Collection and frozen inference protocol” and the tagged “Passport results and material generalisation” discussion; descriptions in [passports.json](student_files/personal_photo_analysis/passports.json) and the [passport gallery](student_files/personal_photo_analysis/index.html). | `[AI-LANGUAGE]` `[AI-CODE]` | Converted structured model outputs and measurements into explanatory report prose and passport descriptions; associated result-generation code is covered in the preceding row. | Compared factual claims with figures, code, CSVs, and source images, then rewrote or edited unsuitable wording.|
| Gemini Flash 3.6 (as recorded) | [Notebook 4](notebooks/04_personal_photo_and_vlm_analysis.ipynb), cell 17; all 15 D1–D3 responses in `student_files/task_d/outputs/`; [task_d_predictions.csv](student_files/task_d/task_d_predictions.csv); report, Task D prediction table. | `[AI-VISION]` | Produced defect decisions, defect names, confidence scores, and visible-evidence sentences for the five queries under three conditions. | N/A |
| Student photographer and human annotators | [personal_annotations.csv](manifests/personal_annotations.csv); the 15 photographs in `student_files/Personal_Photos/`; report, Custom Photos — “Collection and frozen inference protocol”; photographs and human annotations used in the passports and agreement analysis. | `[HUMAN-DATA]` | Captured/selected the personal photographs and supplied human texture annotations. The report records that friends contributed the second annotation set. | N/A |
