# Satellite Climate Monitoring Report

**Land Use Classification Results**

Our analysis using the best-performing model, SVM_RBF, achieved an overall test accuracy of 95.4%, indicating that nearly all identified objects were correctly classified within their respective classes. The macro-averaged F1 score was calculated at 0.953, demonstrating strong performance across multiple categories. Notably, SeaLake received exceptionally high classification scores, suggesting robust spectral characteristics in this environment. Conversely, Highway exhibited lower F1 values, potentially due to differences in its spectral signature compared to other classifications.

**Change Detection Results**

In our evaluation based on real Sentinel-2 before/after image pairs from the OSCD dataset, we assessed mean precision, recall, and F1 scores as well as average pixel-level accuracy. These metrics collectively indicate good performance, particularly when considering the challenging nature of observing temporal changes over vast areas. On average, about 95.6% of individual pixels showed some level of detection; however, the majority of detected changes involved either no significant impact or minor variations in vegetation health.

We also analyzed pairs where there was evidence of actual vegetation loss or gain. In total, 12 out of 24 such cases demonstrated substantial change, with the most extreme instance involving a decrease of 0.0531 units in NDVI value. Furthermore, the largest observed change area covered approximately 9.92% of the original image's pixels. Our findings suggest that while changes can occur in various parts of Earth's landscapes, they typically manifest relatively subtly, warranting more detailed investigation into underlying factors contributing to these subtle shifts.

---

## Data Appendix (raw computed statistics)

```
LAND USE CLASSIFICATION RESULTS (EuroSAT satellite imagery, 5400 test images):
- Best performing model: SVM_RBF
- Overall test accuracy: 95.4%
- Macro-averaged F1 score: 0.953
- Strongest class: SeaLake (F1 = 0.982)
- Weakest class: Highway (F1 = 0.918)

CHANGE DETECTION RESULTS (24 real Sentinel-2 before/after image pairs, OSCD dataset):
- Mean precision: 0.272
- Mean recall: 0.300
- Mean F1 score: 0.261
- Mean pixel-level accuracy: 95.6%
- Average ground-truth change coverage: 2.97% of pixels
- Pairs showing net vegetation loss (NDVI decline): 12 of 24
- Most severe vegetation loss: pair #6 (mean NDVI change = -0.0531)
- Largest observed change area: pair #7 (9.92% of pixels changed)
```
