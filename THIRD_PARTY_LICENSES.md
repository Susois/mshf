# Third-Party Components

This inventory is a compliance aid, not a substitute for reviewing the exact package and checkpoint licenses installed in an experiment.

| Component | Use | License/source to verify |
|---|---|---|
| NumPy | Numerical computation | Installed package metadata and https://numpy.org |
| pandas | Tabular processing | Installed package metadata and https://pandas.pydata.org |
| scikit-learn | Splits, metrics, calibration | Installed package metadata and https://scikit-learn.org |
| SciPy | Statistical tests | Installed package metadata and https://scipy.org |
| XGBoost | Primary classifiers | Installed package metadata and https://xgboost.ai |
| RapidFuzz | Edit distance | Installed package metadata and project repository |
| joblib | Checkpoint serialization | Installed package metadata and project repository |
| PyTorch | Optional GPU inference | Installed package metadata and https://pytorch.org |
| Transformers | Optional model loading | Installed package metadata and https://huggingface.co/docs/transformers |
| PhoBERT checkpoint | Existing document semantic features | Exact model card, revision and license |
| LayoutLMv3 checkpoint | Existing layout feature | Exact model card, revision and license |
| XLM-R/XNLI checkpoint | Optional B2 contradiction baseline | Exact model card, revision and license |
| PaddleOCR | Upstream OCR artifacts | Exact package/model license and repository |

For every reported run, record:

- exact package versions;
- pretrained model ID and immutable revision/commit;
- model-card license;
- whether weights were redistributed or only referenced;
- citation required by the model/software authors.

Do not assume that a model checkpoint has the same license as the library used to load it.
