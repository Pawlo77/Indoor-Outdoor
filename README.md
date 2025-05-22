# Indoor - Outdoor classifier

Following repo holds code for end-to-end training of indoor-outdoor classifier.

## Wikimedia Dataset Preparation

Entire net is trained on [Wikimedia](https://huggingface.co/datasets/wikimedia/wit_base) image dataset, containing around 6.5 millions of unique images. They are not classified into indoor/outdoor, so labels are generated using [Indoor-Outdoor net](https://huggingface.co/prithivMLmods/IndoorOutdoorNet). MLLMs were also checked as potential models for label generation in [exploration notebook](test_models.ipynb), yet they yielded very poor performance. For maximizing dataset quality, data preparation includes following steps:

### Initial data extraction and preparation:

Performed by [the pipeline](./src/data_extraction_pipeline.py), example call (it will download first 15 files from source, process them, save processed version and remove original file).
```python
python src/data_extraction_pipeline.py --remove-original-files --end 15
```

Processing steps are as follows
- download data batch from source
- extract image, embedding
- remove images that are corrupted, ie length of unique pixel values is smaller than 10
- remove images that are not helpful for training using zero-shot-image-classification with openai/clip-vit-large-patch14 model
- get images class with indoor outdoor net
- remove images where model confidence is low (ie in (0.1, 0.9)) - risk of mislabelling
- perform images deduplication within the file using provided embeddings
- save label, image and embedding

Deduplication threshold is set to 0.9 (similarity stronger than that will lead to removed instances), classification cutoff to 0.1 (if indoor-outdoor net is less certain than 90% about the target class, it will be removed) and noise to 0.95 (is siglip model is more certain than 0.95 that image is "animated chart or infographic" than it will be removed).

### Secondary dataset preparation

Performed in [this notebook](./data_preparation.ipynb). Final dataset is calculated from batches saved in previous point using additional deduplication step and removing embedding for space optimizations. It is stratified splitted into train, valid, test of ratios 0.8, 0.1, 0.1 and saved into .npz files of 1000 entries each.

## Additional datasets

All used additional datasets are described in [notebook](./datasets_info.ipynb).

## Model training
