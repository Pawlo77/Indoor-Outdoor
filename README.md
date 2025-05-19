# Indoor - Outdoor classifier

Following repo holds code for end-to-end training of indoor-outdoor classifier.

## Dataset Preparation

Entire net is trained on [Wikimedia](https://huggingface.co/datasets/wikimedia/wit_base) image dataset, containing around 6.5 millions of unique images. They are not classified into indoor/outdoor, so labels are generated using [Indoor-Outdoor net](https://huggingface.co/prithivMLmods/IndoorOutdoorNet). MLLMs were also checked as potential models for label generation in [exploration notebook](test_models.ipynb), yet they yielded very poor performance. For maximizing dataset quality, data preparation includes following steps:

- download data batch from source
- extract image, embedding
- remove images that are corrupted, ie length of unique pixel values is smaller than 10
- get images class with indoor outdoor net
- remove images where model confidence is low (ie in (0.1, 0.9)) - risk of mislabelling
- perform images deduplication
- save label, image and embedding

This created initial dataset, that is created in batches. Final dataset is calculated from it using additional deduplication step and removing embedding for space optimizations. It is stratified splitted into train, valid, test of ratios 0.8, 0.1, 0.1 and saved into .npz files of 10000 entries each.

## Model training
