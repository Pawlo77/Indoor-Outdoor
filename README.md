# Indoor-Outdoor Classifier

This repository contains code for end-to-end training of an indoor-outdoor image classifier.

## Dataset Preparation

- Uses [Wikimedia WIT](https://huggingface.co/datasets/wikimedia/wit_base) dataset (~6.5M images)
- Labels generated with [IndoorOutdoorNet](https://huggingface.co/prithivMLmods/IndoorOutdoorNet)
- Data extraction pipeline: `python src/data_extraction_pipeline.py --remove-original-files --end 15`
- Processing steps:
  - Download batch, extract image & embedding
  - Remove corrupted/unhelpful images (CLIP, SigLIP, IndoorOutdoorNet)
  - Deduplicate using embeddings (threshold 0.9)
  - Save label, image, embedding
- Secondary preparation: see [`data_preparation.ipynb`](./data_preparation.ipynb)
  - Further deduplication, remove embeddings, stratified split (train/val/test)

## Used Models & Notebooks

- IndoorOutdoorNet ([HuggingFace](https://huggingface.co/prithivMLmods/IndoorOutdoorNet)) — label generation
- OpenAI CLIP ViT-L/14 ([HuggingFace](https://huggingface.co/openai/clip-vit-large-patch14)) — filtering
- SigLIP ([HuggingFace](https://huggingface.co/google/siglip-base-patch16-224)) — noise removal
- MobileNetV4 — see [`mobile_net_v4.ipynb`](./mobile_net_v4.ipynb)
- SigLIP — see [`siglip2.ipynb`](./siglip2.ipynb)
- Results and evaluation — see [`results.ipynb`](./results.ipynb)
- Model exploration — see [`test_models.ipynb`](./test_models.ipynb)

## Additional Datasets

See [`datasets_info.ipynb`](./datasets_info.ipynb) for details on all used datasets.

## Model Training

Model training and evaluation code is provided in the repository. For details on experiments, results, and methodology, see the [full report (PDF)](./report/report.pdf).
