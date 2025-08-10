# SDERM AI
SDERM AI (Skin Disease Early Recognition Model) is an AI-powered dermatology image classification tool designed to enhance accessibility, accuracy, and inclusivity in skin disease diagnostics, particularly for underprivileged and remote communities.

This project uses a pretrained ConvNeXt-Tiny model fine-tuned on a curated dermatology dataset, enabling it to classify various skin conditions from medical images.

Features
Pretrained ConvNeXt-Tiny model with fine-tuning for higher accuracy.

Support for diverse skin tones to reduce bias.

Easy-to-run scripts for training, evaluation, and prediction.

Built with PyTorch and Torchvision.

Dataset
We used the DermNet Skin Disease Images Dataset, available here:
https://www.kaggle.com/datasets/umairshahab/dermnet-skin-diesease-images

Download and extract the dataset before running the code.

Installation

git clone https://github.com/yourusername/ai_derm.git
cd ai_derm
pip install -r requirements.txt

Model Details
Base Model: ConvNeXt-Tiny (ImageNet pretrained)

Loss Function: CrossEntropyLoss with class weighting

Optimizer: Adam / AdamW

Mixed Precision Training enabled

Credits
Dataset: DermNet Skin Disease Images

Research Guidance: Prof. Kai Shu, Emory University

Development: Michael Wang

