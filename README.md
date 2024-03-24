# TMR - Text-to-Multimodal Retrieval with Bimodal Input Fusion in Shared Cross-Modal Transformer

[Text-to-Multimodal Retrieval with Bimodal Input Fusion in Shared Cross-Modal Transformer Pranav Arora, Selen Pehlivan, Jorma Laaksonen](url)

![alt text](https://github.com/Pranav260/TMR/blob/main/arch.png)

The paper is accepted in LREC-COLING 2024.

Following is contained in the Repository:
- Code and parameters for our benchmark model
- model weights for our benchmark model
-  The implementation is an extension of the following [work](https://github.com/ninatu/everything_at_once). This repository provides the pretrained backbone models and features used for fine-tuning and evaluations.

## Environment setup:

```bash
conda create python=3.8 -y -n TMR
conda activate everything_at_once 
conda install -y pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install gensim==3.8.0 sacred==0.8.2 humanize==3.14.0 transformers==4.10.2 librosa==0.8.1 timm==0.4.12
pip install neptune-contrib==0.28.1 --ignore-installed certifi
```

