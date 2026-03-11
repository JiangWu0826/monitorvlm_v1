

# MonitorVLM Project

## Project Introduction
MonitorVLM is an industrial safety monitoring and analysis system based on Vision Language Model (VLM), designed to identify and analyze safety violations in industrial scenarios.
Since the dataset is sourced from mining companies, various violation photos and behaviors cannot be open-sourced.
This project roughly covers the entire process from data balance mining, adversarial augmentation to multimodal LoRA fine-tuning and inference plugin enhancement. However, deployment datasets and text have been omitted.

## Project Structure

```
monitorvlm_v1/
├── bm/                      # Base model related code
│   └── ft_swift_infer_api_pos.py  # Safety analysis script integrated with GroundingDino detection
├── dataset_enrichment_code/ # Dataset enrichment code
│   ├── 1video_frame_extractor.py     # Video frame extraction
│   ├── 2image_augmentation.py       # Image augmentation
│   ├── 2image_flipped.py            # Image flipping
│   ├── 2MASK_frame.py               # Human detection and masking
│   ├── 2dataset_image_location.py   # Dataset image location processing
│   ├── 2masked.py                   # Mask processing
│   └── 3auto_merge_datasets.py      # Automatic dataset merging
├── moe/                     # Mixture of Experts model related
│   ├── best_dynamic_model.pth        # Best dynamic model
│   ├── moe_data_generation.py        # Data generation
│   ├── moe_test_batchsize.py         # Batch testing
│   ├── moe_train_batchsize.py        # Batch training
│   └── safety_analysis_moe_v2.json   # Safety analysis results
├── ms-swift_train/          # Model training related
│   ├── infer_api_test.py             # Inference API test
│   ├── train_qwen2.5vl.sh            # Qwen2.5VL training script
│   └── deploy_inference_72b.sh       # 72B model deployment script
├── data_construction/       # Data construction
│   ├── 202506027total.json           # Total dataset
│   ├── 20250627_original.json        # Original dataset
│   ├── 20250627_original_enrichment.json  # Enriched original dataset
│   └── 20250627_original_location.json   # Original dataset with location information
├── MonitorVLM_interface.mp4 # Interface demonstration video
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```
## Video
https://drive.google.com/file/d/1Qj23DLqOToCt8VlPdW0eGLSVEGbx20kR/view
## Features

### 1. Dataset Processing and Enrichment
- **Video Frame Extraction**: Extract key frames from monitoring videos
- **Image Augmentation**: Augment extracted images
- **Human Detection and Masking**: Detect humans and generate masks using YOLOv8
- **Dataset Merging**: Automatically merge multiple datasets

### 2. Safety Analysis
- **Object Detection**: Detect safety-related objects using GroundingDino
- **Safety Rule Checking**: Check safety rule violations based on detection results
- **Multi-model Analysis**: Use Mixture of Experts model for more accurate safety analysis

### 3. Model Training and Deployment
- **Model Training**: Train Qwen2.5VL model for safety analysis
- **Inference Deployment**: Deploy 72B model for efficient inference

## Installation and Usage

### Environment Requirements
- Python 3.8+
- CUDA 11.7+ (recommended)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Usage Instructions for Main Modules

#### 1. Dataset Processing

**Video Frame Extraction**:
```bash
python dataset_enrichment_code/1video_frame_extractor.py
```

**Human Detection and Masking**:
```bash
python dataset_enrichment_code/2MASK_frame.py
```

**Dataset Merging**:
```bash
python dataset_enrichment_code/3auto_merge_datasets.py
```

#### 2. Safety Analysis

**Safety Analysis with GroundingDino Detection**:
```bash
python bm/ft_swift_infer_api_加入加测模型.py
```

#### 3. Model Training

**Train Qwen2.5VL Model**:
```bash
bash ms-swift_train/train_qwen2.5vl.sh
```

**Deploy Inference Service**:
```bash
bash ms-swift_train/deploy_inference_72b.sh
```

## Key Dependencies

- **ultralytics**: YOLOv8 object detection
- **transformers**: Hugging Face models
- **GroundingDino**: Vision-language object detection
- **PyTorch**: Deep learning framework
- **requests**: API calls
- **Pillow**: Image processing
- **numpy**: Numerical computation

## Notes

1. **Model Path Configuration**: Please modify the model paths in the code according to your actual environment
2. **GPU Requirements**: Some modules require GPU support, CUDA-enabled devices are recommended
3. **Data Paths**: Please modify the input/output paths in the code to suit your environment
4. **Dependency Versions**: Some dependencies may require specific versions, please refer to official documentation if you encounter issues

## Example Output

Safety analysis example output:
- Scene Description: Detailed description of the environment, personnel, equipment, etc. in the image
- Detection Information Verification: Verify object detection results
- Thought Process: Detailed analysis process
- Violated Rules: List confirmed and suspected violated rules
- Overall Conclusion: Safety compliant or violation exists
- Improvement Suggestions: Specific suggestions based on analysis

## Citation
@article{wu2025monitorvlm,
  title={MonitorVLM: A Vision Language Framework for Safety Violation Detection in Mining Operations},
  author={Wu, Jiang and Wu, Sichao and Ma, Yinsong and Yu, Guangyuan and Xu, Haoyuan and Zheng, Lifang and Duan, Jingliang},
  journal={arXiv preprint arXiv:2510.03666},
  year={2025}
}
