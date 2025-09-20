# MDIP: Misinformation Detection and Information Processing

This repository contains a comprehensive system for detecting and analyzing COVID-19 misinformation using machine learning techniques. The project combines fake news classification with topic modeling to provide detailed insights into misinformation patterns.

## Overview

The MDIP system consists of two main components:
1. **Fake News Classification**: A BERT-based classifier trained to distinguish between real and fake news
2. **Topic Modeling**: BERTopic-based analysis to identify and categorize misinformation themes

## Datasets Used

### Training Datasets for Classifier
- **AAAI 2020 Constraint**
- **FNIR COVID**

### Topic Modeling Dataset
- **IFCN COVID Misinformation**: International Fact-Checking Network's COVID-19 misinformation dataset used for topic modeling and theme identification

## Project Structure

```
MDIP/
├── classofier.py          # BERT-based fake news classifier training
├── MDIS.py               # Main detection and information system
├── Topic_description.py  # Topic description generation using GPT-4
├── Topic_theme.py        # Theme categorization and analysis
├── TopicModeling.py      # BERTopic implementation for topic modeling
├── themes_descriptions.txt  # Detailed descriptions of misinformation themes
├── themes_qa_list.txt    # Q&A examples for theme classification
└── README.md            # This file
```

## Misinformation Themes

The system identifies 13 main categories of COVID-19 misinformation:

1. **Home Remedies and Misconceptions** - Unproven treatments and prevention methods
2. **COVID-19 Deaths and Statistics** - Misleading data and death counts
3. **Conspiracy Theories** - Unfounded theories about pandemic origins
4. **Vaccine Controversies** - Misinformation about vaccine safety and efficacy
5. **COVID-19 Testing** - False claims about test accuracy and methods
6. **Ivermectin and Treatments** - Unproven medical treatments
7. **Government Responses** - Misinformation about political actions
8. **Lockdown and Restrictions** - False claims about containment measures
9. **Transmission Paths** - Misconceptions about how COVID-19 spreads
10. **Protective Measures** - Misinformation about prevention methods
11. **International Incidents** - False reports about global responses
12. **Media and Communication** - Misinformation spread through social media
13. **Miscellaneous Topics** - Other minor misinformation themes

## Installation

### Prerequisites
```bash
pip install torch transformers datasets scikit-learn pandas numpy
pip install bertopic umap-learn sentence-transformers
pip install openai matplotlib
```

### Setup
1. Clone the repository
2. Install required dependencies
3. Set up OpenAI API key for GPT-4 integration
4. Update file paths in the scripts to point to your datasets

## Usage

### Training the Classifier
```python
# Update paths in classofier.py
train_path = '/path/to/train_data.csv'
val_path = '/path/to/val_data.csv'

# Run training
python classofier.py
```

### Running Topic Modeling
```python
# Update path in TopicModeling.py
df = pd.read_csv('/path/to/Misinfo.csv')

# Run topic modeling
python TopicModeling.py
```

### Using the Detection System
```python
# Update paths in MDIS.py
model_save_path = "/path/to/model"
tokenizer_save_path = "/path/to/tokenizer"

# Set input text
input_text = "Your text to analyze"

# Run detection
python MDIS.py
```


## Output

The system provides:
1. **Classification Results**: Real/Fake prediction with confidence
2. **Theme Identification**: Specific misinformation category for fake content
3. **Inoculation Message**: Factual information to counter misinformation
4. **Topic Analysis**: Detailed topic modeling results with document assignments

## API Integration

The system integrates with OpenAI's GPT-4 API for:
- Theme classification
- Topic description generation
- Theme categorization

Ensure you have a valid OpenAI API key and sufficient credits.


## License

This project is for research and educational purposes. Please ensure compliance with dataset licenses and OpenAI's terms of service.

## Citation

If you use this work in your research, please cite the relevant datasets:
- AAAI 2020 Constraint Dataset
- FNIR COVID Dataset
- IFCN COVID Misinformation Dataset

## Contact

For questions or issues, please open an issue in the repository.
