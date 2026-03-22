# Modern Recommender Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kimfalk/modern-recommender-systems)

Code examples and implementations for **"Modern Recommender Systems"** published by Manning Publications.

## 📖 About the Book

This book provides a comprehensive guide to building modern recommender systems, with a focus on:

- **Semantic IDs** for representing items and users
- **Deep learning approaches** for recommendations
- **Scalable architectures** for production systems
- **Evaluation and optimization** techniques
- **Real-world case studies** and best practices

## 🎯 What You'll Learn

- How to build recommender systems from scratch
- Implementing semantic ID models for better representations
- Training and evaluating deep learning recommendation models
- Handling cold-start problems and sparse data
- Deploying recommender systems at scale
- A/B testing and online evaluation strategies

read more on manning.com
## 🚀 Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/modern-recommender-systems.git
cd modern-recommender-systems

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Launch Jupyter
jupyter notebook

# To run the MLFlow browser version use

mlflow ui
```

# API Keys Configuration
Several examples in this book use external APIs that require authentication. To run these examples:

* Copy the example environment file
```bash
cp .env.example .env
```

* Open .env in a text editor and add your API keys:
```bash
TMDB_API_KEY=your_actual_api_key_here
ANOTHER_API_KEY=your_other_key_here
```
* Save the file. The .env file is already in .gitignore and will not be committed to version control.

## Where to get API keys
API_KEY: [https://developer.themoviedb.org/reference/authentication] - Sign up at https://developer.themoviedb.org/reference/authentication

# Support
* Book forum: https://livebook.manning.com/book/modern-recommender-systems
* Issues: Please report any problems with the code at https://github.com/yourusername/modern-recommender-systems/issues

# Repository Structure
modern-recsys/
├── README.md
├── requirements.txt
├── setup.py
├── recsys/
│   ├── __init__.py
│   ├── framework.py          # FourStageRecommender, interfaces
│   ├── context.py             # RecommendationContext, ScoredItem
│   ├── retrievals/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── popularity.py
│   │   ├── content_based.py
│   │   └── collaborative.py
│   ├── scorers/
│   │   ├── __init__.py
│   │   ├── passthrough.py
│   │   └── ensemble.py
│   ├── filters/
│   │   ├── __init__.py
│   │   └── history.py
│   └── rankers/
│       ├── __init__.py
│       └── topk.py
├── notebooks/
│   ├── chapter02_popularity.ipynb
│   ├── chapter02_content_based.ipynb
│   ├── chapter02_itemknn.ipynb
│   └── chapter02_comparison.ipynb
├── tests/
└── data/
    └── README.md  # Instructions for downloading MovieLens