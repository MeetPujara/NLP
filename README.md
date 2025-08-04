# Emotion Detection Web App 😃

This project is a Streamlit-based web application that detects the emotion in a given text. It utilizes a pre-trained machine learning model to predict the emotion and displays it with a corresponding emoji and color for better visual representation. The model is trained using a dataset of text samples and their corresponding emotion labels. This application provides a user-friendly interface for real-time emotion analysis.

🚀 **Key Features**

*   **Real-time Emotion Detection:** Predicts the emotion of user-inputted text in real-time.
*   **Pre-trained Model:** Uses a pre-trained machine learning model for accurate emotion prediction.
*   **Text Preprocessing:** Includes text cleaning steps like expanding contractions, removing punctuation, and removing stop words.
*   **Emoji and Color Representation:** Displays the predicted emotion with a corresponding emoji and color for visual appeal.
*   **Streamlit UI:** User-friendly web interface built with Streamlit.
*   **Model Persistence:** The trained model, vectorizer, and label mapping are saved for later use.
*   **Training Pipeline:** Includes a Jupyter Notebook for training and evaluating the emotion detection model.

🛠️ **Tech Stack**

*   **Frontend:**
    *   Streamlit
*   **Backend:**
    *   Python
*   **ML Libraries:**
    *   scikit-learn (sklearn)
    *   nltk
*   **Data Handling:**
    *   pandas
    *   joblib
*   **Other:**
    *   string
    *   re
    *   typing
    *   numpy
*   **Notebook Environment:**
    *   Jupyter Notebook

📦 **Getting Started**

### Prerequisites

*   Python 3.6 or higher
*   pip package installer

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/MeetPujara/NLP
    cd NLP
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Running Locally

1.  **Train the model (if necessary):**

    *   Open `NLP.ipynb` in Jupyter Notebook.
    *   Run all cells to train the model and save the necessary files.

2.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

3.  **Open the application in your browser:**

    *   Streamlit will provide a URL (usually `http://localhost:8501`).

📂 **Project Structure**

```
├── app.py               # Main Streamlit application
├── NLP.ipynb            # Jupyter Notebook for model training
├── train.txt            # Training dataset
├── requirements.txt     # Project dependencies
├── model.pkl            # Trained machine learning model (output of NLP.ipynb)
├── vectorizer.pkl       # TF-IDF vectorizer (output of NLP.ipynb)
└── label_mapping.pkl    # Label mapping (output of NLP.ipynb)
```


🤝 **Contributing**

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with descriptive messages.
4.  Submit a pull request.

📬 **Contact**

If you have any questions or suggestions, feel free to contact me at meetpujara02@gmail.com

💖 **Thanks**

Thank you for checking out this project! I hope you find it useful.

This is written by [readme.ai](https://readme-generator-phi.vercel.app/).
