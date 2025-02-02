# Azure AI Services Web App
This repository contains code designed to faciliate your experimentation and exploration with the Azure AI Services such as Azure AI Language, Azure AI Speech, Azure AI Translator, etc.

Azure AI services help developers and organizations to rapidly create intelligent, cutting-edge, market-ready, and responsible applications with out-of-the-box and pre-built and customizable APIs and models. Example applications include natural language processing (NLP) for conversations, search, monitoring, translation, speech, vision, and decision-making.

Most Azure AI services are available through REST APIs and client library SDKs in popular development languages. For more information, see each service's documentation.

## Exploring the Wonderful World of AI with Azure AI Services
<img width="878" alt="image" src="https://github.com/user-attachments/assets/63bf51df-fb1e-4c1b-91a0-0e337ec64c51">
<br>
<img width="878" alt="image" src="https://github.com/user-attachments/assets/633d76b6-f70c-4d0a-a724-334eaf95ca2b">


## Getting Started

Follow the steps below to set up your environment and start utilizing the features of this repository:

### Prerequisites

- Python (>= 3.8)
- Visual Studio Code
  
### Installation

1. Clone this repository to your local machine.

```bash
git clone https://github.com/VincentK16/azure-ai-services-webapp.git
```

2. Navigate to the project directory.

```bash
cd azure-ai-services-webapp
```

3. Create a virtual environment.

```bash
python -m venv <name_of_your_env>
```

4. Activate the virtual environment.

- On Windows:

```bash
<name_of_your_env>\Scripts\activate
```

- On macOS/Linux:
```bash
source <name_of_your_env>/bin/activate
```

5. Install project dependencies from the requirements.txt file.

```bash
pip install -r requirements.txt
```

6. Create a .env file in the root directory of your project to store sensitive information such as the Azure OpenAI resource's keys. You can find a sample .env file in the repository called `.env_sample`. Duplicate this file and rename it to `.env`, then fill in the necessary values.

```bash
cp .env_sample .env
```

Now, you're ready to use the secrets stored in your `.env` file securely within your project. Feel free to customize the `.env` file with your other specific secrets and configurations as needed.

**Note: Never share your `.env` file publicly or commit it to version control systems like Git, as it contains sensitive information. The best practice is to use a `.gitignore` file in your repo to avoid commiting the `.env` file.**

7. Run the web app.
   
```bash
python app.py
```

8. Browse to the web app.

```bash
https://127.0.0.1:5000/ailanguage
```
