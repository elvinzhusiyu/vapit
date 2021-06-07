## VAPIT - Vertex Ai PIpeline Template


- Who is the audience for this article?
    Data Scientists, Developers, AI/ML practitioners


- What problem(s) are we solving for that audience with this article?
    Provide an end-to-end ML pipeline from concept to production with usable templates


- What action(s) do we want the audience to take once they’re done reading this article?
    Clone the sample repository and try running the templates in their own GCP environments

### Introduction
(~150 words)
- Highlight business value of end-to-end ML pipeline on GCP in prod vs running locally in Jupyter notebook
- Highlight template structure for different frameworks (Tensorflow, XGBoost, Scikit-learn, AutoML, etc.)
- Link to frameworks, link to GCP tools, link to the public repo


### Overview
(~150 words)
- Cover end-to-end process on a high level
- Data store → Prep → HPT → Training → Deploy → Prediction
- Not sure if Model explanation is in-scope
- Orchestration aspect
- From workshop slides:
- ML Pipeline should be:
    - Solve your use case: Start from scratch or use pre-existing models.
    - Easy to deploy: Onboard your models and create pipeline easily
    - Scalable as workload changes
    - Composable: Must consist of composable components
    - Orchestrated: Can be orchestrated
    - Secure: Should be secure

### Part 1: Setting up the environment
(~300 words)

### Part 2: Running the components locally
(~300 words)
- Open Jupyterlab on the Notebooks instance and clone the workshop repo
- Open xgboost-gcloud.ipynb
- Talk through each component

### Part 3: Deploying components to Kubeflow pipelines
(~300 words)
- Open xgboost-pipeline.ipynb
- Talk through components quickly
- Talk through DAG orchestration
- Talk through KFP deployment using tarball or using programmatic

### Conclusion
(~150 words)
