import kagglehub
# Authenticate
kagglehub.login() # This will prompt you for your credentials.
# We also offer other ways to authenticate (credential file & env variables): https://github.com/Kaggle/kagglehub?tab=readme-ov-file#authenticate

# Download latest version
path = kagglehub.model_download("nosovartiom/deeppavlovrubert-base-cased-sentence/transformers/vacancy-resume-next-sentence-prediction/1")

print("Path to model files:", path)