# Suppress Git error when importing from ecephys, so it doesn't throw an exception
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
