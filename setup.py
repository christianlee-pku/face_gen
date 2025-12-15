from setuptools import setup, find_packages

setup(
    name="face-gen-pipeline",
    version="0.1.0",
    description="End-to-End ML Face Generation Pipeline",
    author="Face Gen Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "torchvision",
        "mlflow",
        "boto3",
        "scikit-learn",
        "fastapi",
        "uvicorn[standard]",
        "onnxruntime",
        "python-multipart",
    ],
)
