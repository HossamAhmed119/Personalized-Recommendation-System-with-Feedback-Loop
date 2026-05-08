import os

PROJECT_ROOT = "recommendation-system"

folders_with_init = [
    "src",
    "src/utils",
    "src/data_pipeline",
    "src/models",
    "src/rag",
    "src/services",
    "src/api",
    "mlops",
    "tests"
]

def create_init_files():
    print("\n🔧 Creating __init__.py files...\n")

    for folder in folders_with_init:
        path = os.path.join(PROJECT_ROOT, folder)
        os.makedirs(path, exist_ok=True)

        init_file = os.path.join(path, "__init__.py")

        with open(init_file, "w", encoding="utf-8") as f:
            f.write("# Auto-generated package initializer\n")

        print(f"[+] Added package: {init_file}")

def create_path_setup():
    print("\n🔗 Creating path setup utility...\n")

    utils_path = os.path.join(PROJECT_ROOT, "src/utils")
    os.makedirs(utils_path, exist_ok=True)

    file_path = os.path.join(utils_path, "path_setup.py")

    content = """import os
import sys

# Get project root dynamically
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Add to system path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print(f\"[INFO] Project root added to sys.path: {PROJECT_ROOT}\")
"""

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[+] Created: {file_path}")

def main():
    print("\n Setting up project structure connectivity...\n")

    create_init_files()
    create_path_setup()

    print("\n DONE Your project is now fully wired and import-ready\n")

if __name__ == "__main__":
    main()