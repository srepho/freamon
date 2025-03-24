"""
Build and upload the freamon package to PyPI.

This script:
1. Cleans up old builds
2. Builds the package (wheel and sdist)
3. Uploads the package to PyPI
"""
import os
import subprocess
import shutil
import sys

VERSION = "0.3.17"
PACKAGE_NAME = "freamon"

def clean():
    """Clean up old builds and distribution files."""
    print("Cleaning up old builds...")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists(f"{PACKAGE_NAME}.egg-info"):
        shutil.rmtree(f"{PACKAGE_NAME}.egg-info")
    
    print("Cleanup complete!")

def build():
    """Build the package."""
    print(f"Building {PACKAGE_NAME} {VERSION}...")
    
    # Check if we have the build package
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "build"])
    except Exception as e:
        print(f"Error installing build package: {e}")
        return False
    
    # Build the package
    try:
        subprocess.run([sys.executable, "-m", "build"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error building package: {e}")
        return False
    
    print("Build complete!")
    return True

def upload():
    """Upload the package to PyPI."""
    print(f"Uploading {PACKAGE_NAME} {VERSION} to PyPI...")
    
    # Check if we have twine
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "twine"])
    except Exception as e:
        print(f"Error installing twine: {e}")
        return False
    
    # Upload the package
    try:
        subprocess.run([
            sys.executable, "-m", "twine", "upload", 
            "dist/*"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error uploading package: {e}")
        return False
    
    print("Upload complete!")
    
    # Copy files to latest dist folder
    try:
        if not os.path.exists("dist_latest"):
            os.makedirs("dist_latest")
        
        # Copy all files from dist to dist_latest
        for file in os.listdir("dist"):
            shutil.copy(os.path.join("dist", file), os.path.join("dist_latest", file))
        
        print("Copied dist files to dist_latest folder!")
    except Exception as e:
        print(f"Error copying dist files: {e}")
    
    return True

def main():
    """Run the build and upload process."""
    print(f"=== Building and uploading {PACKAGE_NAME} {VERSION} ===")
    
    # Verify version numbers match
    print("Verifying version numbers...")
    
    # Check pyproject.toml
    with open("pyproject.toml", "r") as f:
        pyproject_content = f.read()
        if f'version = "{VERSION}"' not in pyproject_content:
            print(f"ERROR: Version in pyproject.toml does not match {VERSION}")
            return False
    
    # Check __init__.py
    with open(f"{PACKAGE_NAME}/__init__.py", "r") as f:
        init_content = f.read()
        if f'__version__ = "{VERSION}"' not in init_content:
            print(f"ERROR: Version in {PACKAGE_NAME}/__init__.py does not match {VERSION}")
            return False
    
    # Check setup.py
    with open("setup.py", "r") as f:
        setup_content = f.read()
        if f'version="{VERSION}"' not in setup_content:
            print(f"ERROR: Version in setup.py does not match {VERSION}")
            return False
    
    print("All version numbers match!")
    
    # Clean up old builds
    clean()
    
    # Build the package
    if not build():
        return False
    
    # Confirm upload
    while True:
        response = input(f"Upload {PACKAGE_NAME} {VERSION} to PyPI? (y/n): ")
        if response.lower() == 'y':
            break
        elif response.lower() == 'n':
            print("Upload canceled.")
            return False
        else:
            print("Please enter 'y' or 'n'.")
    
    # Upload the package
    if not upload():
        return False
    
    print(f"=== Successfully built and uploaded {PACKAGE_NAME} {VERSION} ===")
    print(f"Package available at: https://pypi.org/project/{PACKAGE_NAME}/{VERSION}/")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)