#!/usr/bin/env python3
"""
Build and upload freamon package to PyPI.
"""
import os
import sys
import subprocess
import argparse

def run_command(command, exit_on_error=True):
    """Run a command and optionally exit on error."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"Error output: {result.stderr}")
        if exit_on_error:
            sys.exit(1)
        return False
    print(result.stdout)
    return True

def check_dependencies():
    """Check for required build dependencies."""
    run_command("python -m pip install --upgrade pip")
    run_command("python -m pip install --upgrade build")
    run_command("python -m pip install --upgrade twine")

def clean_previous_builds():
    """Clean up previous build artifacts."""
    if os.path.exists("dist"):
        run_command("rm -rf dist/*")
    if os.path.exists("build"):
        run_command("rm -rf build")
    if os.path.exists("freamon.egg-info"):
        run_command("rm -rf freamon.egg-info")

def build_package():
    """Build source distribution and wheel."""
    run_command("python -m build")

def check_package():
    """Check package for PyPI compatibility."""
    run_command("python -m twine check dist/*")

def upload_to_pypi(test=False):
    """Upload package to PyPI or TestPyPI."""
    if test:
        run_command("python -m twine upload --repository testpypi dist/*")
    else:
        run_command("python -m twine upload dist/*")

def create_latest_dist_copy():
    """Create a copy of the latest distribution in dist_latest."""
    if not os.path.exists("dist_latest"):
        os.makedirs("dist_latest")
    run_command("cp dist/* dist_latest/")

def main():
    """Main function to build and upload the package."""
    parser = argparse.ArgumentParser(description="Build and upload freamon package to PyPI")
    parser.add_argument("--test", action="store_true", help="Upload to TestPyPI instead of PyPI")
    parser.add_argument("--build-only", action="store_true", help="Build package without uploading")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks")
    
    args = parser.parse_args()
    
    if not args.skip_checks:
        print("Checking dependencies...")
        check_dependencies()
    
    print("Cleaning previous builds...")
    clean_previous_builds()
    
    print("Building package...")
    build_package()
    
    print("Checking package...")
    check_package()
    
    print("Creating copy in dist_latest...")
    create_latest_dist_copy()
    
    if not args.build_only:
        print(f"Uploading to {'TestPyPI' if args.test else 'PyPI'}...")
        upload_to_pypi(args.test)
        print("Upload complete!")
    else:
        print("Package built successfully. Skipping upload as requested.")

if __name__ == "__main__":
    main()