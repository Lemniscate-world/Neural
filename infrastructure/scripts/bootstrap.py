import os
import subprocess
import platform


def bootstrap():
    # 1. Ensure VENV
    print("[bootstrap] Preparing virtual environment...")
    # Use system python to run ensure_venv.py
    system_python = "python" if platform.system() == "Windows" else "python3"
    subprocess.run([system_python, "infrastructure/scripts/ensure_venv.py"], check=True)

    # 2. Get VENV Python
    venv_dir = ".venv"
    if platform.system() == "Windows":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    # 3. Install hooks
    print("[bootstrap] Installing git hooks...")
    # We can use pre-commit to install hooks
    subprocess.run([venv_python, "-m", "pre_commit", "install"], check=True)

    # 4. Final check
    print("[bootstrap] Onboarding complete.")
    if platform.system() == "Windows":
        print(f"[bootstrap] Activate with: .\\{venv_dir}\\Scripts\\activate")
    else:
        print(f"[bootstrap] Activate with: source {venv_dir}/bin/activate")


if __name__ == "__main__":
    bootstrap()
