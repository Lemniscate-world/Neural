import os
import subprocess
import shutil
import platform
import urllib.request


def get_python_version(executable):
    try:
        result = subprocess.run(
            [executable, "--version"], capture_output=True, text=True
        )
        version_str = result.stdout.split()[1]
        return ".".join(version_str.split(".")[:2])
    except Exception:
        return None


def ensure_venv():
    venv_dir = ".venv"
    if platform.system() == "Windows":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
        venv_pip = os.path.join(venv_dir, "Scripts", "pip.exe")
        system_python = "python"
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")
        venv_pip = os.path.join(venv_dir, "bin", "pip")
        system_python = "python3"

    sys_ver = get_python_version(system_python)
    venv_ver = get_python_version(venv_python)

    if not venv_ver or sys_ver != venv_ver:
        print(f"[ensure_venv] Recreating .venv (sys={sys_ver}, venv={venv_ver})...")
        if os.path.exists(venv_dir):
            shutil.rmtree(venv_dir)

        subprocess.run(
            [system_python, "-m", "venv", "--without-pip", venv_dir], check=True
        )

        # Bootstrap pip
        get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
        temp_dir = os.environ.get("TEMP", os.environ.get("TMP", "/tmp"))
        get_pip_path = os.path.join(temp_dir, "get-pip.py")

        print(f"[ensure_venv] Downloading get-pip.py to {get_pip_path}...")
        urllib.request.urlretrieve(get_pip_url, get_pip_path)

        subprocess.run([venv_python, get_pip_path, "--quiet"], check=True)
        os.remove(get_pip_path)

        # Install dependencies
        print("[ensure_venv] Installing dependencies...")
        subprocess.run([venv_pip, "install", "--upgrade", "pip", "--quiet"], check=True)

        # Install torch with CPU preference for dev stability
        subprocess.run(
            [
                venv_pip,
                "install",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                "--extra-index-url",
                "https://pypi.org/simple",
                "torch",
                "--quiet",
            ],
            check=True,
        )

        # Install other requirements via pyproject.toml if possible, or requirements if they exist
        if os.path.exists("pyproject.toml"):
            subprocess.run(
                [
                    venv_pip,
                    "install",
                    "-e",
                    ".[dev,automation,mlops,sync,docs]",
                    "--quiet",
                ],
                check=True,
            )

        print(
            f"[ensure_venv] Done — .venv now uses Python {get_python_version(venv_python)}."
        )
    else:
        print("[ensure_venv] .venv is up to date.")


if __name__ == "__main__":
    ensure_venv()
