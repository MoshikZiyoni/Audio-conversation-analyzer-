import os
import sys
import platform
import subprocess
import zipfile
import tempfile
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import urllib.request
import ctypes

class FFmpegInstaller:
    def __init__(self, root=None):
        self.root = root
        if self.root:
            self.setup_ui()
        
    def setup_ui(self):
        self.root.title("FFmpeg Installer")
        self.root.geometry("500x350")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="FFmpeg Installation Helper", 
                 font=("Arial", 14, "bold")).pack(pady=(0, 10))
        
        # Description
        desc_text = "This utility will help you install FFmpeg, which is required for audio processing."
        ttk.Label(main_frame, text=desc_text, wraplength=450).pack(pady=(0, 15))
        
        # Status label
        self.status_var = tk.StringVar(value="Checking FFmpeg installation...")
        ttk.Label(main_frame, textvariable=self.status_var, wraplength=450).pack(pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 15))
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.install_btn = ttk.Button(btn_frame, text="Install FFmpeg", command=self.start_install)
        self.install_btn.pack(side=tk.LEFT, padx=5)
        
        self.manual_btn = ttk.Button(btn_frame, text="Show Manual Instructions", 
                                    command=self.show_manual_instructions)
        self.manual_btn.pack(side=tk.LEFT, padx=5)
        
        self.close_btn = ttk.Button(btn_frame, text="Close", command=self.root.destroy)
        self.close_btn.pack(side=tk.RIGHT, padx=5)
        
        # Output log
        frame = ttk.LabelFrame(main_frame, text="Log")
        frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = tk.Text(frame, height=6, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Check for FFmpeg
        self.check_ffmpeg()
    
    def log(self, message):
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.root.update_idletasks()
        print(message)
    
    def update_status(self, message, progress=None):
        if hasattr(self, 'status_var'):
            self.status_var.set(message)
            if progress is not None:
                self.progress_var.set(progress)
            self.root.update_idletasks()
        print(message)
    
    def check_ffmpeg(self):
        """Check if FFmpeg is already installed and accessible."""
        self.update_status("Checking for FFmpeg installation...", 10)
        
        try:
            # Try to run ffmpeg -version
            result = subprocess.run(["ffmpeg", "-version"], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   timeout=5)
            
            if result.returncode == 0:
                self.update_status("FFmpeg is already installed!", 100)
                self.log(f"FFmpeg detected: {result.stdout.splitlines()[0]}")
                self.install_btn.config(text="Reinstall FFmpeg")
                return True
            return False
        except (subprocess.SubprocessError, FileNotFoundError):
            self.update_status("FFmpeg not found. Please install it.", 0)
            self.log("FFmpeg is not installed or not in PATH.")
            return False
    
    def start_install(self):
        """Start FFmpeg installation process in a separate thread."""
        import threading
        self.install_btn.config(state=tk.DISABLED)
        self.manual_btn.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.install_ffmpeg)
        thread.daemon = True
        thread.start()
    
    def is_admin(self):
        """Check if the program is running with administrator privileges."""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    
    def install_ffmpeg(self):
        """Install FFmpeg based on the operating system."""
        system = platform.system()
        self.update_status(f"Installing FFmpeg for {system}...", 20)
        
        try:
            if system == "Windows":
                self.install_ffmpeg_windows()
            elif system == "Darwin":  # macOS
                self.install_ffmpeg_macos()
            elif system == "Linux":
                self.install_ffmpeg_linux()
            else:
                self.update_status(f"Unsupported OS: {system}", 0)
                self.log(f"Sorry, automatic installation not supported for {system}")
                self.show_manual_instructions()
        except Exception as e:
            self.update_status(f"Installation failed: {str(e)}", 0)
            self.log(f"Error during installation: {str(e)}")
            self.install_btn.config(state=tk.NORMAL)
            self.manual_btn.config(state=tk.NORMAL)
    
    def install_ffmpeg_windows(self):
        """Install FFmpeg on Windows."""
        self.log("Starting Windows FFmpeg installation...")
        
        # Define download URL - using a reliable source
        ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        
        try:
            # Create temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "ffmpeg.zip")
                
                # Download FFmpeg
                self.update_status("Downloading FFmpeg...", 30)
                self.log(f"Downloading from {ffmpeg_url}")
                urllib.request.urlretrieve(ffmpeg_url, zip_path)
                
                # Extract the zip file
                self.update_status("Extracting FFmpeg...", 50)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find the bin directory
                for root, dirs, files in os.walk(temp_dir):
                    if "bin" in dirs:
                        bin_dir = os.path.join(root, "bin")
                        break
                else:
                    raise Exception("Could not find bin directory in the extracted files")
                
                # Determine installation directory
                if self.is_admin():
                    # Install for all users if running as admin
                    install_dir = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "FFmpeg")
                else:
                    # Install for current user
                    install_dir = os.path.join(os.path.expanduser("~"), "FFmpeg")
                
                # Create installation directory if it doesn't exist
                os.makedirs(install_dir, exist_ok=True)
                
                # Copy FFmpeg files
                self.update_status("Installing FFmpeg...", 70)
                for file in os.listdir(bin_dir):
                    if file.endswith(".exe"):
                        shutil.copy2(os.path.join(bin_dir, file), install_dir)
                
                # Add to PATH
                self.update_status("Adding FFmpeg to PATH...", 90)
                path_added = self.add_to_path_windows(install_dir)
                
                if path_added:
                    self.update_status("FFmpeg installed successfully!", 100)
                    self.log(f"FFmpeg installed to: {install_dir}")
                    self.log("Please restart your application for the PATH changes to take effect.")
                    messagebox.showinfo("Success", 
                                       "FFmpeg installed successfully! Please restart your application.")
                else:
                    self.update_status("FFmpeg installed, but couldn't add to PATH", 90)
                    self.log(f"FFmpeg installed to: {install_dir}")
                    self.log("Please manually add this directory to your PATH environment variable.")
                    self.show_manual_path_instructions(install_dir)
        
        except Exception as e:
            raise Exception(f"Windows installation failed: {str(e)}")
        finally:
            self.install_btn.config(state=tk.NORMAL)
            self.manual_btn.config(state=tk.NORMAL)
    
    def add_to_path_windows(self, directory):
        """Add a directory to the PATH environment variable on Windows."""
        try:
            if self.is_admin():
                # System PATH
                key_path = "Environment"
                scope = "SYSTEM"
            else:
                # User PATH
                key_path = "Environment"
                scope = "USER"
            
            # Get current PATH
            current_path = os.environ.get("PATH", "")
            
            # Check if directory is already in PATH
            if directory.lower() in [p.lower() for p in current_path.split(os.pathsep)]:
                self.log("Directory already in PATH")
                return True
            
            # Add directory to PATH using setx command
            result = subprocess.run(
                ["setx", "PATH", f"{current_path}{os.pathsep}{directory}", "-m" if scope == "SYSTEM" else ""],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                self.log("Added to PATH successfully")
                # Update current process's PATH (won't affect parent processes)
                os.environ["PATH"] = f"{current_path}{os.pathsep}{directory}"
                return True
            else:
                self.log(f"Failed to add to PATH: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"Error adding to PATH: {str(e)}")
            return False
    
    def install_ffmpeg_macos(self):
        """Install FFmpeg on macOS using Homebrew."""
        self.update_status("Installing FFmpeg for macOS...", 30)
        
        try:
            # Check if Homebrew is installed
            try:
                subprocess.run(["brew", "--version"], check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except (subprocess.SubprocessError, FileNotFoundError):
                self.update_status("Homebrew not found. Installing Homebrew...", 40)
                self.log("Installing Homebrew first...")
                
                # Install Homebrew
                homebrew_install = subprocess.run(
                    ['/bin/bash', '-c', 
                     '$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if homebrew_install.returncode != 0:
                    raise Exception("Failed to install Homebrew")
            
            # Install FFmpeg
            self.update_status("Installing FFmpeg with Homebrew...", 60)
            self.log("Running: brew install ffmpeg")
            
            ffmpeg_install = subprocess.run(
                ["brew", "install", "ffmpeg"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if ffmpeg_install.returncode == 0:
                self.update_status("FFmpeg installed successfully!", 100)
                self.log("FFmpeg installed successfully with Homebrew")
                messagebox.showinfo("Success", "FFmpeg installed successfully!")
            else:
                raise Exception(f"Homebrew ffmpeg installation failed: {ffmpeg_install.stderr}")
                
        except Exception as e:
            raise Exception(f"macOS installation failed: {str(e)}")
        finally:
            self.install_btn.config(state=tk.NORMAL)
            self.manual_btn.config(state=tk.NORMAL)
    
    def install_ffmpeg_linux(self):
        """Install FFmpeg on Linux."""
        self.update_status("Installing FFmpeg for Linux...", 30)
        
        try:
            # Detect package manager
            package_manager = self.detect_linux_package_manager()
            
            if not package_manager:
                raise Exception("Could not detect Linux package manager")
            
            self.log(f"Detected package manager: {package_manager['name']}")
            
            # Install FFmpeg
            self.update_status(f"Installing FFmpeg with {package_manager['name']}...", 50)
            self.log(f"Running: {' '.join(package_manager['install_cmd'])}")
            
            # Use pkexec or sudo for privilege escalation
            if shutil.which("pkexec"):
                install_cmd = ["pkexec"] + package_manager["install_cmd"]
            elif shutil.which("sudo"):
                install_cmd = ["sudo"] + package_manager["install_cmd"]
            else:
                install_cmd = package_manager["install_cmd"]
                self.log("Warning: Running without privilege escalation")
            
            # Run installation command
            install_process = subprocess.run(
                install_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if install_process.returncode == 0:
                self.update_status("FFmpeg installed successfully!", 100)
                self.log("FFmpeg installed successfully")
                messagebox.showinfo("Success", "FFmpeg installed successfully!")
            else:
                raise Exception(f"Installation failed: {install_process.stderr}")
                
        except Exception as e:
            raise Exception(f"Linux installation failed: {str(e)}")
        finally:
            self.install_btn.config(state=tk.NORMAL)
            self.manual_btn.config(state=tk.NORMAL)
    
    def detect_linux_package_manager(self):
        """Detect the Linux package manager."""
        package_managers = [
            {
                "name": "apt",
                "check_cmd": ["apt", "--version"],
                "install_cmd": ["apt", "install", "-y", "ffmpeg"]
            },
            {
                "name": "dnf",
                "check_cmd": ["dnf", "--version"],
                "install_cmd": ["dnf", "install", "-y", "ffmpeg"]
            },
            {
                "name": "yum",
                "check_cmd": ["yum", "--version"],
                "install_cmd": ["yum", "install", "-y", "ffmpeg"]
            },
            {
                "name": "pacman",
                "check_cmd": ["pacman", "--version"],
                "install_cmd": ["pacman", "-S", "--noconfirm", "ffmpeg"]
            },
            {
                "name": "zypper",
                "check_cmd": ["zypper", "--version"],
                "install_cmd": ["zypper", "install", "-y", "ffmpeg"]
            }
        ]
        
        for pm in package_managers:
            try:
                subprocess.run(pm["check_cmd"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=True)
                return pm
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        
        return None
    
    def show_manual_instructions(self):
        """Show manual installation instructions."""
        system = platform.system()
        
        instructions = {
            "Windows": """
Manual FFmpeg Installation for Windows:

1. Download FFmpeg from: https://ffmpeg.org/download.html
   (Look for Windows builds, like gyan.dev or BtbN builds)

2. Extract the zip file to a permanent location (e.g., C:\\FFmpeg)

3. Add FFmpeg to your PATH:
   a. Press Win+X and select "System"
   b. Click "Advanced system settings"
   c. Click "Environment Variables"
   d. Under "System variables" or "User variables", find "Path"
   e. Click "Edit" and add the path to FFmpeg's bin folder
   f. Click "OK" on all dialogs

4. Restart your command prompt or application
""",
            "Darwin": """
Manual FFmpeg Installation for macOS:

1. Install Homebrew (if not already installed):
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

2. Install FFmpeg using Homebrew:
   brew install ffmpeg

3. Verify installation:
   ffmpeg -version
""",
            "Linux": """
Manual FFmpeg Installation for Linux:

1. Debian/Ubuntu:
   sudo apt update
   sudo apt install ffmpeg

2. Fedora:
   sudo dnf install ffmpeg

3. Arch Linux:
   sudo pacman -S ffmpeg

4. CentOS/RHEL:
   sudo yum install epel-release
   sudo yum install ffmpeg

5. Verify installation:
   ffmpeg -version
"""
        }
        
        messagebox.showinfo("Manual Installation Instructions", 
                           instructions.get(system, "Please visit https://ffmpeg.org/download.html for installation instructions."))
    
    def show_manual_path_instructions(self, install_dir):
        """Show instructions for manually adding FFmpeg to PATH."""
        instructions = f"""
Please add the following directory to your PATH environment variable:

{install_dir}

Steps to add to PATH:
1. Press Win+X and select "System"
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "User variables", find "Path"
5. Click "Edit" and add the path above
6. Click "OK" on all dialogs
7. Restart your command prompt or application
"""
        messagebox.showinfo("Add to PATH Instructions", instructions)


def main():
    root = tk.Tk()
    app = FFmpegInstaller(root)
    root.mainloop()

if __name__ == "__main__":
    main()