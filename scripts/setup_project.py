#!/usr/bin/env python3
# scripts/setup_project.py
"""
Quick setup script to get the project running
"""
import os
import shutil

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'simulation/envs',
        'simulation/robot',
        'perception/datasets',
        'perception/models',
        'navigation',
        'serving',
        'data/raw',
        'data/processed/images',
        'data/processed/labels',
        'data/simulation_logs',
        'configs',
        'experiments/runs',
        'experiments/notebooks',
        'monitoring',
        'tests',
        'scripts',
        'models'
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create __init__.py for Python packages
        if directory.startswith(('simulation', 'perception', 'navigation', 'serving')):
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                open(init_file, 'w').close()
    
    print("✓ Directory structure created")

def copy_core_files():
    """Copy core implementation files to proper locations"""
    files_to_copy = {
        'kinematics.py': 'simulation/robot/kinematics.py',
        'controller.py': 'simulation/robot/controller.py',
        'gridworld_enhanced.py': 'simulation/envs/gridworld.py',
        'vision_dataset.py': 'perception/datasets/vision_dataset.py',
        'train.py': 'perception/train.py',
        'train_rl.py': 'navigation/train_rl.py',
        'app.py': 'serving/app.py',
        'test_simulation.py': 'tests/test_simulation.py'
    }
    
    print("\nCopying implementation files...")
    for src, dst in files_to_copy.items():
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            print(f"  ✓ {src} -> {dst}")
    
    print("✓ Core files copied")

def create_config_files():
    """Create basic configuration files"""
    import yaml
    
    print("\nCreating configuration files...")
    
    # Data config for YOLO
    data_config = {
        'path': 'data/processed',
        'train': 'images',
        'val': 'images',
        'nc': 2,
        'names': ['obstacle', 'human']
    }
    
    with open('configs/data.yaml', 'w') as f:
        yaml.dump(data_config, f)
    print("  ✓ configs/data.yaml")
    
    # Environment config
    env_config = {
        'world_size': 10,
        'grid_resolution': 64,
        'max_steps': 500,
        'dt': 0.1,
        'num_obstacles': 3
    }
    
    with open('configs/env.yaml', 'w') as f:
        yaml.dump(env_config, f)
    print("  ✓ configs/env.yaml")
    
    # Training config
    train_config = {
        'perception': {
            'epochs': 50,
            'batch_size': 16,
            'image_size': 640,
            'model': 'yolov8n'
        },
        'navigation': {
            'total_timesteps': 100000,
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64
        }
    }
    
    with open('configs/training.yaml', 'w') as f:
        yaml.dump(train_config, f)
    print("  ✓ configs/training.yaml")

def create_gitignore():
    """Create .gitignore file"""
    print("\nCreating .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# Data
data/raw/*
data/processed/*
!data/processed/.gitkeep
data/simulation_logs/*

# Models
*.pt
*.pth
*.onnx
*.zip
models/*
!models/.gitkeep

# Experiments
experiments/runs/*
mlruns/
.mlflow/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("  ✓ .gitignore")

def create_placeholders():
    """Create placeholder files"""
    print("\nCreating placeholder files...")
    
    placeholders = [
        'data/processed/.gitkeep',
        'models/.gitkeep',
        'experiments/runs/.gitkeep'
    ]
    
    for placeholder in placeholders:
        os.makedirs(os.path.dirname(placeholder), exist_ok=True)
        open(placeholder, 'w').close()
    
    print("  ✓ Placeholder files created")

def print_next_steps():
    """Print next steps for user"""
    print("\n" + "="*60)
    print("PROJECT SETUP COMPLETE!")
    print("="*60)
    print("\n📋 NEXT STEPS:\n")
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt\n")
    print("2. Generate training data:")
    print("   python simulation/generate_data.py\n")
    print("3. Train perception model:")
    print("   python perception/train.py\n")
    print("4. Train navigation policy:")
    print("   python navigation/train_rl.py\n")
    print("5. Test the simulation:")
    print("   python scripts/run_simulation.py\n")
    print("6. Deploy API:")
    print("   python serving/app.py\n")
    print("="*60)
    print("\n📚 Check README.md for detailed documentation")
    print("🧪 Run tests with: pytest tests/\n")

def main():
    """Main setup function"""
    print("="*60)
    print("ROBOT MLOPS AUTONOMY - PROJECT SETUP")
    print("="*60)
    
    create_directory_structure()
    copy_core_files()
    create_config_files()
    create_gitignore()
    create_placeholders()
    print_next_steps()

if __name__ == '__main__':
    main()
