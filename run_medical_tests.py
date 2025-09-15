"""
Medical Image Algorithm Testing Script

Vision-101의 모든 의료 이미지 알고리즘을 순차적으로 테스트하고
결과를 저장하는 통합 스크립트입니다.

실행되는 알고리즘:
1. Medical Image VAE
2. Medical Image GAN (+ variants: DCGAN, WGAN, WGAN-GP, StyleGAN, CycleGAN, Conditional GAN)
3. Medical Diffusion Models (DDPM, DDIM, Latent Diffusion, Score-based SDE, Conditional Diffusion)
4. Medical NeRF
5. 기타 3D reconstruction 알고리즘들

모든 결과는 results/ 디렉토리에 timestamp와 함께 저장됩니다.
"""

import os
import sys
import time
import subprocess
import argparse
from datetime import datetime

def run_algorithm(script_path, algorithm_name, timeout=3600):
    """
    개별 알고리즘 실행

    Args:
        script_path: 실행할 스크립트 경로
        algorithm_name: 알고리즘 이름 (로깅용)
        timeout: 최대 실행 시간 (초)

    Returns:
        bool: 성공 여부
    """
    print(f"\n{'='*60}")
    print(f"Running {algorithm_name}")
    print(f"Script: {script_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    try:
        # Check if script exists
        if not os.path.exists(script_path):
            print(f"ERROR: Script not found: {script_path}")
            return False

        # Run the script
        start_time = time.time()

        result = subprocess.run(
            [sys.executable, script_path],
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path)
        )

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"\nExecution completed in {execution_time:.2f} seconds")

        if result.returncode == 0:
            print(f"✅ {algorithm_name} completed successfully!")
            print("\nOutput:")
            print(result.stdout[-1000:])  # Show last 1000 characters
        else:
            print(f"❌ {algorithm_name} failed with return code: {result.returncode}")
            print("\nError output:")
            print(result.stderr[-1000:])
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {algorithm_name} timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"💥 {algorithm_name} crashed with error: {e}")
        return False

    return True

def main():
    parser = argparse.ArgumentParser(description="Run medical image algorithm tests")
    parser.add_argument('--algorithms', nargs='+',
                       choices=['vae', 'gan', 'nerf',
                               'gan_variants', 'diffusion_variants',
                               'dcgan', 'wgan', 'wgan_gp', 'stylegan', 'cyclegan', 'conditional_gan',
                               'ddpm', 'ddim', 'latent_diffusion', 'score_based_sde', 'conditional_diffusion',
                               'all', 'all_gan', 'all_diffusion'],
                       default=['all'],
                       help='Which algorithms to run')
    parser.add_argument('--dataset',
                       choices=['chest_xray', 'brain_mri', 'skin_lesion'],
                       default='chest_xray',
                       help='Dataset type to use')
    parser.add_argument('--timeout', type=int, default=1800,
                       help='Timeout per algorithm in seconds (default: 1800)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with reduced epochs for quick testing')

    args = parser.parse_args()

    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Define algorithms to run
    algorithms = {
        # Core algorithms
        'vae': {
            'script': os.path.join(base_dir, 'generating', 'vae_medical_example.py'),
            'name': 'Medical VAE'
        },
        'gan': {
            'script': os.path.join(base_dir, 'generating', 'gan_medical_example.py'),
            'name': 'Medical GAN'
        },
        'nerf': {
            'script': os.path.join(base_dir, '3d', 'nerf_medical_example.py'),
            'name': 'Medical NeRF'
        },

        # GAN Variants
        'dcgan': {
            'script': os.path.join(base_dir, 'generating', 'gan_variants', 'dcgan_medical_example.py'),
            'name': 'Medical DCGAN'
        },
        'wgan': {
            'script': os.path.join(base_dir, 'generating', 'gan_variants', 'wgan_medical_example.py'),
            'name': 'Medical WGAN'
        },
        'wgan_gp': {
            'script': os.path.join(base_dir, 'generating', 'gan_variants', 'wgan_gp_medical_example.py'),
            'name': 'Medical WGAN-GP'
        },
        'stylegan': {
            'script': os.path.join(base_dir, 'generating', 'gan_variants', 'stylegan_medical_example.py'),
            'name': 'Medical StyleGAN'
        },
        'cyclegan': {
            'script': os.path.join(base_dir, 'generating', 'gan_variants', 'cyclegan_medical_example.py'),
            'name': 'Medical CycleGAN'
        },
        'conditional_gan': {
            'script': os.path.join(base_dir, 'generating', 'gan_variants', 'conditional_gan_medical_example.py'),
            'name': 'Medical Conditional GAN'
        },

        # Diffusion Models
        'ddpm': {
            'script': os.path.join(base_dir, 'generating', 'diffusion_variants', 'ddpm_medical_example.py'),
            'name': 'Medical DDPM'
        },
        'ddim': {
            'script': os.path.join(base_dir, 'generating', 'diffusion_variants', 'ddim_medical_example.py'),
            'name': 'Medical DDIM'
        },
        'latent_diffusion': {
            'script': os.path.join(base_dir, 'generating', 'diffusion_variants', 'latent_diffusion_medical_example.py'),
            'name': 'Medical Latent Diffusion'
        },
        'score_based_sde': {
            'script': os.path.join(base_dir, 'generating', 'diffusion_variants', 'score_based_diffusion_medical_example.py'),
            'name': 'Medical Score-based SDE'
        },
        'conditional_diffusion': {
            'script': os.path.join(base_dir, 'generating', 'diffusion_variants', 'conditional_diffusion_medical_example.py'),
            'name': 'Medical Conditional Diffusion'
        }
    }

    # Define algorithm groups
    gan_variants = ['dcgan', 'wgan', 'wgan_gp', 'stylegan', 'cyclegan', 'conditional_gan']
    diffusion_variants = ['ddpm', 'ddim', 'latent_diffusion', 'score_based_sde', 'conditional_diffusion']
    core_algorithms = ['vae', 'gan', 'nerf']

    # Determine which algorithms to run
    algorithms_to_run = []
    for algo in args.algorithms:
        if algo == 'all':
            algorithms_to_run.extend(list(algorithms.keys()))
        elif algo == 'all_gan':
            algorithms_to_run.extend(['gan'] + gan_variants)
        elif algo == 'all_diffusion':
            algorithms_to_run.extend(diffusion_variants)
        elif algo == 'gan_variants':
            algorithms_to_run.extend(gan_variants)
        elif algo == 'diffusion_variants':
            algorithms_to_run.extend(diffusion_variants)
        elif algo in algorithms:
            algorithms_to_run.append(algo)
        else:
            print(f"⚠️  Unknown algorithm: {algo}")

    # Remove duplicates while preserving order
    algorithms_to_run = list(dict.fromkeys(algorithms_to_run))

    print("Medical Image Algorithm Testing")
    print("="*50)
    print(f"Dataset: {args.dataset}")
    print(f"Algorithms requested: {args.algorithms}")
    print(f"Algorithms to run: {len(algorithms_to_run)} algorithms")
    print(f"  - Core: {len([a for a in algorithms_to_run if a in core_algorithms])}")
    print(f"  - GAN Variants: {len([a for a in algorithms_to_run if a in gan_variants])}")
    print(f"  - Diffusion Models: {len([a for a in algorithms_to_run if a in diffusion_variants])}")
    print(f"Timeout per algorithm: {args.timeout} seconds")
    print(f"Quick test mode: {args.quick_test}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create results summary
    results_summary = {
        'start_time': datetime.now(),
        'dataset': args.dataset,
        'algorithms': {},
        'quick_test': args.quick_test
    }

    # Set environment variables for quick testing
    if args.quick_test:
        os.environ['QUICK_TEST'] = '1'
        os.environ['TEST_EPOCHS'] = '5'
        print("⚡ Quick test mode enabled (5 epochs per algorithm)")

    # Run each algorithm
    for algo_key in algorithms_to_run:
        if algo_key not in algorithms:
            print(f"⚠️  Unknown algorithm: {algo_key}")
            continue

        algo_info = algorithms[algo_key]

        start_time = time.time()
        success = run_algorithm(
            algo_info['script'],
            algo_info['name'],
            args.timeout
        )
        end_time = time.time()

        results_summary['algorithms'][algo_key] = {
            'name': algo_info['name'],
            'success': success,
            'execution_time': end_time - start_time,
            'script_path': algo_info['script']
        }

        # Brief pause between algorithms
        if len(algorithms_to_run) > 1:
            print("\nTaking a 5-second break before next algorithm...")
            time.sleep(5)

    # Final summary
    results_summary['end_time'] = datetime.now()
    total_time = results_summary['end_time'] - results_summary['start_time']

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time}")
    print(f"Dataset used: {args.dataset}")
    print(f"Quick test mode: {args.quick_test}")

    successful_algorithms = []
    failed_algorithms = []

    for algo_key, result in results_summary['algorithms'].items():
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        execution_time = result['execution_time']

        print(f"\n{result['name']}: {status}")
        print(f"  Execution time: {execution_time:.2f} seconds")
        print(f"  Script: {result['script_path']}")

        if result['success']:
            successful_algorithms.append(result['name'])
        else:
            failed_algorithms.append(result['name'])

    print(f"\n📊 Results: {len(successful_algorithms)} passed, {len(failed_algorithms)} failed")

    if successful_algorithms:
        print("✅ Successful algorithms:")
        for name in successful_algorithms:
            print(f"  - {name}")

    if failed_algorithms:
        print("❌ Failed algorithms:")
        for name in failed_algorithms:
            print(f"  - {name}")

    # Save results summary
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    summary_file = os.path.join(results_dir, f'test_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Medical Image Algorithm Test Summary\n")
        f.write("="*50 + "\n")
        f.write(f"Start time: {results_summary['start_time']}\n")
        f.write(f"End time: {results_summary['end_time']}\n")
        f.write(f"Total time: {total_time}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Quick test mode: {args.quick_test}\n")
        f.write(f"Algorithms tested: {len(results_summary['algorithms'])}\n")
        f.write(f"Successful: {len(successful_algorithms)}\n")
        f.write(f"Failed: {len(failed_algorithms)}\n\n")

        f.write("Detailed Results:\n")
        f.write("-"*20 + "\n")
        for algo_key, result in results_summary['algorithms'].items():
            status = "PASSED" if result['success'] else "FAILED"
            f.write(f"{result['name']}: {status}\n")
            f.write(f"  Execution time: {result['execution_time']:.2f}s\n")
            f.write(f"  Script: {result['script_path']}\n\n")

    print(f"\n📄 Test summary saved to: {summary_file}")

    # Show where to find results
    print(f"\n📁 Individual algorithm results saved in:")
    print(f"  {results_dir}/")
    print(f"  Each algorithm creates its own timestamped subdirectory")

    return len(failed_algorithms) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)