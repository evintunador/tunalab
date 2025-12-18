from tunalab.benchmarking import OptimizerBenchmarkRunner
from tunalab.optimizers.adamw import AdamW


def main():
    runner = OptimizerBenchmarkRunner()
    
    print("Benchmarking AdamW...")
    
    results = runner.run_optimizer_benchmark(
        optimizer_class=AdamW,
        optimizer_name='AdamW',
        parameter_space={
            'lr': [1e-4, 1e-3, 1e-2],
            'weight_decay': [0.0, 0.01, 0.1],
        },
        optimizer_kwargs_builder=lambda p: {
            'lr': p['lr'],
            'weight_decay': p['weight_decay'],
        },
        num_steps=100,
    )
    
    print(f"\nBenchmarked {len(results)} configurations")
    print(f"Results saved to artifacts/optimizers/")


if __name__ == '__main__':
    main()
