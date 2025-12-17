from tunalab.benchmark_modules import OptimizerBenchmarkRunner
from tunalab.optimizers.muon import MuonTestAdapter


def main():
    runner = OptimizerBenchmarkRunner()
    
    print("Benchmarking Muon (via MuonTestAdapter)...")
    
    results = runner.run_optimizer_benchmark(
        optimizer_class=MuonTestAdapter,
        optimizer_name='Muon',
        parameter_space={
            'muon_lr': [1e-3, 1e-2, 1e-1],
            'momentum': [0.9, 0.95, 0.99],
        },
        optimizer_kwargs_builder=lambda p: {
            'muon_lr': p['muon_lr'],
            'adamw_lr': p['muon_lr'] * 0.1,  # AdamW uses lower lr for biases
            'momentum': p['momentum'],
        },
        num_steps=100,
    )
    
    print(f"\nBenchmarked {len(results)} configurations")
    print(f"Results saved to artifacts/optimizers/")


if __name__ == '__main__':
    main()
