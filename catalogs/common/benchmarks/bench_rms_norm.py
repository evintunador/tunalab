"""Benchmark for RMSNorm module."""

import torch
from tunalab.testing import BenchmarkRunner
from tunalab.nn_modules.norms.rms_norm import RMSNorm


def main():
    """Run RMSNorm benchmarks."""
    runner = BenchmarkRunner()
    
    results = runner.run_module_benchmark(
        module_class=RMSNorm,
        module_name='RMSNorm',
        parameter_space={
            'dim': [256, 512, 1024, 2048, 4096],
        },
        init_arg_builder=lambda params: {'dim': params['dim']},
        input_provider=lambda init_args: (
            torch.randn(32, 512, init_args['dim'], requires_grad=True),
        ),
    )
    
    print(f"\nBenchmarked {len(results)} configurations")
    print(f"Results saved to artifacts/nn_modules/")


if __name__ == '__main__':
    main()

