import torch

from tunalab.benchmark_modules import ModuleBenchmarkRunner
from tunalab.nn_modules.channel_mixing.mlp import MLP


def main():
    """Run MLP benchmarks."""
    runner = ModuleBenchmarkRunner()
    
    results = runner.run_module_benchmark(
        module_class=MLP,
        module_name='MLP',
        parameter_space={
            'dim': [128, 512, 2048],
            'hidden_mult': [2, 4, 8],
            'activation': ['relu', 'silu', 'gelu'],
        },
        init_arg_builder=lambda params: {
            'in_dim': params['dim'],
            'hidden_dim': params['dim'] * params['hidden_mult'],
            'activation': params['activation'],
        },
        input_provider=lambda init_args: (
            torch.randn(512, init_args['in_dim'], requires_grad=True),
        ),
    )
    
    print(f"\nBenchmarked {len(results)} configurations")
    print(f"Results saved to artifacts/nn_modules/")


if __name__ == '__main__':
    main()
