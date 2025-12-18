import torch

from tunalab.benchmarking import ModuleBenchmarkRunner
from tunalab.modules.losses.fused_cross_entropy import TorchLinearCELoss, FusedLinearCELoss, LIGER_AVAILABLE


def main():
    runner = ModuleBenchmarkRunner()
    
    parameter_space = {
        'hidden_size': [512, 1024, 2048],
        'vocab_size': [8192, 16384, 32768],
    }
    BATCH_SIZE = 16
    SEQ_LEN = 256
    
    print("Benchmarking TorchLinearCELoss (baseline)...")
    results_torch = runner.run_module_benchmark(
        module_class=TorchLinearCELoss,
        module_name='TorchLinearCELoss',
        parameter_space=parameter_space,
        init_arg_builder=lambda p: {
            'D': p['hidden_size'],
            'V': p['vocab_size'],
            'dtype': torch.float32,
        },
        input_provider=lambda init_args: (
            # x: (batch_size * seq_len, hidden_size)
            torch.randn(
                BATCH_SIZE * SEQ_LEN,
                init_args['D'],
                requires_grad=True
            ),
            # y: (batch_size * seq_len,) with token ids
            torch.randint(
                0, init_args['V'],
                (BATCH_SIZE * SEQ_LEN,)
            ),
        ),
    )
    
    if LIGER_AVAILABLE:
        print("\nBenchmarking FusedLinearCELoss (fused version)...")
        results_fused = runner.run_module_benchmark(
            module_class=FusedLinearCELoss,
            module_name='FusedLinearCELoss',
            parameter_space=parameter_space,
            init_arg_builder=lambda p: {
                'D': p['hidden_size'],
                'V': p['vocab_size'],
                'dtype': torch.float32,
            },
            input_provider=lambda init_args: (
                # x: (batch_size * seq_len, hidden_size)
                torch.randn(
                    BATCH_SIZE * SEQ_LEN,
                    init_args['D'],
                    requires_grad=True
                ),
                # y: (batch_size * seq_len,) with token ids
                torch.randint(
                    0, init_args['V'],
                    (BATCH_SIZE * SEQ_LEN,)
                ),
            ),
        )
        print(f"\nBenchmarked {len(results_fused)} configurations for FusedLinearCELoss")
    else:
        print("\nSkipping FusedLinearCELoss - liger_kernel not available")
    
    print(f"Benchmarked {len(results_torch)} configurations for TorchLinearCELoss")
    print(f"Results saved to artifacts/modules/")


if __name__ == '__main__':
    main()
