# coding:utf-8
"""
Export trained PyTorch model to ONNX format for deployment.
Usage: python export_onnx.py --checkpoint models/checkpoints/best.pth --output models/sign_recognizer.onnx

The exported ONNX model can be loaded by core.inference.ONNXRecognizer
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to PyTorch checkpoint (.pth)')
    parser.add_argument('--output', type=str, default='models/sign_recognizer.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--dynamic', action='store_true', default=True,
                        help='Use dynamic axes for batch and sequence length')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX graph using onnx-simplifier')
    parser.add_argument('--opset', type=int, default=14,
                        help='ONNX opset version')
    return parser.parse_args()


def export():
    args = parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print('PyTorch is required. Install with: pip install torch')
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f'Checkpoint not found: {args.checkpoint}')
        sys.exit(1)

    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Build model
    from core.temporal.slowfast_tcn import SlowFastTCN
    num_classes = checkpoint['tcn_model']['output_proj.3.weight'].shape[0]
    model = SlowFastTCN(input_dim=256, num_classes=num_classes)

    if 'tcn_model' in checkpoint:
        model.load_state_dict(checkpoint['tcn_model'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f'Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Dummy input
    batch_size = 1
    seq_length = 32
    dummy_input = torch.randn(batch_size, seq_length, 256)

    # Dynamic axes
    dynamic_axes = {
        'input': {0: 'batch_size', 1: 'sequence_length'},
        'output': {0: 'batch_size', 1: 'sequence_length'}
    } if args.dynamic else None

    # Export
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    print(f'Exported to: {args.output}')

    # Simplify
    if args.simplify:
        try:
            import onnx
            from onnxsim import simplify
            model_onnx = onnx.load(args.output)
            simplified, check = simplify(model_onnx)
            if check:
                onnx.save(simplified, args.output)
                print('ONNX model simplified')
            else:
                print('Simplification failed, keeping original')
        except ImportError:
            print('onnx-simplifier not installed, skipping simplification')

    # Verify
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(args.output)
        outputs = session.run(None, {'input': dummy_input.numpy()})
        print(f'ONNX verification OK. Output shape: {outputs[0].shape}')
    except Exception as e:
        print(f'ONNX verification failed: {e}')

    print('Export complete!')


if __name__ == '__main__':
    export()
