import os
import copy
import torch
import numpy as np
import onnxruntime as ort

from src.model.network import ChessNet, NUM_PLANES


def fold_batchnorm(conv, bn):
    """Fold BatchNorm parameters into Conv weights for faster inference.
    Returns new conv with folded weights and biases."""
    folded = copy.deepcopy(conv)
    w = conv.weight.data
    b = conv.bias.data if conv.bias is not None else torch.zeros(w.size(0))

    mu = bn.running_mean
    var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)

    folded.weight.data = w * scale.reshape(-1, 1, 1, 1)
    if folded.bias is None:
        folded.bias = torch.nn.Parameter(torch.zeros(w.size(0)))
    folded.bias.data = (b - mu) * scale + beta

    return folded


def fold_model_batchnorms(model):
    """Create a copy of the model with all BatchNorm layers folded into convolutions."""
    model = copy.deepcopy(model)
    model.eval()

    # input conv + bn
    model.input_conv = fold_batchnorm(model.input_conv, model.input_bn)
    model.input_bn = torch.nn.Identity()

    # residual blocks
    for block in model.res_blocks:
        block.conv1 = fold_batchnorm(block.conv1, block.bn1)
        block.bn1 = torch.nn.Identity()
        block.conv2 = fold_batchnorm(block.conv2, block.bn2)
        block.bn2 = torch.nn.Identity()

    # policy head
    model.policy_conv = fold_batchnorm(model.policy_conv, model.policy_bn)
    model.policy_bn = torch.nn.Identity()

    # value head
    model.value_conv = fold_batchnorm(model.value_conv, model.value_bn)
    model.value_bn = torch.nn.Identity()

    return model


def export_to_onnx(checkpoint_path, output_path="chess_model.onnx", verify=True):
    """Load a checkpoint and export to ONNX format with BatchNorm folding."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    config = ckpt.get('config')
    if config:
        model = ChessNet(num_blocks=config.num_blocks, channels=config.channels)
    else:
        # infer architecture from state dict
        state = ckpt['model_state']
        channels = state['input_conv.weight'].shape[0]
        num_blocks = sum(1 for k in state if k.startswith('res_blocks.') and k.endswith('.conv1.weight'))
        model = ChessNet(num_blocks=num_blocks, channels=channels)

    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # fold batchnorms
    print("Folding BatchNorm layers...")
    folded = fold_model_batchnorms(model)

    # export
    dummy = torch.randn(1, NUM_PLANES, 8, 8)
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        folded,
        (dummy,),
        output_path,
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={"board": {0: "batch"}},
        opset_version=17,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported: {output_path} ({size_mb:.1f} MB)")

    if verify:
        print("Verifying ONNX output matches PyTorch...")
        _verify_export(model, output_path)

    return output_path


def _verify_export(pytorch_model, onnx_path):
    """Check that ONNX output matches PyTorch output within tolerance."""
    pytorch_model.eval()
    dummy = torch.randn(1, NUM_PLANES, 8, 8)

    with torch.no_grad():
        pt_policy, pt_value = pytorch_model(dummy)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_result = sess.run(None, {"board": dummy.numpy()})
    ort_policy, ort_value = ort_result

    policy_diff = np.abs(pt_policy.numpy() - ort_policy).max()
    value_diff = np.abs(pt_value.numpy() - ort_value).max()

    print(f"  Policy max diff: {policy_diff:.6f}")
    print(f"  Value max diff: {value_diff:.6f}")

    if policy_diff < 0.01 and value_diff < 0.01:
        print("  Verification passed")
    else:
        print("  WARNING: outputs differ more than expected")
