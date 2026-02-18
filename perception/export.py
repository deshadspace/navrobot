"""
Model export utilities.

Convert trained models to production formats (ONNX, TorchScript).
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class ModelExporter:
    """Export models to various formats."""
    
    @staticmethod
    def to_onnx(
        model: nn.Module,
        input_shape: tuple,
        output_path: Path,
        opset_version: int = 12,
    ) -> None:
        """
        Export model to ONNX format.
        
        Args:
            model: PyTorch model
            input_shape: dummy input shape (batch, channels, height, width)
            output_path: where to save ONNX file
            opset_version: ONNX opset version
        """
        model.eval()
        dummy_input = torch.randn(*input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
        print(f"✓ Model exported to ONNX: {output_path}")
    
    @staticmethod
    def to_torchscript(
        model: nn.Module,
        output_path: Path,
        scripted: bool = True,
    ) -> None:
        """
        Export model to TorchScript.
        
        Args:
            model: PyTorch model
            output_path: where to save model
            scripted: use scripting (True) or tracing (False)
        """
        model.eval()
        
        if scripted:
            traced = torch.jit.script(model)
        else:
            dummy_input = torch.randn(1, 3, 224, 224)
            traced = torch.jit.trace(model, dummy_input)
        
        traced.save(output_path)
        print(f"✓ Model exported to TorchScript: {output_path}")
    
    @staticmethod
    def quantize(model: nn.Module) -> nn.Module:
        """Apply quantization for faster inference."""
        model_quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        return model_quantized
