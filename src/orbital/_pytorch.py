"""Parse PyTorch models through the torch ONNX exporter."""

from typing import cast

import torch

from .ast import EnsureConcatenatedInputs, ParsedPipeline
from .types import FeaturesTypes


def parse_pytorch_model(
    model: torch.nn.Module, features: FeaturesTypes
) -> ParsedPipeline:
    """Parse a PyTorch model into a [orbital.ast.ParsedPipeline][].

    :param model: The trained PyTorch model to parse
    :param features: Mapping of column names to their [orbital.types.ColumnType][] objects
    """
    non_passthrough_features = {
        fname: ftype for fname, ftype in features.items() if not ftype.is_passthrough
    }

    if not non_passthrough_features:
        raise ValueError(
            "All provided features are passthrough. "
            "The model would not do anything useful."
        )

    concatenated_inputs = EnsureConcatenatedInputs(non_passthrough_features)
    # A neural network consumes a single input tensor, so mixed feature
    # types cannot be concatenated. Raises a clear error on mixed types.
    concatenated_inputs.uniform_type()

    # The exporter traces one forward pass to record the operations graph:
    # the dummy input's values are discarded, only shape/dtype/device matter.
    # dtype/device must match the model's, otherwise tracing fails for
    # non-float32 or GPU-resident models. Parameterless models fall back to
    # torch defaults.
    param = next(model.parameters(), None)
    if param is None:
        dummy_input = torch.zeros(1, len(non_passthrough_features))
    else:
        dummy_input = torch.zeros(
            1,
            len(non_passthrough_features),
            dtype=param.dtype,
            device=param.device,
        )
    # The exporter traces the model as-is, but the trace must capture eval
    # behavior: SQL always runs inference, and train-mode ops (e.g. Dropout
    # random masking) cannot be translated. Models are commonly left in
    # training mode after training (torch's default), so instead of asking
    # callers to eval() first, flip the model ourselves and restore its
    # exact state afterwards. eval()/train() recurse into every submodule,
    # so snapshot each module's own flag: mixed states like the fine-tuning
    # freeze pattern (frozen submodules in eval) must survive the restore.
    training_flags = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        onnx_program = torch.onnx.export(
            model,
            (dummy_input,),
            input_names=["input"],
            # Pin the lowest opset the dynamo exporter supports so the
            # emitted graph does not change with the installed torch version.
            opset_version=18,
            dynamo=True,
            # The exporter prints conversion progress by default,
            # a library function must stay silent.
            verbose=False,
        )
    finally:
        # eval() mutated the caller's model. If the flags were not restored,
        # a user resuming training afterwards would silently train with
        # eval behavior (e.g. Dropout disabled, BatchNorm stats frozen).
        for module, was_training in training_flags:
            module.training = was_training
    # export() is typed Optional only because the legacy exporter could
    # return None, with dynamo=True a program is always returned.
    # The cast only informs mypy, it has no runtime effect.
    onnx_model = cast(torch.onnx.ONNXProgram, onnx_program).model_proto

    # The network expects a single concatenated tensor, while SQL provides
    # individual columns. Inject a Concat step to bridge the two.
    onnx_model = concatenated_inputs.inject_concat_step(onnx_model)
    return ParsedPipeline._from_onnx_model(onnx_model, features)
