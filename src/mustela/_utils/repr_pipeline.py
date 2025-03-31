import typing

import onnx as _onnx

from .onnx import get_attr_value, get_initializer_data

if typing.TYPE_CHECKING:
    from ..ast import ParsedPipeline


class ParsedPipelineStr:
    def __init__(self, pipeline: "ParsedPipeline", maxlen: int = 80) -> None:
        self._maxlen = maxlen
        self._pipeline = pipeline
        self._constants: dict[str, typing.Any] = {
            init.name: get_initializer_data(init)
            for init in self._pipeline._model.graph.initializer
        }

    def __str__(self) -> str:
        """Generate a string representation of the pipeline."""
        return f"""{self._pipeline.__class__.__name__}(
  features={{\n{self._features_str()}\n  }},
  steps=[\n{self._steps_str()}\n  ],
)
"""

    def _features_str(self) -> str:
        """Generate a string representation of the features."""
        return "\n".join(
            (
                f"    {feature_name}: {feature_type}"
                for feature_name, feature_type in self._pipeline.features.items()
            )
        )

    def _steps_str(self) -> str:
        """Generate a string representation of the pipeline steps."""
        return "\n".join(
            (self._node_str(node) for node in self._pipeline._model.graph.node)
        )

    def _node_str(self, node: _onnx.NodeProto) -> str:
        """Generate a string representation of a pipeline step."""
        return f"""    {self._varnames(None, node.output)}={node.op_type}(
      inputs: {self._varnames(node, node.input)},
      attributes: {self._attributes(node.attribute)}
    )"""

    def _varnames(
        self, node: _onnx.NodeProto | None, varlist: typing.Iterable[str]
    ) -> str:
        """Generate a string representation of a list of variables or constants."""

        def _var_value(var: str) -> str:
            if var in self._constants:
                return self._shorten(f"{var}={self._constants[var]}")
            return f"{var}"

        return ", ".join((f"{_var_value(var)}" for var in varlist))

    def _attributes(self, attributes: typing.Iterable[_onnx.AttributeProto]) -> str:
        """Generate a string representation of a list of attributes."""

        def _attr_value(attr: _onnx.AttributeProto) -> str:
            return self._shorten(str(get_attr_value(attr)))

        indent = "\n        "
        content = indent.join(
            (f"{attr.name}={_attr_value(attr)}" for attr in attributes)
        )
        if content.strip():
            return f"{indent}{content}"
        else:
            return ""

    def _shorten(self, value: str) -> str:
        """Shorten a string to maxlen characters."""
        if self._maxlen and len(value) > self._maxlen:
            return f"{value[: self._maxlen]}..."
        return value
