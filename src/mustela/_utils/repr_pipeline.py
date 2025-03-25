from .onnx import get_attr_value, get_initializer_data


class ParsedPipelineStr:
    def __init__(self, pipeline: "ParsedPipeline", maxlen: int = 80) -> None:
        self._maxlen = maxlen
        self._pipeline = pipeline
        self._constants = {init.name: get_initializer_data(init) for init in self._pipeline._model.graph.initializer}
   
    def __str__(self) -> str:
        """Generate a string representation of the pipeline."""
        return f""""{self._pipeline.__class__.__name__}(
  features={{\n{self._features_str()}\n  }},
  steps=[\n{self._graph_str()}\n  ],
)
"""

    def _features_str(self) -> str:
        """Generate a string representation of the features."""
        return"\n".join((f"    {feature_name}: {feature_type}" for feature_name, feature_type in self._pipeline.features.items()))
    
    def _graph_str(self) -> str:
        """Generate a string representation of the pipeline graph."""
        return "\n".join((self._node_str(node) for node in self._pipeline._model.graph.node))

    def _node_str(self, node: "Node") -> str:
        """Generate a string representation of a pipeline step."""
        return f"""    {self._varnames(None, node.output)}={node.op_type}(
      inputs: {self._varnames(node, node.input)},
      attributes: {self._attributes(node.attribute)}
    )"""

    def _varnames(self, node, varlist) -> str:
        """Generate a string representation of a list of variables."""
        def _var_value(var):
            if var in self._constants:
                return self._shorten(f"{var}={self._constants[var]}")
            return f"{var}"
        return ", ".join((f"{_var_value(var)}" for var in varlist))

    def _attributes(self, attributes) -> str:
        """Generate a string representation of a list of attributes."""
        def _attr_value(attr):
            return self._shorten(str(get_attr_value(attr)))
        return ", ".join((f"{attr.name}={_attr_value(attr)}" for attr in attributes))

    def _shorten(self, value: str) -> str:
        """Shorten a string to 80 characters."""
        if self._maxlen and len(value) > self._maxlen:
            return f"{value[:self._maxlen]}..."
        return value