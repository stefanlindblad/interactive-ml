node {
  name: "input"
  op: "Placeholder"
  device: "/device:GPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: -1
        }
        dim {
          size: -1
        }
        dim {
          size: 4
        }
      }
    }
  }
}
node {
  name: "InteractiveNormalsInput"
  op: "InteractiveNormalsInput"
  input: "input"
  device: "/device:GPU:0"
}
node {
  name: "adjust_contrast/Identity"
  op: "Identity"
  input: "InteractiveNormalsInput"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "adjust_contrast/contrast_factor"
  op: "Const"
  device: "/device:GPU:0"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.5
      }
    }
  }
}
node {
  name: "adjust_contrast"
  op: "AdjustContrastv2"
  input: "adjust_contrast/Identity"
  input: "adjust_contrast/contrast_factor"
  device: "/device:GPU:0"
}
node {
  name: "adjust_contrast/Identity_1"
  op: "Identity"
  input: "adjust_contrast"
  device: "/device:GPU:0"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
node {
  name: "InteractiveOutput"
  op: "InteractiveOutput"
  input: "adjust_contrast/Identity_1"
  device: "/device:GPU:0"
}
versions {
  producer: 24
}
