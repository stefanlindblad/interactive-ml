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
  name: "InteractiveDepthInput"
  op: "InteractiveDepthInput"
  input: "input"
  device: "/device:GPU:0"
}
node {
  name: "InteractiveNormalsInput"
  op: "InteractiveNormalsInput"
  input: "InteractiveDepthInput"
  device: "/device:GPU:0"
}
node {
  name: "InteractiveOutput"
  op: "InteractiveOutput"
  input: "InteractiveNormalsInput"
  device: "/device:GPU:0"
}
versions {
  producer: 24
}
