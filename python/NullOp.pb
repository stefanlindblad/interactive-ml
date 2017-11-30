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
  name: "InteractiveInput"
  op: "InteractiveInput"
  input: "input"
  device: "/device:GPU:0"
}
node {
  name: "InteractiveOutput"
  op: "InteractiveOutput"
  input: "InteractiveInput"
  device: "/device:GPU:0"
}
versions {
  producer: 24
}
